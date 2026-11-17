"""High-level context assembly.

Replaces the legacy ``history[-N:]`` slicing with a token-budget aware
pipeline that assembles the final ``[{"role": ..., "content": ...}]`` list
fed to the LLM.

Design decisions
----------------
- Mandatory sections (``system_prompt``, ``current_question``) are never
  dropped; if total budget is too small to fit them too, the allocator still
  keeps them (and the LLM call may fail upstream — we surface this clearly
  via ``dropped_sections`` so the caller can react).
- ``recent_history`` is greedy: we walk turns from the most recent backwards
  and stop as soon as the next turn would exceed the *remaining* budget after
  all other sections are accounted for. Each turn (user+assistant) is kept
  whole — never half a turn — to avoid corrupting the conversation.
- ``photo_context`` has its own dedicated section with an independent quota,
  so photo descriptions no longer steal budget from history or retrieved docs.
  The photo data is also **compressed** by the Agent before reaching the
  ContextManager — URLs, VLM scores, and match scores are stripped.
- Ordering of the returned messages reflects the natural conversation shape
  expected by chat APIs:
      [system, summary?, long_term?, profile?, docs?, photos?, strategy?, ...history, user]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.context.budget import (
    AllocationResult,
    BudgetAllocator,
    ContextSection,
    TruncateMode,
)
from app.context.tokenizer import Tokenizer, get_default_tokenizer
from app.core.logging import get_logger
from app.core.settings import MemorySettings, get_settings

_logger = get_logger(__name__)


# ----------------------------- Inputs / Outputs -----------------------------

@dataclass
class HistoryTurn:
    """One round of chat history (user + assistant)."""
    user: str = ""
    assistant: str = ""

    def render(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self.user:
            out.append({"role": "user", "content": self.user})
        if self.assistant:
            out.append({"role": "assistant", "content": self.assistant})
        return out


@dataclass
class BuildResult:
    messages: List[Dict[str, str]]
    stats: Dict[str, Any]


# ----------------------------- Manager --------------------------------------

class ContextManager:
    """Assembles a chat-completion message list under a global token budget."""

    SECTION_SYSTEM = "system"
    SECTION_QUESTION = "question"
    SECTION_STRATEGY = "strategy"
    SECTION_DOCS = "docs"
    SECTION_PHOTOS = "photos"
    SECTION_PROFILE = "profile"
    SECTION_LONG_TERM = "long_term"
    SECTION_SUMMARY = "summary"
    SECTION_HISTORY = "history"  # virtual; per-turn handling differs

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        memory_settings: Optional[MemorySettings] = None,
    ) -> None:
        self._tok = tokenizer or get_default_tokenizer()
        self._mem = memory_settings or get_settings().memory

    # ------------------------------------------------------------------
    def build(
        self,
        *,
        system_prompt: str,
        question: str,
        history: Sequence[HistoryTurn] = (),
        retrieved_docs: str = "",
        photo_context: str = "",
        strategy_text: str = "",
        user_profile: str = "",
        long_term_memory: str = "",
        session_summary: str = "",
        budget_override: Optional[int] = None,
    ) -> BuildResult:
        budget = budget_override or self._mem.max_input_tokens

        # 1. Build the static (non-history) sections under the global allocator
        sections: List[ContextSection] = [
            ContextSection(
                name=self.SECTION_SYSTEM,
                content=system_prompt,
                droppable=False,
                priority=1000,
            ),
            ContextSection(
                name=self.SECTION_QUESTION,
                content=question,
                droppable=False,
                priority=999,
            ),
        ]
        if strategy_text:
            sections.append(ContextSection(
                name=self.SECTION_STRATEGY,
                content=strategy_text,
                quota=self._mem.quota_strategy,
                priority=600,
                truncate=TruncateMode.TAIL,
            ))
        if retrieved_docs:
            sections.append(ContextSection(
                name=self.SECTION_DOCS,
                content=retrieved_docs,
                quota=self._mem.quota_docs,
                priority=700,
                truncate=TruncateMode.TAIL,
            ))
        if photo_context:
            sections.append(ContextSection(
                name=self.SECTION_PHOTOS,
                content=photo_context,
                quota=self._mem.quota_photos,
                priority=650,
                truncate=TruncateMode.TAIL,
            ))
        if user_profile:
            sections.append(ContextSection(
                name=self.SECTION_PROFILE,
                content=user_profile,
                quota=self._mem.quota_profile,
                priority=400,
                truncate=TruncateMode.TAIL,
            ))
        if long_term_memory:
            sections.append(ContextSection(
                name=self.SECTION_LONG_TERM,
                content=long_term_memory,
                quota=self._mem.quota_long_term,
                priority=300,
                truncate=TruncateMode.TAIL,
            ))
        if session_summary:
            sections.append(ContextSection(
                name=self.SECTION_SUMMARY,
                content=session_summary,
                quota=self._mem.quota_summary,
                priority=500,
                truncate=TruncateMode.TAIL,
            ))

        allocator = BudgetAllocator(self._tok, budget)
        alloc: AllocationResult = allocator.allocate(sections)

        # 2. Greedy fit history into whatever budget is left
        remaining = max(0, budget - alloc.tokens_used)
        kept_turns, history_tokens = self._fit_history(history, remaining)

        # 3. Assemble messages in canonical order
        kept_by_name = {s.name: s for s in alloc.sections}
        messages: List[Dict[str, str]] = []

        if self.SECTION_SYSTEM in kept_by_name:
            messages.append({"role": "system", "content": kept_by_name[self.SECTION_SYSTEM].content})

        # Memory-style additions live as auxiliary system messages so they
        # don't masquerade as user input.
        for aux in (self.SECTION_SUMMARY, self.SECTION_LONG_TERM, self.SECTION_PROFILE):
            sec = kept_by_name.get(aux)
            if sec and sec.content.strip():
                messages.append({
                    "role": "system",
                    "content": _label_section(aux, sec.content),
                })

        sec_docs = kept_by_name.get(self.SECTION_DOCS)
        if sec_docs and sec_docs.content.strip():
            messages.append({
                "role": "system",
                "content": _label_section(self.SECTION_DOCS, sec_docs.content),
            })

        sec_photos = kept_by_name.get(self.SECTION_PHOTOS)
        if sec_photos and sec_photos.content.strip():
            messages.append({
                "role": "system",
                "content": _label_section(self.SECTION_PHOTOS, sec_photos.content),
            })

        sec_strategy = kept_by_name.get(self.SECTION_STRATEGY)
        if sec_strategy and sec_strategy.content.strip():
            messages.append({
                "role": "system",
                "content": _label_section(self.SECTION_STRATEGY, sec_strategy.content),
            })

        for turn in kept_turns:
            messages.extend(turn.render())

        # current_question is always the final user message
        if self.SECTION_QUESTION in kept_by_name:
            messages.append({"role": "user", "content": kept_by_name[self.SECTION_QUESTION].content})

        # 4. Stats
        per_section_tokens = {s.name: s.tokens for s in alloc.sections}
        per_section_tokens[self.SECTION_HISTORY] = history_tokens
        stats = {
            "tokens_used": alloc.tokens_used + history_tokens,
            "budget": budget,
            "dropped_sections": alloc.dropped_sections + (
                [self.SECTION_HISTORY] if not kept_turns and history else []
            ),
            "truncated_sections": alloc.truncated_sections,
            "per_section_tokens": per_section_tokens,
            "history_turns_kept": len(kept_turns),
            "history_turns_total": len(list(history)),
        }
        return BuildResult(messages=messages, stats=stats)

    # ------------------------------------------------------------------
    def _fit_history(
        self,
        history: Sequence[HistoryTurn],
        remaining: int,
    ) -> "tuple[List[HistoryTurn], int]":
        if remaining <= 0 or not history:
            return [], 0
        # Walk from the most recent turn backwards.
        kept: List[HistoryTurn] = []
        used = 0
        for turn in reversed(list(history)):
            t = self._tok.count(turn.user) + self._tok.count(turn.assistant)
            if t == 0:
                continue
            if used + t > remaining:
                break
            kept.append(turn)
            used += t
        kept.reverse()
        return kept, used


# ----------------------------- Helpers --------------------------------------

_LABELS = {
    ContextManager.SECTION_SUMMARY: "【会话摘要】",
    ContextManager.SECTION_LONG_TERM: "【跨会话记忆】",
    ContextManager.SECTION_PROFILE: "【用户画像】",
    ContextManager.SECTION_DOCS: "【参考资料】",
    ContextManager.SECTION_PHOTOS: "【照片描述】",
    ContextManager.SECTION_STRATEGY: "【策略提示】",
}


def _label_section(name: str, content: str) -> str:
    label = _LABELS.get(name, "")
    return f"{label}\n{content}".strip() if label else content
