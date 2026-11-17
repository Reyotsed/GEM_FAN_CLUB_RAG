"""Per-section token budgeting and truncation strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from app.context.tokenizer import Tokenizer


class TruncateMode(str, Enum):
    """How to shrink a section that exceeds its quota."""

    TAIL = "tail"           # keep prefix, drop suffix (e.g. retrieved docs)
    HEAD = "head"           # keep suffix, drop prefix
    DROP_WHOLE = "drop"     # if over quota, drop the section entirely


@dataclass
class ContextSection:
    """One logical block of context (system prompt, docs, history, ...).

    ``content`` may be empty after truncation; callers should skip empty
    sections when assembling messages.
    """
    name: str
    content: str
    quota: Optional[int] = None      # max tokens; None = no per-section cap
    priority: int = 100              # higher = more likely to keep
    droppable: bool = True
    truncate: TruncateMode = TruncateMode.TAIL
    tokens: int = 0                  # filled by the allocator


@dataclass
class AllocationResult:
    sections: List[ContextSection]
    tokens_used: int
    budget: int
    dropped_sections: List[str] = field(default_factory=list)
    truncated_sections: List[str] = field(default_factory=list)


class BudgetAllocator:
    """Allocates a global token budget across heterogeneous sections.

    Algorithm:
      1. Mandatory (``not droppable``) sections are kept verbatim. If their
         combined size already exceeds the budget the allocator still keeps
         them — caller must size them sanely (system prompt + question are
         usually << budget).
      2. Each remaining section is first truncated to its own quota using
         the configured ``TruncateMode``.
      3. Sections are then admitted in priority order (highest first). Each
         admitted section consumes from the remaining budget; if it doesn't
         fit, the allocator either truncates further (TAIL/HEAD) or drops it
         (DROP_WHOLE / no remaining room).
    """

    def __init__(self, tokenizer: Tokenizer, budget: int) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self._tok = tokenizer
        self._budget = budget

    # ------------------------------------------------------------------
    def allocate(self, sections: List[ContextSection]) -> AllocationResult:
        # Step 1: per-section quota truncation
        for s in sections:
            s.tokens = self._tok.count(s.content)
            if s.quota is not None and s.tokens > s.quota:
                s.content = self._truncate_to(s.content, s.quota, s.truncate)
                s.tokens = self._tok.count(s.content)

        # Step 2: keep mandatory sections first
        kept: List[ContextSection] = []
        used = 0
        dropped: List[str] = []
        truncated: List[str] = []

        mandatory = [s for s in sections if not s.droppable]
        for s in mandatory:
            kept.append(s)
            used += s.tokens

        # Step 3: admit droppable sections by priority
        remaining_pool = sorted(
            (s for s in sections if s.droppable),
            key=lambda x: x.priority,
            reverse=True,
        )

        for s in remaining_pool:
            free = self._budget - used
            if free <= 0:
                dropped.append(s.name)
                continue
            if s.tokens <= free:
                kept.append(s)
                used += s.tokens
                continue
            # Doesn't fit: try further truncation if mode permits
            if s.truncate is TruncateMode.DROP_WHOLE:
                dropped.append(s.name)
                continue
            shrunk = self._truncate_to(s.content, free, s.truncate)
            shrunk_tokens = self._tok.count(shrunk)
            if shrunk_tokens <= 0 or not shrunk.strip():
                dropped.append(s.name)
                continue
            s.content = shrunk
            s.tokens = shrunk_tokens
            kept.append(s)
            used += shrunk_tokens
            truncated.append(s.name)

        # Preserve original logical order for downstream rendering
        order = {sec.name: i for i, sec in enumerate(sections)}
        kept.sort(key=lambda x: order.get(x.name, 0))

        return AllocationResult(
            sections=kept,
            tokens_used=used,
            budget=self._budget,
            dropped_sections=dropped,
            truncated_sections=truncated,
        )

    # ------------------------------------------------------------------
    def _truncate_to(self, text: str, max_tokens: int, mode: TruncateMode) -> str:
        """Binary-search a substring whose token count fits ``max_tokens``.

        Operates on characters; for our heuristic + tiktoken usage this is
        precise enough. Always preserves whole UTF-8 codepoints because Python
        strings are unicode.
        """
        if max_tokens <= 0:
            return ""
        if self._tok.count(text) <= max_tokens:
            return text

        lo, hi = 0, len(text)
        # Find the largest character window <= max_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = (
                text[:mid] if mode is TruncateMode.TAIL else text[-mid:]
            )
            if self._tok.count(candidate) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            return ""
        if mode is TruncateMode.TAIL:
            kept = text[:lo].rstrip()
            return kept + ("…" if len(kept) < len(text) else "")
        # HEAD
        kept = text[-lo:].lstrip()
        return ("…" if len(kept) < len(text) else "") + kept
