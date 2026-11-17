"""
PhotoSkill — semantic photo retrieval over the curated GEM photo pool.

Functions:
    • search_gem_photos — return top-K photos matching the user's query,
      based on tags + note BM25 over the in-memory index, re-ranked by
      VLM score.

Dependencies (injected via ``configure``):
    • photo_repo — :class:`PhotoRepository`
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TYPE_CHECKING

from agent.skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from rag_modules.photo_repo import PhotoRepository

logger = logging.getLogger(__name__)


class PhotoSkill(BaseSkill):
    """Skill for retrieving curated GEM photos by semantic / tag query."""

    def __init__(self) -> None:
        super().__init__()
        self._repo: Optional["PhotoRepository"] = None
        # Default Top-K when caller does not specify
        self._default_k: int = 3
        # Hard cap to avoid the LLM asking for absurdly many
        self._max_k: int = 6

    # ---- BaseSkill interface ----------------------------------------

    @property
    def name(self) -> str:
        return "photo"

    @property
    def description(self) -> str:
        return (
            "Curated photo retrieval skill — finds GEM photos that match "
            "the user's described scene / outfit / mood, returning CDN URLs."
        )

    def configure(self, **kwargs: Any) -> None:
        """
        Required kwargs:
            photo_repo — PhotoRepository
        Optional kwargs:
            default_k — default number of photos to return (default 3)
            max_k — hard cap on the number of photos (default 6)
        """
        self._repo = kwargs["photo_repo"]
        self._default_k = int(kwargs.get("default_k", 3))
        self._max_k = int(kwargs.get("max_k", 6))
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="search_gem_photos",
            description=(
                "ONLY call this when the user explicitly asks to SEE photos of GEM "
                "(e.g. '发张图', '给我看张照片', '有没有 XX 的图', '来张演唱会的图'). "
                "DO NOT call this for normal Q&A. "
                "Two modes:\n"
                "  1) USER HAS A CLEAR PREFERENCE — set `query` to Chinese keywords "
                "describing the wanted scene/outfit/mood, e.g. '演唱会 舞台', '红裙', "
                "'自拍 笑容'. The skill will do hybrid retrieval (BM25 + semantic).\n"
                "  2) USER HAS NO CLEAR PREFERENCE — leave `query` empty (\"\" or "
                "omit it). Examples: '来一张你的照片', '随便发张图', '给我看看'. "
                "The skill will return random photos from the pool.\n"
                "If the user explicitly mentions a number (一张/1张=1, 两张/2张=2, "
                "几张=3, 多发几张/五张=5), set `k` to that integer; "
                "otherwise omit `k` to use the default."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Chinese keywords describing the wanted photo "
                                       "when the user has a clear preference, "
                                       "e.g. '演唱会 舞台', '红裙', '自拍 笑容'. "
                                       "Leave empty (\"\") when the user has no preference "
                                       "(e.g. '随便发张', '给我一张').",
                    },
                    "k": {
                        "type": "integer",
                        "description": f"How many photos to return (1-{self._max_k}). "
                                       f"Default {self._default_k} when omitted. "
                                       f"Map user's number words: 一张=1, 两张=2, 几张=3, 五张=5.",
                    },
                },
                "required": [],
            },
            func=self._search_gem_photos,
        )

    # ---- Function implementation ------------------------------------

    def _search_gem_photos(self, query: str = "", k: int = 0) -> str:
        if self._repo is None:
            return json.dumps({"items": [], "error": "photo repo not initialised"},
                              ensure_ascii=False)

        # Clamp k
        if k <= 0:
            k = self._default_k
        k = max(1, min(k, self._max_k))

        # 模式 1：query 为空（用户无明确倾向）→ 全池随机抽 k 张
        # 这是 "来一张你的照片" / "随便发张图" 这类问法的标准处理
        if not query or not query.strip():
            try:
                hits = self._repo.random_pick(k=k)
            except Exception as exc:  # noqa: BLE001
                logger.error("photo random_pick failed: %s", exc, exc_info=True)
                return json.dumps({"items": [], "error": str(exc)}, ensure_ascii=False)
            logger.info("search_gem_photos: random mode k=%d picked=%d", k, len(hits))
        else:
            # 模式 2：有 query → 召回 2k 候选，再随机抽 k 张
            # 既保留语义匹配的相关性，又给同 query 多次提问留多样性空间
            fetch_k = min(k * 2, self._max_k * 2)
            try:
                candidates = self._repo.search(query.strip(), top_k=fetch_k)
            except Exception as exc:  # noqa: BLE001
                logger.error("photo search failed: %s", exc, exc_info=True)
                return json.dumps({"items": [], "error": str(exc)}, ensure_ascii=False)
            if len(candidates) > k:
                import random
                hits = random.sample(candidates, k)
            else:
                hits = candidates
            logger.info(
                "search_gem_photos: query=%r k=%d candidates=%d picked=%d",
                query, k, len(candidates), len(hits),
            )

        # Compact, JSON-serialised payload — the agent will pass these
        # photos through to the SSE stream as a `photos` event.
        items = [
            {
                "url": h["url"],
                "score": h["score"],
                "note": h["note"],
                "captionZh": h.get("captionZh", ""),
                "tags": h["tags"],
                "dims": h.get("dims", {}),
                "match": round(h["match_score"], 4),
            }
            for h in hits
        ]
        return json.dumps({"items": items}, ensure_ascii=False)
