"""User-profile extractor.

Triggered every ``PROFILE_UPDATE_EVERY_N_TURNS`` rounds. Calls a fast LLM to
distill the user's recent conversation into a structured ``preferences`` JSON
plus a short narrative ``summary``. Persisted via the LongTermStore.

This module is best-effort: any failure is logged at WARNING level and the
caller continues without updating the profile.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.core.logging import get_logger
from app.memory.stores.base import LongTermStore, StoredMessage, UserProfile

_logger = get_logger(__name__)


_SYSTEM = (
    "你是用户画像抽取助手。请阅读下面的多轮对话，输出严格的 JSON："
    "{\"preferences\": {key: value}, \"summary\": \"<=200字\"}。"
    "preferences 中只包含明确表达的喜好/事实/偏好（如 favorite_song, favorite_album,"
    " hometown, age_range, mood 等）；不存在则省略对应键。"
    "不要输出 JSON 之外的任何文字。"
)


class ProfileExtractor:
    def __init__(
        self,
        *,
        long_term_store: LongTermStore,
        invoke: Callable[[List[dict]], str],
        every_n_turns: int = 10,
    ) -> None:
        self._lt = long_term_store
        self._invoke = invoke
        self._every_n = max(1, every_n_turns)

    def should_run(self, total_user_turns: int) -> bool:
        return total_user_turns > 0 and total_user_turns % self._every_n == 0

    async def maybe_update(
        self,
        user_id: str,
        recent_messages: Sequence[StoredMessage],
    ) -> Optional[UserProfile]:
        if not recent_messages:
            return None
        try:
            previous = await self._lt.get_profile(user_id)
        except Exception as e:  # noqa: BLE001
            _logger.warning("extractor.get_profile_failed", error=str(e))
            previous = None

        rendered = "\n".join(
            f"{m.role}: {m.content}" for m in recent_messages if m.content
        )
        prompt: List[dict] = [{"role": "system", "content": _SYSTEM}]
        if previous and (previous.preferences or previous.summary):
            prompt.append({
                "role": "system",
                "content": "已有画像（可在此基础上更新或修正）：" + json.dumps(
                    {"preferences": previous.preferences, "summary": previous.summary},
                    ensure_ascii=False,
                ),
            })
        prompt.append({"role": "user", "content": rendered})

        try:
            text = self._invoke(prompt)
        except Exception as e:  # noqa: BLE001
            _logger.warning("extractor.invoke_failed", error=str(e))
            return None

        parsed = _parse_json(text)
        if parsed is None:
            _logger.warning("extractor.parse_failed", raw=text[:200])
            return None

        prefs = parsed.get("preferences") or {}
        summary = (parsed.get("summary") or "").strip()
        if not isinstance(prefs, dict):
            prefs = {}
        # Merge with previous prefs (new wins)
        merged = dict((previous.preferences if previous else {}) or {})
        merged.update({k: v for k, v in prefs.items() if v is not None and v != ""})

        profile = UserProfile(
            user_id=user_id,
            preferences=merged,
            summary=summary[:500] if summary else (previous.summary if previous else ""),
        )
        try:
            await self._lt.upsert_profile(profile)
        except Exception as e:  # noqa: BLE001
            _logger.warning("extractor.upsert_failed", error=str(e))
            return None
        return profile


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Strip code fences if the model wrapped its output
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
        text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    # Tolerate trailing prose by extracting the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            return None
    return None
