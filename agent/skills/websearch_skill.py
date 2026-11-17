"""
WebSearchSkill — real-time web search for time-sensitive questions about GEM.

Functions:
    • web_search_gem — query the public internet (Tavily) for fresh info

Dependencies:
    - TAVILY_API_KEY (env var)
    - requests
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import requests

from agent.skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_DEFAULT_QUERY_SUFFIX = " 邓紫棋 G.E.M."


class WebSearchSkill(BaseSkill):
    """Skill that performs live web search for up-to-date GEM information."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Live web search skill — fetches real-time information about "
            "GEM (邓紫棋) from the public internet when the local knowledge "
            "base is outdated (e.g. latest songs, recent news, upcoming shows)."
        )

    def configure(
        self,
        api_key: Optional[str],
        max_results: int = 5,
        search_depth: str = "basic",
        timeout: float = 15.0,
        query_suffix: str = _DEFAULT_QUERY_SUFFIX,
        log_file: Optional[str] = "logs/web_search.log",
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._search_depth = search_depth
        self._timeout = timeout
        self._query_suffix = query_suffix
        self._log_file = log_file
        self._enabled = bool(api_key)
        if not self._enabled:
            logger.warning(
                "WebSearchSkill disabled — TAVILY_API_KEY not configured"
            )
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="web_search_gem",
            description=(
                "Search the public internet for the LATEST information about "
                "GEM / 邓紫棋 that is NOT in the local knowledge base. "
                "Use this ONLY when the question clearly needs fresh, real-time "
                "data, e.g. 'GEM 最近在干嘛', '最新动态', '新歌发布', "
                "'今年的演唱会安排', '刚刚上了什么综艺', 近期新闻/微博/活动等. "
                "Do NOT use it for lyrics, biography, or historical facts "
                "— those live in the local RAG."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search query. Keep it concise "
                            "and include the time hint if the user asked for "
                            "a specific period (e.g. '邓紫棋 2026 新歌')."
                        ),
                    }
                },
                "required": ["query"],
            },
            func=self._web_search,
        )

    # ---- implementation ---------------------------------------------

    def _web_search(self, query: str) -> str:
        if not self._enabled:
            return "网络检索未启用（未配置 TAVILY_API_KEY），无法获取实时信息。"

        # Auto-append topic suffix if the query does not already mention GEM
        q = query.strip()
        if "邓紫棋" not in q and "gem" not in q.lower():
            q = f"{q}{self._query_suffix}"

        payload = {
            "api_key": self._api_key,
            "query": q,
            "search_depth": self._search_depth,
            "max_results": self._max_results,
            "include_answer": True,
        }

        try:
            resp = requests.post(
                _TAVILY_ENDPOINT, json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Tavily request failed: %s", exc)
            self._dump_log(q, None, error=str(exc))
            return f"实时检索失败：{exc}"
        except ValueError as exc:
            logger.error("Tavily returned non-JSON: %s", exc)
            self._dump_log(q, None, error=str(exc))
            return "实时检索失败：返回格式异常。"

        formatted = self._format_results(q, data)
        logger.info(
            "[web_search_gem] query=%s | results=%d | answer_len=%d",
            q,
            len(data.get("results") or []),
            len((data.get("answer") or "")),
        )
        self._dump_log(q, data, formatted=formatted)
        return formatted

    def _dump_log(
        self,
        query: str,
        data: Optional[dict],
        formatted: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Append one JSON line per call to log_file for later inspection."""
        if not self._log_file:
            return
        try:
            path = Path(self._log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "error": error,
                "raw": data,
                "formatted": formatted,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # logging must never break the pipeline
            logger.warning("failed to dump web_search log: %s", exc)

    @staticmethod
    def _format_results(query: str, data: dict) -> str:
        answer: str = (data.get("answer") or "").strip()
        results: List[dict] = data.get("results") or []

        if not answer and not results:
            return f"未搜索到与「{query}」相关的实时信息。"

        lines: List[str] = [f"[实时检索 · query={query}]"]
        if answer:
            lines.append(f"摘要：{answer}")

        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip().replace("\n", " ")
            if len(content) > 400:
                content = content[:400] + "…"
            lines.append(f"{i}. {title}\n   {content}\n   来源：{url}")

        return "\n".join(lines)
