"""函数执行模块 — 通过 SkillRegistry 分派并收集结果。

从 ``agent/gem_agent.py`` 拆分出来。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.skills import FunctionResult

logger = logging.getLogger(__name__)


def execute_functions(
    *,
    skill_registry,
    function_calls: List[Dict[str, Any]],
    rewritten_question: str,
    original_question: str,
    extract_photo_count_fn,
) -> List["FunctionResult"]:
    """Execute all planned function calls via the SkillRegistry.

    返回的 FunctionResult 列表中，每个元素附带了 skill_name / function_name
    / success / data / error 字段，供下游（responder / caller）消费。
    """
    if not function_calls:
        logger.info("No functions to call — direct answer mode")
        return []

    results: List["FunctionResult"] = []
    for fc in function_calls:
        func_name = fc.get("tool", "")
        arguments = fc.get("arguments", {})

        # Ensure required 'query' param is present for retrieval functions
        if func_name in ("search_knowledge_base", "search_concert_schedule"):
            if "query" not in arguments or not arguments["query"]:
                arguments["query"] = rewritten_question
                logger.warning(
                    "Function '%s' missing 'query' arg, auto-filled with: %s",
                    func_name,
                    rewritten_question,
                )
            else:
                original_arg = arguments["query"]
                arguments["query"] = (
                    rewritten_question
                    if rewritten_question != original_question
                    else original_arg
                )

        # 歌曲搜索同样的保护逻辑
        if func_name == "search_song_info":
            if "song_name" not in arguments or not arguments["song_name"]:
                arguments["song_name"] = original_question
                logger.warning(
                    "Function '%s' missing 'song_name' arg, auto-filled with: %s",
                    func_name,
                    original_question,
                )

        # 图片搜索：从原始问题强制提取数量词，覆盖 LLM 可能给的错误 k
        if func_name == "search_gem_photos":
            forced_k = extract_photo_count_fn(original_question)
            if forced_k is not None:
                if arguments.get("k") != forced_k:
                    logger.info(
                        "search_gem_photos: forcing k=%d from question (LLM gave k=%s)",
                        forced_k,
                        arguments.get("k"),
                    )
                arguments["k"] = forced_k

        result = skill_registry.invoke(func_name, arguments)
        results.append(result)

        if result.success:
            logger.info(
                "Function '%s' (skill '%s') succeeded (%d chars)",
                func_name,
                result.skill_name,
                len(result.data),
            )
        else:
            logger.warning(
                "Function '%s' (skill '%s') failed: %s",
                func_name,
                result.skill_name,
                result.error,
            )

    return results


def extract_photo_payload(
    results: List["FunctionResult"],
) -> Optional[Dict[str, Any]]:
    """Pull the JSON payload from a successful ``search_gem_photos`` call.

    Returns ``None`` if no photo skill result is present (or it failed),
    which keeps the SSE stream completely untouched for non-photo queries.

    重要：返回给前端的 payload **只保留 url**，不下发 captionZh / tags / dims。
    这些字段是给主干 LLM 在 context 里"看图"用的内部信号，前端不该感知。
    """
    if not results:
        return None
    import json as _json

    for r in results:
        if r.skill_name != "photo" or not r.success:
            continue
        try:
            obj = _json.loads(r.data) if r.data else None
        except Exception:  # noqa: BLE001
            logger.warning("photo_skill returned non-JSON payload: %r", r.data[:120] if r.data else "")
            return None
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            # 仅暴露最小必要字段给前端
            slim_items = [
                {"url": item["url"]}
                for item in obj["items"]
                if isinstance(item, dict) and item.get("url")
            ]
            return {"items": slim_items}
    return None
