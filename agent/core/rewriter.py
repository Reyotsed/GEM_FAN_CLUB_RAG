"""问题重写模块 — 提升检索命中率。

从 ``agent/gem_agent.py`` 拆分出来。
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def rewrite_question(
    *,
    fast_llm,
    rewrite_prompt,
    question: str,
    history: List[Dict[str, str]],
    guard_config,
    format_history_fn,
) -> str:
    """Rewrite the user question for better retrieval (with retry + validation)."""
    from agent.llm_guard import retry_llm_call, validate_rewrite

    try:

        def _do_rewrite() -> str:
            result = (rewrite_prompt | fast_llm).invoke(
                {
                    "current_question": question,
                    "conversation_history": format_history_fn(history),
                }
            )
            return (result.content or "").strip()

        raw_rewrite = retry_llm_call(
            _do_rewrite,
            max_retries=guard_config.max_retries,
            base_delay=guard_config.retry_base_delay,
            jitter=guard_config.retry_jitter,
            label="rewrite",
        )
        rewritten = validate_rewrite(raw_rewrite, question, guard_config)
        logger.info("Question rewrite: '%s' → '%s'", question, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Question rewrite failed after retries, using original: %s", exc)
        return question
