"""
Time-related tools for the GEM Agent.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# =========================================================================
# Tool: get_current_datetime
# =========================================================================

GET_DATETIME_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}

GET_DATETIME_DESCRIPTION = (
    "Get the current date and time. "
    "Use this when you need to determine whether a concert or event "
    "is in the past or in the future, or when the fan asks about 'today', "
    "'now', or time-relative questions."
)


def get_current_datetime() -> str:
    """Return the current date/time in a human-friendly Chinese format."""
    now = datetime.now()
    weekday_map = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekday_map[now.weekday()]
    result = f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M')} {weekday}"
    logger.info("get_current_datetime: %s", result)
    return result
