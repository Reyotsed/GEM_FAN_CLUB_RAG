"""Lightweight text helpers shared by the server and the container.

Lives in ``app.core`` to avoid circular imports between ``rag_server`` and
``app.core.container``.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional

import jieba
from langchain.schema import Document


# ---------------------------------------------------------------------------
# jieba cache (idempotent)
# ---------------------------------------------------------------------------

_JIEBA_CACHE_DIR = os.path.join(os.getcwd(), "jieba_cache")
os.makedirs(_JIEBA_CACHE_DIR, exist_ok=True)
jieba.dt.cache_file = os.path.join(_JIEBA_CACHE_DIR, "jieba.cache")


def chinese_preprocess_func(text: str) -> List[str]:
    """jieba-based tokenizer used by BM25Retriever."""
    words = jieba.cut(text)
    return [
        w.strip()
        for w in words
        if w.strip() and not re.match(r"^[^\w\u4e00-\u9fff]+$", w)
    ]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_DATE_PATTERN = r"(\d{4})年(\d{1,2})月(\d{1,2})(?:-(\d{1,2}))?日"


def parse_chinese_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        matches = re.findall(_DATE_PATTERN, date_str)
        if not matches:
            return None
        return min(datetime(int(m[0]), int(m[1]), int(m[2])) for m in matches)
    except (ValueError, AttributeError):
        return None


def sort_docs_by_date(docs: List[Document], reverse: bool = True) -> List[Document]:
    def _key(doc: Document) -> datetime:
        cd = doc.metadata.get("concert_date", "")
        d = parse_chinese_date(cd) if cd else None
        if d is not None:
            return d
        m = re.search(_DATE_PATTERN, doc.page_content)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return datetime(1900, 1, 1)

    with_date, without_date = [], []
    for doc in docs:
        d = _key(doc)
        (with_date if d.year > 1900 else without_date).append((doc, d))
    with_date.sort(key=lambda x: x[1], reverse=reverse)
    return [d for d, _ in with_date] + [d for d, _ in without_date]


def get_current_time() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M")


def is_time_related_query(question: str) -> bool:
    keywords = [
        "最近", "最新", "即将", "马上", "接下来", "未来", "以后",
        "最近的", "最新一场", "最近一场", "下一场", "即将举办",
        "最新演唱会", "最近演唱会", "即将开始的", "最近的演出",
        "latest", "recent", "upcoming", "next", "last",
    ]
    if re.search(r"\b20[2-3]\d\b", question):
        return True
    return any(k in question for k in keywords)
