"""Backward-compatible config shim.

Historically this module exposed flat module-level constants (e.g.
``config.ZHIPUAI_API_KEY``). All new code should use ``app.core.settings``
directly, but to avoid touching 30+ call sites in ``agent/`` and
``rag_modules/`` we keep this file as a thin re-export layer that derives its
values from the new ``Settings`` singleton.
"""
from __future__ import annotations

import logging
from typing import List

from dotenv import load_dotenv

# Load .env early so direct ``os.getenv`` callers (legacy) still work.
load_dotenv()

from app.core.settings import get_settings  # noqa: E402

_s = get_settings()

# ==================== 智谱AI配置 ====================
ZHIPUAI_API_KEY: str = _s.llm.api_key
ZHIPUAI_MODEL: str = _s.llm.model
ZHIPUAI_EMBEDDING_MODEL: str = _s.llm.embedding_model
ZHIPUAI_TEMPERATURE: float = _s.llm.temperature
ZHIPUAI_TIMEOUT: float = _s.llm.timeout

# Validation already happened in get_settings(); kept for parity.
if not ZHIPUAI_API_KEY:
    raise ValueError(
        "错误：请设置 ZHIPUAI_API_KEY 环境变量。\n"
        "方法1：创建 .env 文件并添加：ZHIPUAI_API_KEY=your_api_key\n"
        "方法2：在终端中运行：export ZHIPUAI_API_KEY=your_api_key"
    )

# ==================== 数据路径配置 ====================
DATA_PATH: str = _s.retriever.data_path
CHROMA_PATH: str = _s.retriever.chroma_path
HOT_SONG_PATH: str = _s.retriever.hot_song_path

# ==================== 日志配置 ====================
LOG_LEVEL: str = _s.observability.log_level
LOG_FILE: str = _s.observability.log_file
LOG_FORMAT: str = _s.observability.log_format

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
LOG_LEVEL_VALUE: int = _s.observability.log_level_value

# ==================== 检索器配置 ====================
VECTOR_RETRIEVER_K: int = _s.retriever.vector_retriever_k
BM25_RETRIEVER_K: int = _s.retriever.bm25_retriever_k
HYBRID_RETRIEVER_NUM_RESULTS: int = _s.retriever.hybrid_retriever_num_results
RRF_K: int = _s.retriever.rrf_k
HISTORY_MAX_TURNS: int = _s.retriever.history_max_turns

# ==================== Agent 配置 ====================
AGENT_PLANNING_TEMPERATURE: float = _s.agent.planning_temperature
AGENT_MAX_TOOL_CALLS: int = _s.agent.max_tool_calls
AGENT_LLM_MAX_RETRIES: int = _s.agent.llm_max_retries
AGENT_PLANNING_MAX_RETRIES: int = _s.agent.planning_max_retries

# ==================== 数据准备配置 ====================
VECTOR_DB_BATCH_SIZE: int = _s.retriever.vector_db_batch_size

# ==================== 实时网络检索配置（Tavily） ====================
TAVILY_API_KEY: str = _s.agent.tavily_api_key
TAVILY_MAX_RESULTS: int = _s.agent.tavily_max_results
TAVILY_SEARCH_DEPTH: str = _s.agent.tavily_search_depth
TAVILY_TIMEOUT: float = _s.agent.tavily_timeout

# ==================== CORS配置 ====================
CORS_ORIGINS: List[str] = _s.server.cors_origins
CORS_ALLOW_CREDENTIALS: bool = _s.server.cors_allow_credentials
CORS_ALLOW_METHODS: str = _s.server.cors_allow_methods_raw
CORS_ALLOW_METHODS_LIST: List[str] = _s.server.cors_allow_methods
CORS_ALLOW_HEADERS: str = _s.server.cors_allow_headers_raw
CORS_ALLOW_HEADERS_LIST: List[str] = _s.server.cors_allow_headers

# ==================== FastAPI配置 ====================
API_TITLE: str = _s.server.api_title
API_DESCRIPTION: str = _s.server.api_description

# ==================== 服务器配置 ====================
HOST: str = _s.server.host
PORT: int = _s.server.port
