"""Industrial-grade settings using pydantic-settings.

Splits configuration into focused sub-models. Loads from environment variables
and `.env` file. Existing `config.py` is kept as a backward-compat shim that
derives its module-level constants from the `Settings` singleton produced here.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ----------------------------- Sub-settings ---------------------------------

class LLMSettings(BaseSettings):
    """LLM gateway configuration — supports ZhipuAI, DeepSeek, OpenAI."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # Provider selection
    provider: Literal["zhipuai", "deepseek", "openai"] = Field(
        default="deepseek", validation_alias="LLM_PROVIDER",
    )

    # ---- ZhipuAI (chat + embedding) ----
    api_key: str = Field(default="", validation_alias="ZHIPUAI_API_KEY")
    model: str = Field(default="glm-4.7", validation_alias="ZHIPUAI_MODEL")
    fast_model: str = Field(default="glm-4-flash", validation_alias="ZHIPUAI_FAST_MODEL")
    embedding_model: str = Field(default="embedding-3", validation_alias="ZHIPUAI_EMBEDDING_MODEL")
    temperature: float = Field(default=0.8, validation_alias="ZHIPUAI_TEMPERATURE")
    timeout: float = Field(default=120.0, validation_alias="ZHIPUAI_TIMEOUT")

    # ---- DeepSeek / OpenAI (chat only) ----
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_MODEL")
    deepseek_fast_model: str = Field(default="deepseek-chat", validation_alias="DEEPSEEK_FAST_MODEL")
    deepseek_temperature: float = Field(default=0.8, validation_alias="DEEPSEEK_TEMPERATURE")

    # ---- Derived properties (used by the rest of the code) ----

    @property
    def chat_api_key(self) -> str:
        """LLM 聊天模型使用的 API Key（按 provider 选择）。"""
        if self.provider in ("deepseek", "openai"):
            return self.deepseek_api_key
        return self.api_key

    @property
    def chat_model(self) -> str:
        """主干聊天模型名。"""
        if self.provider in ("deepseek", "openai"):
            return self.deepseek_model
        return self.model

    @property
    def chat_fast_model(self) -> str:
        """快速聊天模型名（rewrite / planning）。"""
        if self.provider in ("deepseek", "openai"):
            return self.deepseek_fast_model
        return self.fast_model

    @property
    def chat_base_url(self) -> str:
        """聊天模型 base_url（OpenAI 兼容 API 需要）。"""
        if self.provider in ("deepseek", "openai"):
            return self.deepseek_base_url
        return ""

    @property
    def chat_temperature(self) -> float:
        if self.provider in ("deepseek", "openai"):
            return self.deepseek_temperature
        return self.temperature

    @model_validator(mode="after")
    def _validate_keys(self) -> "LLMSettings":
        if self.provider == "zhipuai" and not self.api_key:
            raise ValueError(
                "LLM_PROVIDER=zhipuai 但未设置 ZHIPUAI_API_KEY"
            )
        if self.provider in ("deepseek", "openai") and not self.deepseek_api_key:
            raise ValueError(
                f"LLM_PROVIDER={self.provider} 但未设置 DEEPSEEK_API_KEY"
            )
        return self


class RetrieverSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    data_path: str = Field(default="./gem_data", validation_alias="DATA_PATH")
    chroma_path: str = Field(default="./chroma_db_zhipu", validation_alias="CHROMA_PATH")
    chroma_host: Optional[str] = Field(default=None, validation_alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, validation_alias="CHROMA_PORT")
    hot_song_path: str = Field(default="./hot_song.json", validation_alias="HOT_SONG_PATH")

    vector_retriever_k: int = Field(default=10, validation_alias="VECTOR_RETRIEVER_K")
    bm25_retriever_k: int = Field(default=10, validation_alias="BM25_RETRIEVER_K")
    hybrid_retriever_num_results: int = Field(default=5, validation_alias="HYBRID_RETRIEVER_NUM_RESULTS")
    rrf_k: int = Field(default=60, validation_alias="RRF_K")
    history_max_turns: int = Field(default=5, validation_alias="HISTORY_MAX_TURNS")
    vector_db_batch_size: int = Field(default=50, validation_alias="VECTOR_DB_BATCH_SIZE")


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    runtime_mode: Literal["agent", "rag"] = Field(default="agent", validation_alias="RUNTIME_MODE")
    planning_temperature: float = Field(default=0.1, validation_alias="AGENT_PLANNING_TEMPERATURE")
    max_tool_calls: int = Field(default=3, validation_alias="AGENT_MAX_TOOL_CALLS")
    llm_max_retries: int = Field(default=2, validation_alias="AGENT_LLM_MAX_RETRIES")
    planning_max_retries: int = Field(default=2, validation_alias="AGENT_PLANNING_MAX_RETRIES")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    tavily_max_results: int = Field(default=5, validation_alias="TAVILY_MAX_RESULTS")
    tavily_search_depth: str = Field(default="basic", validation_alias="TAVILY_SEARCH_DEPTH")
    tavily_timeout: float = Field(default=15.0, validation_alias="TAVILY_TIMEOUT")


class MemorySettings(BaseSettings):
    """Memory layer configuration (Redis + Postgres + budget)."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    backend: Literal["redis_pg", "inmemory"] = Field(
        default="redis_pg", validation_alias="MEMORY_BACKEND"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, validation_alias="REDIS_PASSWORD")
    postgres_dsn: str = Field(
        default="postgresql://gem:gem@localhost:5432/gem_memory",
        validation_alias="POSTGRES_DSN",
    )

    session_ttl_days: int = Field(default=30, validation_alias="SESSION_TTL_DAYS")
    session_max_raw_messages: int = Field(default=200, validation_alias="SESSION_MAX_RAW_MESSAGES")
    profile_update_every_n_turns: int = Field(default=10, validation_alias="PROFILE_UPDATE_EVERY_N_TURNS")
    long_term_top_k: int = Field(default=3, validation_alias="LONG_TERM_TOP_K")
    embedding_dim: int = Field(default=1024, validation_alias="EMBEDDING_DIM")

    # Context budget (also lives here because it shapes memory selection)
    max_input_tokens: int = Field(default=8000, validation_alias="MAX_INPUT_TOKENS")
    quota_strategy: int = Field(default=500, validation_alias="QUOTA_STRATEGY")
    quota_docs: int = Field(default=3000, validation_alias="QUOTA_DOCS")
    quota_photos: int = Field(default=1500, validation_alias="QUOTA_PHOTOS")
    quota_profile: int = Field(default=300, validation_alias="QUOTA_PROFILE")
    quota_long_term: int = Field(default=500, validation_alias="QUOTA_LONG_TERM")
    quota_summary: int = Field(default=500, validation_alias="QUOTA_SUMMARY")


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    api_title: str = Field(default="G.E.M. AI Chat API", validation_alias="API_TITLE")
    api_description: str = Field(
        default="一个由智谱AI驱动、模仿邓紫棋语气的RAG聊天机器人API",
        validation_alias="API_DESCRIPTION",
    )
    graceful_shutdown_timeout: float = Field(
        default=30.0, validation_alias="GRACEFUL_SHUTDOWN_TIMEOUT"
    )

    cors_origins_raw: str = Field(
        default=(
            "http://localhost:3000,http://localhost:8080,http://localhost:5173,"
            "http://localhost:7071,http://gemfanclub.com,https://gemfanclub.com,"
            "http://www.gemfanclub.com,https://www.gemfanclub.com,"
            "http://140.143.246.54,http://140.143.246.54:7071,"
            "https://localhost,capacitor://localhost,http://localhost"
        ),
        validation_alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, validation_alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods_raw: str = Field(default="*", validation_alias="CORS_ALLOW_METHODS")
    cors_allow_headers_raw: str = Field(default="*", validation_alias="CORS_ALLOW_HEADERS")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def cors_allow_methods(self) -> List[str]:
        return ["*"] if self.cors_allow_methods_raw == "*" else [
            m.strip() for m in self.cors_allow_methods_raw.split(",") if m.strip()
        ]

    @property
    def cors_allow_headers(self) -> List[str]:
        return ["*"] if self.cors_allow_headers_raw == "*" else [
            h.strip() for h in self.cors_allow_headers_raw.split(",") if h.strip()
        ]


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str = Field(default="rag_server.log", validation_alias="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        validation_alias="LOG_FORMAT",
    )
    log_json: bool = Field(default=True, validation_alias="LOG_JSON")
    metrics_enabled: bool = Field(default=True, validation_alias="METRICS_ENABLED")

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return (v or "INFO").upper()

    @property
    def log_level_value(self) -> int:
        return {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(self.log_level, logging.INFO)


# ----------------------------- Aggregated Settings --------------------------


class PhotoSettings(BaseSettings):
    """Configuration for the curated photo-retrieval pool."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    enabled: bool = Field(default=True, validation_alias="PHOTO_SKILL_ENABLED")
    base_url: str = Field(
        default="http://localhost:7071",
        validation_alias="PHOTO_BACKEND_BASE_URL",
        description="Java backend base URL serving /gem-photo/list",
    )
    sync_key: str = Field(default="", validation_alias="PHOTO_SYNC_KEY")
    refresh_ttl_seconds: int = Field(default=600, validation_alias="PHOTO_REFRESH_TTL_SECONDS")
    fetch_timeout_seconds: float = Field(default=5.0, validation_alias="PHOTO_FETCH_TIMEOUT_SECONDS")
    min_score: float = Field(default=0.0, validation_alias="PHOTO_MIN_SCORE")
    default_top_k: int = Field(default=3, validation_alias="PHOTO_DEFAULT_TOP_K")
    max_top_k: int = Field(default=6, validation_alias="PHOTO_MAX_TOP_K")


class Settings(BaseSettings):
    """Aggregated app settings; sub-sections are eagerly instantiated."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    photo: PhotoSettings = Field(default_factory=PhotoSettings)

    def validate_required(self) -> None:
        """Hard-fail if mandatory secrets are missing for the selected LLM provider."""
        if self.llm.provider not in ("deepseek", "openai", "zhipuai"):
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm.provider}")
        # Validation already done in LLMSettings._validate_keys, but
        # we keep a minimal guard here for early detection.
        if self.llm.provider == "zhipuai" and not self.llm.api_key:
            raise ValueError(
                "LLM_PROVIDER=zhipuai 但未设置 ZHIPUAI_API_KEY"
            )
        if self.llm.provider in ("deepseek", "openai") and not self.llm.deepseek_api_key:
            raise ValueError(
                f"LLM_PROVIDER={self.llm.provider} 但未设置 DEEPSEEK_API_KEY"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton accessor."""
    s = Settings()
    s.validate_required()
    return s


# Convenience module-level singleton (kept lightweight; raises only on first
# access if mandatory env is missing).
settings: Settings = get_settings()
