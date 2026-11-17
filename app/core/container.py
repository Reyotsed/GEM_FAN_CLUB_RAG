"""Application-wide DI container.

Owns the lifecycle of every long-lived dependency (LLM, embeddings, vector
store, retriever, agent, memory subsystem, outbox worker) and exposes them
to FastAPI handlers via a single ``get_container`` dependency.

The container is intentionally bare-bones (no third-party DI framework): the
``startup()`` / ``shutdown()`` pair contains plain Python and is easy to read,
test, and reason about.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from app.context.manager import ContextManager
from app.context.summarizer import RollingSummarizer, make_zhipu_invoker
from app.context.tokenizer import get_default_tokenizer
from app.core.logging import configure_logging, get_logger
from app.core.session_validator import SessionOwnershipValidator
from app.core.settings import Settings, get_settings
from app.memory.extractor import ProfileExtractor
from app.memory.outbox import OutboxWorker
from app.memory.service import MemoryService
from app.memory.stores.base import LongTermStore, SessionStore
from app.memory.stores.inmemory_store import (
    InMemoryLongTermStore,
    InMemorySessionStore,
)

_logger = get_logger(__name__)


class AppContainer:
    """Holds all long-lived application dependencies."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        configure_logging()
        self.settings: Settings = settings or get_settings()

        # Heavy components — populated by ``startup()``.
        self.llm: Any = None
        self.fast_llm: Any = None
        self.embeddings: Any = None
        self.db: Any = None
        self.retriever: Any = None
        self.time_retriever: Any = None
        self.agent: Any = None
        self.dp: Any = None  # data preparation module
        self.all_chunks: Any = None

        self.session_store: Optional[SessionStore] = None
        self.long_term_store: Optional[LongTermStore] = None
        self.memory_service: Optional[MemoryService] = None
        self.outbox_worker: Optional[OutboxWorker] = None
        self.session_validator: Optional[SessionOwnershipValidator] = None

        # Photo retrieval (curated GEM photo pool, fetched from Java backend)
        self.photo_repo: Any = None

        self.context_manager: ContextManager = ContextManager(
            tokenizer=get_default_tokenizer(),
            memory_settings=self.settings.memory,
        )
        self.summarizer: Optional[RollingSummarizer] = None
        self.profile_extractor: Optional[ProfileExtractor] = None

        self._redis_client: Any = None
        self._pg_pool: Any = None
        self._started: bool = False

    # ==================================================================
    async def startup(self, *, update_db: bool = False) -> None:
        if self._started:
            return
        _logger.info("container.startup_begin")

        self._init_rag_components(update_db=update_db)
        await self._init_memory_subsystem()
        await self._init_photo_repo()
        self._init_agent()

        self._started = True
        _logger.info(
            "container.startup_complete",
            mode=self.settings.agent.runtime_mode,
            memory_backend=self.settings.memory.backend,
            long_term_available=self.memory_service is not None
            and self.memory_service.long_term_available,
        )

    # ------------------------------------------------------------------
    def _init_rag_components(self, update_db: bool) -> None:
        """Instantiate LLM / embeddings / vector store / retriever.

        We import inside the function so that the container module itself
        stays lightweight (e.g. importable in unit tests that don't need the
        heavy LangChain/Chroma stack).
        """
        from langchain_chroma import Chroma
        from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
        from langchain_community.retrievers import BM25Retriever

        from rag_modules.data_preparation import DataPreparationModule
        from rag_modules.hybrid_retriever import HybridRetriever

        from app.core.text_utils import chinese_preprocess_func

        s = self.settings
        chroma_path = s.retriever.chroma_path
        if not os.path.exists(chroma_path) and not update_db:
            raise FileNotFoundError(
                f"向量数据库路径 '{chroma_path}' 不存在。"
                "请先运行 'python load_data.py' 创建数据库，"
                "或使用 'python rag_server.py --update-db'。"
            )

        # Data preparation
        self.dp = DataPreparationModule(s.retriever.data_path)
        self.dp.load_data(hot_song_path=s.retriever.hot_song_path)
        self.dp.chunk_documents()

        if update_db or not os.path.exists(chroma_path):
            _logger.info("container.creating_vector_db")
            self.dp.create_vector_db()

        # Embeddings & vector DB (always ZhipuAI — best compatibility for
        # Chinese text; DeepSeek has no embedding API).
        self.embeddings = ZhipuAIEmbeddings(
            api_key=s.llm.api_key,
            model=s.llm.embedding_model,
            timeout=s.llm.timeout,
        )
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embeddings)
        self.all_chunks = self.dp.chunks

        # Retrievers
        vector_retriever = self.db.as_retriever(search_kwargs={"k": s.retriever.vector_retriever_k})
        bm25_retriever = BM25Retriever.from_documents(
            self.all_chunks,
            k=s.retriever.bm25_retriever_k,
            preprocess_func=chinese_preprocess_func,
        )
        self.retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            num_results=s.retriever.hybrid_retriever_num_results,
            rrf_k=s.retriever.rrf_k,
        )

        time_k = min(s.retriever.vector_retriever_k * 2, len(self.all_chunks))
        time_vector_retriever = self.db.as_retriever(search_kwargs={"k": time_k})
        time_bm25_retriever = BM25Retriever.from_documents(
            self.all_chunks, k=time_k, preprocess_func=chinese_preprocess_func
        )
        self.time_retriever = HybridRetriever(
            vector_retriever=time_vector_retriever,
            bm25_retriever=time_bm25_retriever,
            num_results=time_k,
            rrf_k=s.retriever.rrf_k,
        )

        # LLMs — create according to provider
        self.llm = self._create_llm(model=s.llm.chat_model, temperature=s.llm.chat_temperature)
        self.fast_llm = self._create_llm(model=s.llm.chat_fast_model, temperature=0.2)

    def _create_llm(self, model: str, temperature: float):
        """Factory: return a LangChain chat model according to LLM_PROVIDER."""
        s = self.settings.llm
        if s.provider in ("deepseek", "openai"):
            # 使用 langchain-community 中的 ChatOpenAI（已安装，无版本冲突）
            from langchain_community.chat_models import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=s.chat_api_key,
                openai_api_base=s.chat_base_url,
                request_timeout=s.timeout,
                max_retries=0,  # we handle retries ourselves in llm_guard
            )
        # Default: ZhipuAI
        from langchain_community.chat_models.zhipuai import ChatZhipuAI

        return ChatZhipuAI(
            model=model,
            temperature=temperature,
            api_key=s.chat_api_key,
            timeout=s.timeout,
        )

    # ------------------------------------------------------------------
    async def _init_memory_subsystem(self) -> None:
        s = self.settings
        backend = s.memory.backend

        # Always have an in-memory fallback ready.
        in_session = InMemorySessionStore(capacity=1024)
        in_lt = InMemoryLongTermStore()

        session_store: SessionStore = in_session
        long_term_store: Optional[LongTermStore] = in_lt
        outbox_redis = None
        worker: Optional[OutboxWorker] = None

        if backend == "redis_pg":
            # ---- Redis ---------------------------------------------------
            try:
                import redis.asyncio as aioredis  # type: ignore

                self._redis_client = aioredis.from_url(
                    s.memory.redis_url,
                    password=s.memory.redis_password,
                    encoding="utf-8",
                    decode_responses=False,
                )
                if await self._redis_client.ping():
                    from app.memory.stores.redis_store import RedisSessionStore

                    session_store = RedisSessionStore(
                        self._redis_client,
                        ttl_seconds=s.memory.session_ttl_days * 86400,
                    )
                    outbox_redis = self._redis_client
                    _logger.info("memory.redis_ready")
                else:
                    raise RuntimeError("ping returned falsy")
            except Exception as e:  # noqa: BLE001
                _logger.warning("memory.redis_unavailable", error=str(e))
                self._redis_client = None
                session_store = in_session

            # ---- Postgres ------------------------------------------------
            try:
                import asyncpg  # type: ignore

                self._pg_pool = await asyncpg.create_pool(
                    dsn=s.memory.postgres_dsn,
                    min_size=1,
                    max_size=10,
                    command_timeout=10,
                )
                async with self._pg_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                from app.memory.stores.pg_store import PgLongTermStore

                long_term_store = PgLongTermStore(
                    self._pg_pool, embedding_dim=s.memory.embedding_dim
                )
                _logger.info("memory.postgres_ready")
            except Exception as e:  # noqa: BLE001
                _logger.warning("memory.postgres_unavailable", error=str(e))
                self._pg_pool = None
                long_term_store = in_lt

        # Embedder closure for the memory layer (uses the LangChain embeddings).
        embedder_fn = self._make_embedder()

        self.session_store = session_store
        self.long_term_store = long_term_store
        self.memory_service = MemoryService(
            session_store=session_store,
            long_term_store=long_term_store,
            embedder=embedder_fn,
            outbox_redis=outbox_redis,
            max_session_messages=s.memory.session_max_raw_messages,
        )
        self.session_validator = SessionOwnershipValidator(redis=self._redis_client)

        # Summarizer + profile extractor (best-effort; failures are non-fatal).
        try:
            invoker = make_zhipu_invoker()
            self.summarizer = RollingSummarizer(invoker)
            if long_term_store is not None:
                self.profile_extractor = ProfileExtractor(
                    long_term_store=long_term_store,
                    invoke=invoker,
                    every_n_turns=s.memory.profile_update_every_n_turns,
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.summarizer_init_failed", error=str(e))

        # Outbox worker (only when both Redis + a non-inmemory long-term store
        # are usable).
        if (
            outbox_redis is not None
            and long_term_store is not None
            and not isinstance(long_term_store, InMemoryLongTermStore)
        ):
            from app.observability.metrics import observe_outbox_lag

            worker = OutboxWorker(
                redis=outbox_redis,
                long_term_store=long_term_store,
                embedder=embedder_fn,
                on_lag=observe_outbox_lag,
            )
            worker.start()
            self.outbox_worker = worker

    def _make_embedder(self):
        emb = self.embeddings

        async def _embed(text: str):
            if not text:
                return []
            try:
                # LangChain's embed_query is sync; offload to thread to keep
                # the event loop responsive.
                return await asyncio.to_thread(emb.embed_query, text)
            except Exception as e:  # noqa: BLE001
                _logger.warning("memory.embed_failed", error=str(e))
                return []

        return _embed

    # ------------------------------------------------------------------
    async def _init_photo_repo(self) -> None:
        """Best-effort init of the curated photo pool. Failures are non-fatal."""
        s = self.settings
        if not getattr(s, "photo", None) or not s.photo.enabled:
            _logger.info("photo_repo.disabled")
            self.photo_repo = None
            return
        try:
            from rag_modules.photo_repo import PhotoRepository

            repo = PhotoRepository(
                base_url=s.photo.base_url,
                sync_key=s.photo.sync_key,
                ttl_seconds=s.photo.refresh_ttl_seconds,
                timeout_seconds=s.photo.fetch_timeout_seconds,
                min_score=s.photo.min_score,
                # 注入 ZhipuAIEmbeddings，启用 caption 语义召回（BM25 + 向量 RRF 融合）。
                # 如果 embeddings 还没初始化好（异常场景），传 None 自动降级为纯 BM25。
                embeddings=self.embeddings,
            )
            await repo.start()
            self.photo_repo = repo
            _logger.info("photo_repo.ready", **repo.stats())
        except Exception as e:  # noqa: BLE001
            _logger.warning("photo_repo.init_failed", error=str(e))
            self.photo_repo = None

    # ------------------------------------------------------------------
    def _init_agent(self) -> None:
        # Local import to avoid pulling agent module at container import time.
        from agent.gem_agent import GemAgent
        import config as legacy_config  # the compat shim

        from app.core.text_utils import sort_docs_by_date

        self.agent = GemAgent(
            llm=self.llm,
            app_config=legacy_config,
            fast_llm=self.fast_llm,  # rewrite/planning 用 fast 模型，大幅降低延迟
        )
        self.agent.initialize(
            retriever=self.retriever,
            time_retriever=self.time_retriever,
            sort_docs_fn=sort_docs_by_date,
            photo_repo=self.photo_repo,
            photo_default_k=self.settings.photo.default_top_k,
            photo_max_k=self.settings.photo.max_top_k,
        )

        # Inject memory + context manager so the agent can orchestrate them.
        self.agent.inject_services(
            memory_service=self.memory_service,
            context_manager=self.context_manager,
        )

    # ==================================================================
    async def shutdown(self) -> None:
        if not self._started:
            return
        _logger.info("container.shutdown_begin")

        if self.outbox_worker is not None:
            try:
                await self.outbox_worker.stop(timeout=5.0)
            except Exception as e:  # noqa: BLE001
                _logger.warning("container.outbox_stop_failed", error=str(e))

        if self.photo_repo is not None:
            try:
                await self.photo_repo.stop()
            except Exception as e:  # noqa: BLE001
                _logger.warning("container.photo_repo_stop_failed", error=str(e))

        if self.session_store is not None:
            try:
                await self.session_store.close()
            except Exception as e:  # noqa: BLE001
                _logger.warning("container.session_close_failed", error=str(e))

        if self.long_term_store is not None:
            try:
                await self.long_term_store.close()
            except Exception as e:  # noqa: BLE001
                _logger.warning("container.long_term_close_failed", error=str(e))

        # Redis client may have been shared with outbox; close once.
        if self._redis_client is not None:
            try:
                await self._redis_client.aclose()
            except Exception:  # noqa: BLE001
                try:
                    await self._redis_client.close()
                except Exception:  # noqa: BLE001
                    pass

        self._started = False
        _logger.info("container.shutdown_complete")
