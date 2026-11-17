"""
邓紫棋 Agent — 基于技能的 RAG Agent。

使用提示词驱动的函数调用实现 ReAct 风格的 Agent 循环
（兼容任何 LLM，包括 ChatZhipuAI）。

架构（v2.6 — 模块化编排 Agent）：
    技能（Skill）是按领域分组的函数集合。SkillRegistry 聚合所有技能，
    向 LLM 暴露扁平函数列表，内部将调用路由到所属技能。

    v2.6 变更：
    • 核心逻辑按阶段拆分到 ``agent.core.*`` 子模块（planner / rewriter /
      executor / responder）
    • GemAgent 类自身仅保留编排逻辑（init / initialize / run / run_stream）

流程：
    1. 问题重写（LLM） + 策略选择（规则引擎，~0ms）
    2. LLM 规划调用哪些函数（看到扁平函数列表）
    3. Agent 通过 SkillRegistry 分派调用 → 所属技能执行
    4. LLM 生成最终人设回答（结合函数结果 + 策略增强提示 + 历史）
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Generator, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.zhipuai import ChatZhipuAI

from agent.skills import (
    SkillRegistry,
    KnowledgeSkill,
    RecommendSkill,
    ProfileSkill,
    UtilitySkill,
    WebSearchSkill,
    PhotoSkill,
)
from agent.prompt_strategy import PromptStrategySelector
from agent.trace import AgentTrace
from agent.llm_guard import GuardConfig, get_fallback_reply
from agent.core.planner import (
    TOOL_PLANNING_SYSTEM_PROMPT,
    TOOL_PLANNING_USER_PROMPT,
    plan_functions,
    extract_photo_count,
)
from agent.core.rewriter import rewrite_question
from agent.core.executor import execute_functions, extract_photo_payload
from agent.core.responder import (
    generate_answer,
    generate_answer_stream,
    apply_memory_context as apply_mem_ctx,
    format_history,
)
from rag_modules.prompts import (
    QUESTION_REWRITE_SYSTEM_PROMPT,
    QUESTION_REWRITE_USER_PROMPT,
)

logger = logging.getLogger(__name__)


class GemAgent:
    """
    基于技能的 RAG Agent。

    生命周期::
        agent = GemAgent(llm, config)
        agent.initialize(retriever, time_retriever, sort_fn)
        answer = agent.run(question, history)
    """

    def __init__(
        self,
        llm: ChatZhipuAI,
        app_config: Any,
        fast_llm: Optional[ChatZhipuAI] = None,
    ) -> None:
        self.llm = llm
        self.fast_llm = fast_llm or llm
        self.config = app_config
        self.skill_registry = SkillRegistry()
        self._initialized = False

        self._max_tool_calls: int = getattr(app_config, "AGENT_MAX_TOOL_CALLS", 3)
        self._strategy_selector = PromptStrategySelector()
        self._guard_config = GuardConfig(
            max_retries=getattr(app_config, "AGENT_LLM_MAX_RETRIES", 2),
            planning_max_retries=getattr(app_config, "AGENT_PLANNING_MAX_RETRIES", 2),
        )

        # Prompt templates
        self._planning_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TOOL_PLANNING_SYSTEM_PROMPT),
                ("user", TOOL_PLANNING_USER_PROMPT),
            ]
        )
        self._rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUESTION_REWRITE_SYSTEM_PROMPT),
                ("user", QUESTION_REWRITE_USER_PROMPT),
            ]
        )

        # Per-request state (protected by _request_lock)
        self._request_lock = threading.RLock()
        self._active_trace: Optional[AgentTrace] = None
        self._memory_fields: Dict[str, str] = {"summary": "", "long_term": "", "profile": ""}
        self.last_photo_urls: List[str] = []

        # Injected by AppContainer (setattr for backward compat)
        self.memory_service: Any = None
        self.context_manager: Any = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(
        self,
        retriever,
        time_retriever,
        sort_docs_fn,
        photo_repo=None,
        photo_default_k: int = 3,
        photo_max_k: int = 6,
    ) -> None:
        """注入运行时依赖（检索器等），连接各技能。"""
        # 1. 知识技能
        knowledge_skill = KnowledgeSkill()
        knowledge_skill.configure(
            retriever=retriever,
            time_retriever=time_retriever,
            sort_docs_fn=sort_docs_fn,
            num_results=self.config.HYBRID_RETRIEVER_NUM_RESULTS,
        )

        # 2. 推荐技能
        recommend_skill = RecommendSkill()
        recommend_skill.configure(retriever=retriever)

        # 3. 档案技能
        profile_skill = ProfileSkill()
        structured_dir = os.path.join(self.config.DATA_PATH, "structured")
        profile_skill.configure(structured_data_dir=structured_dir)

        # 4. 工具技能
        utility_skill = UtilitySkill()
        utility_skill.configure()

        # 5. 实时网络检索技能
        websearch_skill = WebSearchSkill()
        websearch_skill.configure(
            api_key=getattr(self.config, "TAVILY_API_KEY", None),
            max_results=getattr(self.config, "TAVILY_MAX_RESULTS", 5),
            search_depth=getattr(self.config, "TAVILY_SEARCH_DEPTH", "basic"),
            timeout=getattr(self.config, "TAVILY_TIMEOUT", 15.0),
        )

        # 注册所有技能
        self.skill_registry.add_skill(knowledge_skill)
        self.skill_registry.add_skill(recommend_skill)
        self.skill_registry.add_skill(profile_skill)
        self.skill_registry.add_skill(utility_skill)
        if getattr(self.config, "TAVILY_API_KEY", None):
            self.skill_registry.add_skill(websearch_skill)

        # 6. 图片检索技能
        if photo_repo is not None:
            photo_skill = PhotoSkill()
            photo_skill.configure(
                photo_repo=photo_repo,
                default_k=photo_default_k,
                max_k=photo_max_k,
            )
            self.skill_registry.add_skill(photo_skill)

        self._initialized = True
        self._guard_config.planning_known_tools = set(
            self.skill_registry.get_all_function_names()
        )
        logger.info(
            "GemAgent initialized with %d skills (%d functions): %s",
            len(self.skill_registry.get_skill_names()),
            len(self.skill_registry),
            self.skill_registry.get_all_function_names(),
        )

    # ------------------------------------------------------------------
    # 注入服务（供 AppContainer 使用）
    # ------------------------------------------------------------------

    def inject_services(self, *, memory_service: Any = None, context_manager: Any = None) -> None:
        """Set memory_service and context_manager after construction."""
        if memory_service is not None:
            self.memory_service = memory_service
        if context_manager is not None:
            self.context_manager = context_manager

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        history: List[Dict[str, str]] | None = None,
        memory_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """完整Agent循环（v2.6 — 模块化编排）。
        问题重写 + 策略选择 → 函数规划 → 函数执行 → 回答生成
        """
        if not self._initialized:
            raise RuntimeError("GemAgent not initialized — call initialize() first")

        with self._request_lock:
            return self._run_locked(question, history, memory_context)

    def _run_locked(
        self,
        question: str,
        history: List[Dict[str, str]] | None = None,
        memory_context: Optional[Dict[str, str]] = None,
    ) -> str:
        history = history or []
        trace = AgentTrace(question=question)
        self._active_trace = trace
        apply_mem_ctx(self._memory_fields, memory_context)
        self.last_photo_urls = []

        logger.info("==== 收到 query: %r (history=%d turns) ====", question, len(history))

        # ── 阶段1：问题重写 + 策略选择 ──
        with trace.stage("rewrite"):
            rewritten = rewrite_question(
                fast_llm=self.fast_llm,
                rewrite_prompt=self._rewrite_prompt,
                question=question,
                history=history,
                guard_config=self._guard_config,
                format_history_fn=format_history,
            )

        with trace.stage("strategy", detail="") as stage:
            strategy_result = self._strategy_selector.select(question)
            stage.detail = (
                ", ".join(strategy_result.strategy_names)
                if strategy_result.matched_strategies
                else "(none)"
            )

        # ── 阶段2：规划 ──
        with trace.stage("planning"):
            function_calls = plan_functions(
                fast_llm=self.fast_llm,
                planning_prompt=self._planning_prompt,
                skill_registry=self.skill_registry,
                rewritten_question=rewritten,
                original_question=question,
                history=history,
                guard_config=self._guard_config,
                max_tool_calls=self._max_tool_calls,
                format_history_fn=format_history,
            )

        # ── 阶段3：执行 ──
        with trace.stage("execute", detail=", ".join(fc.get("tool", "") for fc in function_calls)):
            function_results = execute_functions(
                skill_registry=self.skill_registry,
                function_calls=function_calls,
                rewritten_question=rewritten,
                original_question=question,
                extract_photo_count_fn=extract_photo_count,
            )

        # 缓存图片 URL 供外部调用者使用
        for fr in function_results:
            if fr.function_name == "search_gem_photos" and fr.success and fr.data:
                try:
                    obj = json.loads(fr.data)
                    items = (obj or {}).get("items") or []
                    self.last_photo_urls = [
                        item["url"]
                        for item in items
                        if isinstance(item, dict) and item.get("url")
                    ]
                except Exception:  # noqa: BLE001
                    pass

        # ── 阶段4：生成回答 ──
        try:
            with trace.stage("generate"):
                answer = generate_answer(
                    llm=self.llm,
                    question=question,
                    history=history,
                    function_results=function_results,
                    strategy_result=strategy_result,
                    guard_config=self._guard_config,
                    context_manager=getattr(self, "context_manager", None),
                    memory_fields=self._memory_fields,
                    history_max_turns=getattr(self.config, "HISTORY_MAX_TURNS", 10),
                    active_trace=trace,
                )

            trace.set_extra("tools_called", [fc.get("tool") for fc in function_calls])
            trace.set_extra("strategies", strategy_result.strategy_names)
            trace.print_summary()
            return answer

        finally:
            self._active_trace = None
            apply_mem_ctx(self._memory_fields, None)

    def run_stream(
        self,
        question: str,
        history: List[Dict[str, str]] | None = None,
        memory_context: Optional[Dict[str, str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """run() 的流式版本（v2.6 — 模块化编排）。
        阶段1-3同步执行，阶段4逐token流式输出最终回答。
        """
        if not self._initialized:
            yield {"event": "error", "data": {"message": "Agent 未初始化"}}
            return

        with self._request_lock:
            yield from self._run_stream_locked(question, history, memory_context)

    def _run_stream_locked(
        self,
        question: str,
        history: List[Dict[str, str]] | None = None,
        memory_context: Optional[Dict[str, str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        history = history or []
        trace = AgentTrace(question=question)
        self._active_trace = trace
        apply_mem_ctx(self._memory_fields, memory_context)
        self.last_photo_urls = []

        logger.info(
            "==== 收到 query[stream]: %r (history=%d turns) ====", question, len(history)
        )

        try:
            # Phase 1: Rewrite + Strategy
            yield {"event": "status", "data": {"stage": "thinking", "message": "正在思考中..."}}
            with trace.stage("rewrite"):
                rewritten = rewrite_question(
                    fast_llm=self.fast_llm,
                    rewrite_prompt=self._rewrite_prompt,
                    question=question,
                    history=history,
                    guard_config=self._guard_config,
                    format_history_fn=format_history,
                )

            with trace.stage("strategy") as stage:
                strategy_result = self._strategy_selector.select(question)
                stage.detail = (
                    ", ".join(strategy_result.strategy_names)
                    if strategy_result.matched_strategies
                    else "(none)"
                )

            # Phase 2: Planning
            yield {"event": "status", "data": {"stage": "planning", "message": "正在分析问题..."}}
            with trace.stage("planning"):
                function_calls = plan_functions(
                    fast_llm=self.fast_llm,
                    planning_prompt=self._planning_prompt,
                    skill_registry=self.skill_registry,
                    rewritten_question=rewritten,
                    original_question=question,
                    history=history,
                    guard_config=self._guard_config,
                    max_tool_calls=self._max_tool_calls,
                    format_history_fn=format_history,
                )

            # Phase 3: Execute
            if function_calls:
                yield {
                    "event": "status",
                    "data": {"stage": "searching", "message": "正在查找资料..."},
                }
                with trace.stage(
                    "execute", detail=", ".join(fc.get("tool", "") for fc in function_calls)
                ):
                    function_results = execute_functions(
                        skill_registry=self.skill_registry,
                        function_calls=function_calls,
                        rewritten_question=rewritten,
                        original_question=question,
                        extract_photo_count_fn=extract_photo_count,
                    )
            else:
                function_results = []

            # Photo payload for SSE
            photo_payload = extract_photo_payload(function_results)
            if photo_payload and photo_payload.get("items"):
                yield {"event": "photos", "data": photo_payload}

            # Phase 4: Stream answer
            yield {
                "event": "status",
                "data": {"stage": "generating", "message": "正在组织回答..."},
            }
            with trace.stage("generate"):
                for token_chunk in generate_answer_stream(
                    llm=self.llm,
                    question=question,
                    history=history,
                    function_results=function_results,
                    strategy_result=strategy_result,
                    guard_config=self._guard_config,
                    context_manager=getattr(self, "context_manager", None),
                    memory_fields=self._memory_fields,
                    history_max_turns=getattr(self.config, "HISTORY_MAX_TURNS", 10),
                    active_trace=trace,
                ):
                    yield {"event": "token", "data": {"content": token_chunk}}

            yield {"event": "done", "data": {}}

            trace.set_extra("tools_called", [fc.get("tool") for fc in function_calls])
            trace.set_extra("strategies", strategy_result.strategy_names)
            trace.print_summary()

        except Exception as exc:
            logger.error("run_stream failed: %s", exc, exc_info=True)
            yield {"event": "token", "data": {"content": get_fallback_reply()}}
            yield {"event": "done", "data": {}}
        finally:
            self._active_trace = None
            apply_mem_ctx(self._memory_fields, None)
