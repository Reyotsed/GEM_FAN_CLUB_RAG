"""
GEM Agent — Skill-based RAG Agent for GEM Fan Club.

This module implements a ReAct-style agent loop using prompt-based
function calling (compatible with any LLM, including ChatZhipuAI which has
limited native function-calling support in LangChain).

Architecture (v2.0 — Skill-based):
    Skills are domain-grouped bundles of functions.  The SkillRegistry
    aggregates all skills and exposes a flat function list to the LLM,
    while internally routing each call to its owning Skill.

Flow:
    1. User question + history → LLM rewrites question for better retrieval
    2. LLM plans which function(s) to call (sees flat function list)
    3. Agent dispatches calls via SkillRegistry → owning Skill executes
    4. LLM generates final persona-rich answer using function results + history
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Generator, List

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.zhipuai import ChatZhipuAI

from agent.skills import (
    SkillRegistry,
    KnowledgeSkill,
    RecommendSkill,
    ProfileSkill,
    UtilitySkill,
    FunctionResult,
)
from rag_modules.prompts import (
    GEM_SYSTEM_PROMPT,
    GEM_USER_PROMPT,
    QUESTION_REWRITE_SYSTEM_PROMPT,
    QUESTION_REWRITE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# ─── Prompt for the PLANNING step (function selection) ──────────────────

TOOL_PLANNING_SYSTEM_PROMPT = """\
你是邓紫棋（G.E.M.）的智能助手调度器。你的职责是分析粉丝的问题，决定需要调用哪些工具来获取信息。

## 可用工具

{tool_schemas}

## 规则
1. 仔细分析问题，选择**最合适**的工具（可以同时选择多个）。
2. 如果问题涉及时间（如"最近"、"下一场"、某个年份），**必须**同时调用 `get_current_datetime` 和相关检索工具。
3. **事实性问题**（生日、星座、家庭、专辑列表、巡演列表、奖项、记录等），**优先**使用 `lookup_artist_profile`。
4. **时间线问题**（某年发生了什么、某事件是什么时候），**优先**使用 `lookup_milestones`。
5. 如果问题涉及某首具体歌曲，优先使用 `search_song_info`。
6. 如果问题涉及演唱会、巡演的具体场次信息，优先使用 `search_concert_schedule`。
7. 如果问题是推荐歌曲、好听的歌，使用 `get_hot_songs_recommendation`。
8. 如果结构化数据不足以回答（如需要详细叙述、背景故事），再用 `search_knowledge_base` 补充。
9. 如果是简单闲聊（如"你好"、"在吗"），可以选择不调用任何工具，返回空列表。
10. 结合对话历史理解上下文（如代词指代）。

## 输出格式
你**必须且只能**输出一个 JSON 数组，每个元素是一个工具调用对象。格式如下：

```json
[
  {{"tool": "工具名称", "arguments": {{"参数名": "参数值"}}}}
]
```

如果不需要工具，输出空数组：
```json
[]
```

**不要输出任何其他文字，只输出 JSON 数组。**
"""

TOOL_PLANNING_USER_PROMPT = """\
## 对话历史
{conversation_history}

## 当前问题
{question}

请分析上述问题，输出需要调用的工具列表（JSON 数组）：
"""


class GemAgent:
    """
    Skill-based RAG Agent for GEM Fan Club.

    Lifecycle::

        agent = GemAgent(llm, config)
        agent.initialize(retriever, time_retriever, sort_fn)
        answer = agent.run(question, history)
    """

    def __init__(self, llm: ChatZhipuAI, app_config: Any) -> None:
        self.llm = llm
        self.config = app_config
        self.skill_registry = SkillRegistry()
        self._initialized = False

        # Read max-tool-calls limit from config (default 3)
        self._max_tool_calls: int = getattr(app_config, "AGENT_MAX_TOOL_CALLS", 3)

        # Prompt templates
        self._planning_prompt = ChatPromptTemplate.from_messages([
            ("system", TOOL_PLANNING_SYSTEM_PROMPT),
            ("user", TOOL_PLANNING_USER_PROMPT),
        ])
        self._rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", QUESTION_REWRITE_SYSTEM_PROMPT),
            ("user", QUESTION_REWRITE_USER_PROMPT),
        ])

    # ------------------------------------------------------------------
    # Initialisation (called once after retrievers are ready)
    # ------------------------------------------------------------------

    def initialize(
        self,
        retriever,
        time_retriever,
        sort_docs_fn,
    ) -> None:
        """Wire up skills with runtime dependencies (retrievers etc.)."""

        # 1. Knowledge Skill
        knowledge_skill = KnowledgeSkill()
        knowledge_skill.configure(
            retriever=retriever,
            time_retriever=time_retriever,
            sort_docs_fn=sort_docs_fn,
            num_results=self.config.HYBRID_RETRIEVER_NUM_RESULTS,
        )

        # 2. Recommend Skill
        recommend_skill = RecommendSkill()
        recommend_skill.configure(retriever=retriever)

        # 3. Profile Skill
        profile_skill = ProfileSkill()
        structured_dir = os.path.join(self.config.DATA_PATH, "structured")
        profile_skill.configure(structured_data_dir=structured_dir)

        # 4. Utility Skill
        utility_skill = UtilitySkill()
        utility_skill.configure()

        # Register all skills
        self.skill_registry.add_skill(knowledge_skill)
        self.skill_registry.add_skill(recommend_skill)
        self.skill_registry.add_skill(profile_skill)
        self.skill_registry.add_skill(utility_skill)

        self._initialized = True
        logger.info(
            "GemAgent initialized with %d skills (%d functions): %s",
            len(self.skill_registry.get_skill_names()),
            len(self.skill_registry),
            self.skill_registry.get_all_function_names(),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str, history: List[Dict[str, str]] | None = None) -> str:
        """
        Full agent loop:
            question rewrite → function planning → function execution → answer generation
        """
        if not self._initialized:
            raise RuntimeError("GemAgent not initialized — call initialize() first")

        history = history or []

        # Step 1: Rewrite question (for better retrieval, with fallback)
        rewritten_question = self._rewrite_question(question, history)

        # Step 2: Plan which functions to call
        function_calls = self._plan_functions(rewritten_question, question, history)

        # Step 3: Execute functions via SkillRegistry
        function_results = self._execute_functions(
            function_calls, rewritten_question, question
        )

        # Step 4: Generate final answer
        return self._generate_answer(question, history, function_results)

    def run_stream(
        self, question: str, history: List[Dict[str, str]] | None = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming version of run().
        Steps 1-3 execute synchronously (their output is internal),
        then Step 4 streams the final answer token-by-token.

        Yields dicts of the form:
            {"event": "status", "data": {"stage": "...", "message": "..."}}
            {"event": "token",  "data": {"content": "..."}}
            {"event": "done",   "data": {}}
            {"event": "error",  "data": {"message": "..."}}
        """
        if not self._initialized:
            yield {"event": "error", "data": {"message": "Agent 未初始化"}}
            return

        history = history or []

        try:
            # Step 1: Rewrite question
            yield {"event": "status", "data": {"stage": "thinking", "message": "正在思考中..."}}
            rewritten_question = self._rewrite_question(question, history)

            # Step 2: Plan which functions to call
            yield {"event": "status", "data": {"stage": "planning", "message": "正在分析问题..."}}
            function_calls = self._plan_functions(rewritten_question, question, history)

            # Step 3: Execute functions
            if function_calls:
                yield {"event": "status", "data": {"stage": "searching", "message": "正在查找资料..."}}
                function_results = self._execute_functions(
                    function_calls, rewritten_question, question
                )
            else:
                function_results = []

            # Step 4: Stream the final answer
            yield {"event": "status", "data": {"stage": "generating", "message": "正在组织回答..."}}
            for token_chunk in self._generate_answer_stream(question, history, function_results):
                yield {"event": "token", "data": {"content": token_chunk}}

            yield {"event": "done", "data": {}}

        except Exception as exc:
            logger.error("run_stream failed: %s", exc, exc_info=True)
            yield {"event": "error", "data": {"message": str(exc)}}

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _rewrite_question(
        self, question: str, history: List[Dict[str, str]]
    ) -> str:
        """Rewrite the user question for better retrieval."""
        try:
            result = (self._rewrite_prompt | self.llm).invoke({
                "current_question": question,
                "conversation_history": self._format_history(history),
            })
            rewritten = result.content.strip()
            logger.info("Question rewrite: '%s' → '%s'", question, rewritten)
            return rewritten
        except Exception as exc:
            logger.warning("Question rewrite failed, using original: %s", exc)
            return question

    def _plan_functions(
        self,
        rewritten_question: str,
        original_question: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which functions to call and with what arguments."""
        schemas_text = json.dumps(
            self.skill_registry.get_all_function_schemas(),
            ensure_ascii=False,
            indent=2,
        )
        try:
            result = (self._planning_prompt | self.llm).invoke({
                "tool_schemas": schemas_text,
                "conversation_history": self._format_history(history),
                "question": original_question,
            })
            raw = result.content.strip()
            calls = self._parse_function_calls(raw)

            # Enforce max-tool-calls limit
            if len(calls) > self._max_tool_calls:
                logger.warning(
                    "LLM planned %d function calls, truncating to %d",
                    len(calls), self._max_tool_calls,
                )
                calls = calls[: self._max_tool_calls]

            logger.info(
                "Function planning decided: %s",
                [c.get("tool") for c in calls],
            )
            return calls
        except Exception as exc:
            logger.warning(
                "Function planning failed, falling back to knowledge search: %s", exc
            )
            return [
                {"tool": "search_knowledge_base", "arguments": {"query": rewritten_question}}
            ]

    def _execute_functions(
        self,
        function_calls: List[Dict[str, Any]],
        rewritten_question: str,
        original_question: str,
    ) -> List[FunctionResult]:
        """Execute all planned function calls via the SkillRegistry."""
        if not function_calls:
            logger.info("No functions to call — direct answer mode")
            return []

        results: List[FunctionResult] = []
        for fc in function_calls:
            func_name = fc.get("tool", "")
            arguments = fc.get("arguments", {})

            # Ensure required 'query' param is present for retrieval functions
            if func_name in (
                "search_knowledge_base",
                "search_concert_schedule",
            ):
                if "query" not in arguments or not arguments["query"]:
                    # LLM omitted the query — fill in with best available
                    arguments["query"] = rewritten_question
                    logger.warning(
                        "Function '%s' missing 'query' arg, auto-filled with: %s",
                        func_name, rewritten_question,
                    )
                else:
                    # Enhance existing query with rewritten version if available
                    original_arg = arguments["query"]
                    arguments["query"] = (
                        rewritten_question
                        if rewritten_question != original_question
                        else original_arg
                    )

            # Same safeguard for song search
            if func_name == "search_song_info":
                if "song_name" not in arguments or not arguments["song_name"]:
                    arguments["song_name"] = original_question
                    logger.warning(
                        "Function '%s' missing 'song_name' arg, auto-filled with: %s",
                        func_name, original_question,
                    )

            result = self.skill_registry.invoke(func_name, arguments)
            results.append(result)

            if result.success:
                logger.info(
                    "Function '%s' (skill '%s') succeeded (%d chars)",
                    func_name, result.skill_name, len(result.data),
                )
            else:
                logger.warning(
                    "Function '%s' (skill '%s') failed: %s",
                    func_name, result.skill_name, result.error,
                )

        return results

    def _generate_answer(
        self,
        question: str,
        history: List[Dict[str, str]],
        function_results: List[FunctionResult],
    ) -> str:
        """Generate the final GEM-persona answer using function results as context."""
        # Build context from function results
        if function_results:
            context_parts = []
            for fr in function_results:
                if fr.success and fr.data:
                    context_parts.append(
                        f"【{fr.function_name} 返回结果】\n{fr.data}"
                    )
                elif not fr.success:
                    context_parts.append(
                        f"【{fr.function_name} 调用失败】{fr.error}"
                    )
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            context_text = "（无需检索资料，直接与粉丝聊天即可）"

        # Extract current time from function results, or generate a fallback
        current_time = self._extract_current_time(function_results)

        # Build message list
        messages = [("system", GEM_SYSTEM_PROMPT)]

        # Add history
        for turn in history[-self.config.HISTORY_MAX_TURNS:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant"):
                messages.append((role, content))

        # Add current question with context
        user_message = GEM_USER_PROMPT.format(
            current_time=current_time,
            context=context_text,
            question=question,
        )
        messages.append(("user", user_message))

        try:
            prompt = ChatPromptTemplate.from_messages(messages)
            response = self.llm.invoke(prompt.invoke({}))
            return response.content
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc, exc_info=True)
            raise RuntimeError("AI回答生成失败，请稍后重试") from exc

    def _generate_answer_stream(
        self,
        question: str,
        history: List[Dict[str, str]],
        function_results: List[FunctionResult],
    ) -> Generator[str, None, None]:
        """Streaming version of _generate_answer. Yields token strings."""
        # Build context from function results (same logic as _generate_answer)
        if function_results:
            context_parts = []
            for fr in function_results:
                if fr.success and fr.data:
                    context_parts.append(
                        f"【{fr.function_name} 返回结果】\n{fr.data}"
                    )
                elif not fr.success:
                    context_parts.append(
                        f"【{fr.function_name} 调用失败】{fr.error}"
                    )
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            context_text = "（无需检索资料，直接与粉丝聊天即可）"

        current_time = self._extract_current_time(function_results)

        messages = [("system", GEM_SYSTEM_PROMPT)]
        for turn in history[-self.config.HISTORY_MAX_TURNS:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant"):
                messages.append((role, content))

        user_message = GEM_USER_PROMPT.format(
            current_time=current_time,
            context=context_text,
            question=question,
        )
        messages.append(("user", user_message))

        try:
            prompt = ChatPromptTemplate.from_messages(messages)
            # Use .stream() instead of .invoke() for token-by-token output
            for chunk in self.llm.stream(prompt.invoke({})):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error("Answer stream generation failed: %s", exc, exc_info=True)
            raise RuntimeError("AI回答生成失败，请稍后重试") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_current_time(function_results: List[FunctionResult]) -> str:
        """Get time from function results if available, otherwise generate it."""
        for fr in function_results:
            if fr.function_name == "get_current_datetime" and fr.success:
                return fr.data
        return datetime.now().strftime("%Y年%m月%d日 %H:%M")

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt injection."""
        if not history:
            return "无对话历史"
        parts = []
        for turn in history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"粉丝: {content}")
            elif role == "assistant":
                parts.append(f"邓紫棋: {content}")
        return "\n".join(parts)

    @staticmethod
    def _parse_function_calls(raw: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM's JSON output into a list of function call dicts.
        Handles markdown code fences, duplicated JSON blocks, and minor formatting issues.
        """
        cleaned = raw.strip()

        # 策略1: 优先提取 markdown 代码块内的 JSON 内容
        # 匹配所有 ```json ... ``` 或 ``` ... ``` 代码块
        fenced_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        
        # 收集候选 JSON 文本：代码块内容 + 代码块外的裸文本
        candidates = []
        
        if fenced_blocks:
            # 有代码块时，优先尝试代码块内的内容
            for block in fenced_blocks:
                block = block.strip()
                if block:
                    candidates.append(block)
            
            # 也尝试去掉所有代码块后的裸文本
            bare_text = re.sub(r"```(?:json)?\s*\n?.*?\n?\s*```", "", cleaned, flags=re.DOTALL).strip()
            if bare_text:
                candidates.append(bare_text)
        else:
            # 没有代码块，直接用原始清理后的文本
            candidates.append(cleaned)

        # 策略2: 依次尝试每个候选文本进行 JSON 解析
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate == "[]":
                continue

            try:
                parsed = json.loads(candidate)
                return GemAgent._validate_function_calls(parsed)
            except json.JSONDecodeError:
                pass

            # 策略3: 如果候选文本包含多行，尝试逐行找第一个有效 JSON
            # 处理类似 "[]\n[]" 的情况
            if "\n" in candidate:
                for line in candidate.split("\n"):
                    line = line.strip()
                    if not line or line == "[]":
                        continue
                    try:
                        parsed = json.loads(line)
                        result = GemAgent._validate_function_calls(parsed)
                        if result:  # 只返回非空的有效结果
                            return result
                    except json.JSONDecodeError:
                        continue

        # 策略4: 最后兜底 - 尝试用最外层方括号/花括号提取 JSON
        # 找到第一个 [ 到最后一个 ] 的区间
        first_bracket = cleaned.find("[")
        if first_bracket != -1:
            # 寻找与之匹配的 ]
            depth = 0
            for i in range(first_bracket, len(cleaned)):
                if cleaned[i] == "[":
                    depth += 1
                elif cleaned[i] == "]":
                    depth -= 1
                    if depth == 0:
                        json_str = cleaned[first_bracket:i + 1]
                        try:
                            parsed = json.loads(json_str)
                            return GemAgent._validate_function_calls(parsed)
                        except json.JSONDecodeError:
                            break

        # 所有候选文本都是空的 "[]"，返回空列表（这不算错误）
        all_empty = all(
            c.strip() in ("", "[]") for c in candidates
        )
        if not all_empty:
            logger.warning(
                "Failed to parse function calls JSON from any candidate.\nRaw: %s",
                raw[:500],
            )

        return []

    @staticmethod
    def _validate_function_calls(parsed) -> List[Dict[str, Any]]:
        """
        Validate and normalize parsed JSON into a list of function call dicts.
        """
        if isinstance(parsed, list):
            valid = []
            for item in parsed:
                if isinstance(item, dict) and "tool" in item:
                    if "arguments" not in item:
                        item["arguments"] = {}
                    valid.append(item)
            return valid
        elif isinstance(parsed, dict) and "tool" in parsed:
            if "arguments" not in parsed:
                parsed["arguments"] = {}
            return [parsed]
        return []
