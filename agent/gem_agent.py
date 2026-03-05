"""
GEM Agent — Tool-augmented RAG Agent for GEM Fan Club.

This module implements a ReAct-style agent loop using prompt-based
tool calling (compatible with any LLM, including ChatZhipuAI which has
limited native function-calling support in LangChain).

Flow:
    1. User question + history → LLM decides which tool(s) to call
    2. Agent executes tool(s) and collects results
    3. LLM generates final persona-rich answer using tool results + history
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.zhipuai import ChatZhipuAI

from agent.tools.tool_registry import ToolRegistry, ToolResult
from agent.tools.knowledge_tools import (
    SEARCH_KB_DESCRIPTION, SEARCH_KB_SCHEMA,
    SEARCH_CONCERT_DESCRIPTION, SEARCH_CONCERT_SCHEMA,
    SEARCH_SONG_DESCRIPTION, SEARCH_SONG_SCHEMA,
    search_knowledge_base, search_concert_schedule, search_song_info,
    configure as configure_knowledge_tools,
)
from agent.tools.time_tools import (
    GET_DATETIME_DESCRIPTION, GET_DATETIME_SCHEMA,
    get_current_datetime,
)
from agent.tools.recommend_tools import (
    HOT_SONGS_DESCRIPTION, HOT_SONGS_SCHEMA,
    get_hot_songs_recommendation,
    configure as configure_recommend_tools,
)
from agent.tools.structured_data_tools import (
    LOOKUP_PROFILE_DESCRIPTION, LOOKUP_PROFILE_SCHEMA,
    LOOKUP_MILESTONES_DESCRIPTION, LOOKUP_MILESTONES_SCHEMA,
    lookup_artist_profile, lookup_milestones,
    configure as configure_structured_data,
)
from rag_modules.prompts import (
    GEM_SYSTEM_PROMPT,
    GEM_USER_PROMPT,
    QUESTION_REWRITE_SYSTEM_PROMPT,
    QUESTION_REWRITE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# ─── Prompt for the PLANNING step (tool selection) ─────────────────────

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
    Tool-augmented RAG Agent for GEM Fan Club.

    Lifecycle:
        agent = GemAgent(llm, config)
        agent.initialize(retriever, time_retriever, sort_fn, ...)
        answer = agent.run(question, history)
    """

    def __init__(self, llm: ChatZhipuAI, app_config: Any) -> None:
        self.llm = llm
        self.config = app_config
        self.registry = ToolRegistry()
        self._initialized = False

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
        """Wire up tools with runtime dependencies (retrievers etc.)."""
        # Configure tool modules with retriever references
        configure_knowledge_tools(
            retriever=retriever,
            time_retriever=time_retriever,
            sort_docs_fn=sort_docs_fn,
            hybrid_num_results=self.config.HYBRID_RETRIEVER_NUM_RESULTS,
        )
        configure_recommend_tools(retriever=retriever)

        # Configure structured data tools
        import os
        structured_dir = os.path.join(self.config.DATA_PATH, "structured")
        configure_structured_data(structured_dir)

        # Register tools in the registry
        self.registry.register(
            name="search_knowledge_base",
            description=SEARCH_KB_DESCRIPTION,
            parameters=SEARCH_KB_SCHEMA,
            func=search_knowledge_base,
        )
        self.registry.register(
            name="search_concert_schedule",
            description=SEARCH_CONCERT_DESCRIPTION,
            parameters=SEARCH_CONCERT_SCHEMA,
            func=search_concert_schedule,
        )
        self.registry.register(
            name="search_song_info",
            description=SEARCH_SONG_DESCRIPTION,
            parameters=SEARCH_SONG_SCHEMA,
            func=search_song_info,
        )
        self.registry.register(
            name="get_current_datetime",
            description=GET_DATETIME_DESCRIPTION,
            parameters=GET_DATETIME_SCHEMA,
            func=get_current_datetime,
        )
        self.registry.register(
            name="get_hot_songs_recommendation",
            description=HOT_SONGS_DESCRIPTION,
            parameters=HOT_SONGS_SCHEMA,
            func=get_hot_songs_recommendation,
        )
        self.registry.register(
            name="lookup_artist_profile",
            description=LOOKUP_PROFILE_DESCRIPTION,
            parameters=LOOKUP_PROFILE_SCHEMA,
            func=lookup_artist_profile,
        )
        self.registry.register(
            name="lookup_milestones",
            description=LOOKUP_MILESTONES_DESCRIPTION,
            parameters=LOOKUP_MILESTONES_SCHEMA,
            func=lookup_milestones,
        )

        self._initialized = True
        logger.info(
            "GemAgent initialized with %d tools: %s",
            len(self.registry),
            self.registry.get_tool_names(),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, question: str, history: List[Dict[str, str]] | None = None) -> str:
        """
        Full agent loop:
            question rewrite → tool planning → tool execution → answer generation
        """
        if not self._initialized:
            raise RuntimeError("GemAgent not initialized — call initialize() first")

        history = history or []

        # Step 1: Rewrite question (for better retrieval, with fallback)
        rewritten_question = self._rewrite_question(question, history)

        # Step 2: Plan which tools to call
        tool_calls = self._plan_tools(rewritten_question, question, history)

        # Step 3: Execute tools
        tool_results = self._execute_tools(tool_calls, rewritten_question, question)

        # Step 4: Generate final answer
        answer = self._generate_answer(question, history, tool_results)
        return answer

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

    def _plan_tools(
        self,
        rewritten_question: str,
        original_question: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which tools to call and with what arguments."""
        tool_schemas_text = json.dumps(
            self.registry.get_tool_schemas(), ensure_ascii=False, indent=2
        )
        try:
            result = (self._planning_prompt | self.llm).invoke({
                "tool_schemas": tool_schemas_text,
                "conversation_history": self._format_history(history),
                "question": original_question,
            })
            raw = result.content.strip()
            tool_calls = self._parse_tool_calls(raw)
            logger.info(
                "Tool planning decided: %s",
                [tc.get("tool") for tc in tool_calls],
            )
            return tool_calls
        except Exception as exc:
            logger.warning("Tool planning failed, falling back to knowledge search: %s", exc)
            # Fallback: just do a general knowledge search
            return [{"tool": "search_knowledge_base", "arguments": {"query": rewritten_question}}]

    def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        rewritten_question: str,
        original_question: str,
    ) -> List[ToolResult]:
        """Execute all planned tool calls and collect results."""
        if not tool_calls:
            logger.info("No tools to call — direct answer mode")
            return []

        results: List[ToolResult] = []
        for tc in tool_calls:
            tool_name = tc.get("tool", "")
            arguments = tc.get("arguments", {})

            # Enhance query arguments with rewritten question if applicable
            if "query" in arguments and tool_name in (
                "search_knowledge_base",
                "search_concert_schedule",
            ):
                # Use rewritten question for vector search benefit
                original_arg = arguments["query"]
                arguments["query"] = rewritten_question if rewritten_question != original_question else original_arg

            result = self.registry.invoke(tool_name, arguments)
            results.append(result)
            if result.success:
                logger.info("Tool '%s' succeeded (%d chars)", tool_name, len(result.data))
            else:
                logger.warning("Tool '%s' failed: %s", tool_name, result.error)

        return results

    def _generate_answer(
        self,
        question: str,
        history: List[Dict[str, str]],
        tool_results: List[ToolResult],
    ) -> str:
        """Generate the final GEM-persona answer using tool results as context."""
        # Build context from tool results
        if tool_results:
            context_parts = []
            for tr in tool_results:
                if tr.success and tr.data:
                    context_parts.append(f"【{tr.tool_name} 返回结果】\n{tr.data}")
                elif not tr.success:
                    context_parts.append(f"【{tr.tool_name} 调用失败】{tr.error}")
            context_text = "\n\n---\n\n".join(context_parts)
        else:
            context_text = "（无需检索资料，直接与粉丝聊天即可）"

        # Find current time from tool results, or generate it
        current_time = ""
        for tr in tool_results:
            if tr.tool_name == "get_current_datetime" and tr.success:
                current_time = tr.data
                break
        if not current_time:
            from datetime import datetime
            current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt injection."""
        if not history:
            return "无对话历史"
        parts = []
        for i, turn in enumerate(history, 1):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "user":
                parts.append(f"粉丝: {content}")
            elif role == "assistant":
                parts.append(f"邓紫棋: {content}")
        return "\n".join(parts)

    @staticmethod
    def _parse_tool_calls(raw: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM's JSON output into a list of tool call dicts.
        Handles markdown code fences and minor formatting issues.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        if not cleaned or cleaned == "[]":
            return []

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                # Validate structure
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
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool calls JSON: %s\nRaw: %s", exc, raw[:500])

        return []
