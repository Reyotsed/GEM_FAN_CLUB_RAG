"""工具规划模块 — LLM 决策调用哪些函数 + JSON 解析 + 输出校验。

从 ``agent/gem_agent.py`` 拆分出来，所有函数均为纯函数（不依赖 GemAgent 实例状态）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── 规划提示词 ──────────────────────────────────────

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
9. **实时/时效问题**（如"最近在干嘛"、"最新动态"、"新歌"、"今年演唱会"、"刚刚/最近发生的事"、"现在在哪"、当年/本月新闻），本地知识库可能过期，**优先**调用 `web_search_gem` 获取最新信息；如需判断时间也同时调用 `get_current_datetime`。
10. **图片请求**（如"发张图"、"看看她的照片"、"有没有 XX 的图"、"来张演唱会的"），**且仅在此时**调用 `search_gem_photos`。普通问答**禁止**调此工具。
    **query 字段填法**（重要）：
    - 用户**有具体倾向**（指明场景/服装/氛围/动作）→ query 写成中文关键词，如"演唱会 红裙"、"自拍 笑容"、"舞台 暖光"
    - 用户**没具体倾向**（只是"发张图""来张照片""随便给我看看""你的美照"）→ **query 留空字符串 ""**，让 skill 内部随机抽
    **数量识别**（重要）：从用户原文里识别明确的数量词，把对应整数填到 `k` 参数：
    - "一张"/"1 张"/"来张" → k=1
    - "两张"/"2 张"/"几张照片" → k=2
    - "三张"/"3 张"/"几张" → k=3
    - "多发几张"/"多来几张"/"五张"/"5张" → k=5
    - 没明说数量 → 省略 k 字段（用默认值）
11. 如果是简单闲聊（如"你好"、"在吗"），可以选择不调用任何工具，返回空列表。
12. 结合对话历史理解上下文（如代词指代）。

## ⚠️ 输出格式（极其重要，必须严格遵守）

你的输出**有且仅有**一个合法的 JSON 数组，**禁止**在 JSON 前后添加任何文字、解释、思考过程或 markdown 标记。

合法输出示例1（调用工具）：
[{{"tool": "search_knowledge_base", "arguments": {{"query": "邓紫棋出道经历"}}}}]

合法输出示例2（调用多个工具）：
[{{"tool": "get_current_datetime", "arguments": {{}}}}, {{"tool": "search_concert_schedule", "arguments": {{"query": "邓紫棋演唱会"}}}}]

合法输出示例3（不调用工具）：
[]

合法输出示例4（图片请求带数量）：
用户："来一张你的自拍" → [{{"tool": "search_gem_photos", "arguments": {{"query": "自拍", "k": 1}}}}]
用户："来几张演唱会的图" → [{{"tool": "search_gem_photos", "arguments": {{"query": "演唱会", "k": 3}}}}]
用户："发张图" → [{{"tool": "search_gem_photos", "arguments": {{"query": ""}}}}]  ← 没明确倾向 query 留空，由 skill 随机抽
用户："给我一张你的照片" → [{{"tool": "search_gem_photos", "arguments": {{"query": "", "k": 1}}}}]  ← 同上，但带数量
用户："随便发几张你的美照" → [{{"tool": "search_gem_photos", "arguments": {{"query": "", "k": 3}}}}]

**非法输出示例（绝对禁止）：**
- ❌ "根据问题分析，我认为需要调用以下工具：[...]"
- ❌ "```json\\n[...]\\n```"
- ❌ "让我分析一下...\\n[...]"

**工具名称必须从上方可用工具列表中选择，不要编造不存在的工具名。**
"""

TOOL_PLANNING_USER_PROMPT = """\
## 对话历史
{conversation_history}

## 当前问题
{question}

请分析上述问题，输出需要调用的工具列表（JSON 数组）：
"""


# ─── 公共函数 ───────────────────────────────────────

def extract_photo_count(question: str) -> Optional[int]:
    """从用户原始问题里硬抽数量词，作为规划 LLM 的兜底。

    规划阶段 LLM 经常忽视"一张/几张"这种隐含数量，
    这里用确定性规则补一道，匹配到则强制覆盖 k。

    返回 None 表示问题里没明确数量词（保留 LLM 给的或默认值）。
    """
    if not question:
        return None
    # 阿拉伯数字：1张 / 3张 / 5 张
    m = re.search(r"(\d+)\s*张", question)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 6:
                return n
        except ValueError:
            pass
    # 中文数量词（按可读性顺序，命中即返回）
    zh_map = [
        ("一张", 1),
        ("两张", 2), ("二张", 2), ("俩张", 2),
        ("三张", 3),
        ("四张", 4),
        ("五张", 5),
        ("六张", 6),
        # 模糊量词
        ("几张", 3), ("多张", 5), ("多发几张", 5), ("多来几张", 5),
    ]
    for kw, n in zh_map:
        if kw in question:
            return n
    return None


def parse_function_calls(raw: str) -> List[Dict[str, Any]]:
    """
    将LLM的JSON输出解析为函数调用字典列表。
    处理 markdown 代码块、重复 JSON 块和格式问题。
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
            return validate_function_calls(parsed)
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
                    result = validate_function_calls(parsed)
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
                    json_str = cleaned[first_bracket : i + 1]
                    try:
                        parsed = json.loads(json_str)
                        return validate_function_calls(parsed)
                    except json.JSONDecodeError:
                        break

    # 所有候选文本都是空的 "[]"，返回空列表（这不算错误）
    all_empty = all(c.strip() in ("", "[]") for c in candidates)
    if not all_empty:
        logger.warning(
            "Failed to parse function calls JSON from any candidate.\nRaw: %s",
            raw[:500],
        )

    return []


def validate_function_calls(parsed: Any) -> List[Dict[str, Any]]:
    """校验并规范化已解析的JSON为函数调用字典列表。"""
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


def plan_functions(
    *,
    fast_llm,
    planning_prompt,
    skill_registry,
    rewritten_question: str,
    original_question: str,
    history: List[Dict[str, str]],
    guard_config,
    max_tool_calls: int,
    format_history_fn,
) -> List[Dict[str, Any]]:
    """
    Ask the LLM which functions to call (with retry + output guard).

    Robustness flow:
        1. Call LLM (with retries on transient failures)
        2. Parse JSON output (multi-strategy parser)
        3. Validate tool names against known tools (filter hallucinations)
        4. Enforce max-tool-calls limit
        5. On total failure → fall back to search_knowledge_base
    """
    from agent.llm_guard import retry_llm_call, validate_planning_output

    schemas_text = json.dumps(
        skill_registry.get_all_function_schemas(),
        ensure_ascii=False,
        indent=2,
    )

    last_raw_output = ""

    def _do_plan() -> str:
        nonlocal last_raw_output
        result = (planning_prompt | fast_llm).invoke(
            {
                "tool_schemas": schemas_text,
                "conversation_history": format_history_fn(history),
                "question": original_question,
            }
        )
        last_raw_output = (result.content or "").strip()
        return last_raw_output

    try:
        # Step 1: Call LLM with retries
        raw = retry_llm_call(
            _do_plan,
            max_retries=guard_config.planning_max_retries,
            base_delay=guard_config.retry_base_delay,
            jitter=guard_config.retry_jitter,
            label="planning",
        )

        # Step 2+3: Parse + validate (filter hallucinated tool names)
        calls = validate_planning_output(
            raw,
            known_tools=guard_config.planning_known_tools,
            parse_fn=parse_function_calls,
        )

        # Step 4: Enforce max-tool-calls limit
        if len(calls) > max_tool_calls:
            logger.warning(
                "LLM planned %d function calls, truncating to %d",
                len(calls),
                max_tool_calls,
            )
            calls = calls[:max_tool_calls]

        logger.info(
            "Function planning decided: %s",
            [c.get("tool") for c in calls],
        )
        return calls

    except Exception as exc:
        logger.warning(
            "Function planning failed after retries, falling back to "
            "knowledge search: %s (last raw output: %s)",
            exc,
            last_raw_output[:200] if last_raw_output else "(none)",
        )
        return [
            {"tool": "search_knowledge_base", "arguments": {"query": rewritten_question}}
        ]
