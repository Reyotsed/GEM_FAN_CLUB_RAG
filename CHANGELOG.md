# GEM Fan Club — CHANGELOG & 版本迭代计划

---

## 版本命名规范

- **v1.x** — 增强型 RAG（已完成）
- **v2.x** — Tool-Augmented Agent → Strategy-Enhanced Agent（当前阶段）
- **v3.x** — ReAct Agent + 自进化闭环（规划中）
- **v4.x** — Multi-Agent + 多媒体能力（远期规划）

---

## v2.5.1 — LLM Output Guard（输出守护）

**发布日期：** 2026-03-19  
**核心变更：** 解决模型输出不规范导致链路故障的问题，建立全链路重试/校验/降级/兜底体系

### 🛡️ 问题背景

在实际使用中，LLM（GLM-4.7）在以下环节可能产生不规范输出导致链路故障：
- **问题重写**：返回空内容、带有"重写后的问题："前缀、过长的解释文字
- **工具规划**：不遵守 JSON 格式、输出含解释文字、编造不存在的工具名（幻觉）
- **回答生成**：返回空内容、暴露 AI 身份（"作为AI"、"语言模型"等）

### ✨ 新增功能

#### LLM 输出守护模块 (`agent/llm_guard.py`) — 新建

完整的 LLM 输出防护工具集：

| 组件                         | 功能             | 说明                         |
| ---------------------------- | ---------------- | ---------------------------- |
| `GuardConfig`                | 可配置的守护参数 | 重试次数、延迟、校验阈值等   |
| `retry_llm_call()`           | 指数退避重试     | 含随机 jitter 避免雷同失败   |
| `validate_rewrite()`         | 问题重写输出校验 | 去前缀/去引号/长度校验       |
| `validate_planning_output()` | 工具规划输出校验 | JSON 校验 + 幻觉工具名过滤   |
| `validate_answer()`          | 回答输出校验     | 空内容检测 + AI 身份泄露过滤 |
| `validate_stream_buffer()`   | 流式输出监控     | 实时检测禁止短语             |
| `get_fallback_reply()`       | GEM 人设兜底回复 | 5 条随机回复，保持角色沉浸   |

#### 工具名幻觉过滤（`_fuzzy_match_tool()`）

当 LLM 编造不存在的工具名时，自动尝试模糊匹配修正：
- 大小写不敏感匹配（`Search_Knowledge_Base` → `search_knowledge_base`）
- 下划线归一化（`search_knowledgebase` → `search_knowledge_base`）
- CamelCase 转换（`SearchKnowledgeBase` → `search_knowledge_base`）
- 后缀/子串匹配（`knowledge_base` → `search_knowledge_base`）
- 匹配失败则**过滤掉**该调用（而非崩溃）

#### GEM 人设兜底回复

当所有重试和降级都失败时，返回符合邓紫棋人设的回复，用户无感知异常：
```
"哎呀，脑袋突然短路了一下下 😅 你再问我一次好不好？我保证认真回答！"
"不好意思呀，刚才走神了一下～再说一遍嘛，我这次一定好好回答你！😊"
```

### 🔄 架构变更

- **`agent/gem_agent.py`** — 核心改造：
  - `__init__()` 新增 `GuardConfig` 初始化
  - `initialize()` 填充 `planning_known_tools` 集合
  - `_rewrite_question()` 新增 `retry_llm_call()` + `validate_rewrite()`
  - `_plan_functions()` 新增 `retry_llm_call()` + `validate_planning_output()`（含幻觉工具名过滤）
  - `_generate_answer()` 新增 `retry_llm_call()` + `validate_answer()` + fallback reply
  - `_generate_answer_stream()` 新增流式重试 + `validate_stream_buffer()` + 空流检测 + fallback
  - `run_stream()` 顶层异常处理从暴露原始错误改为 GEM 人设兜底回复
  - 新增 `_build_context_text()` 消除重复代码

- **`TOOL_PLANNING_SYSTEM_PROMPT`** — 提示词优化：
  - 新增合法/非法输出示例（Few-shot），减少模型输出不规范概率
  - 明确"禁止 markdown 标记、禁止解释文字"
  - 强调"工具名必须从可用列表中选择，不要编造"

- **`agent/__init__.py`** — 新增导出 `GuardConfig`
- **`config.py`** — 新增配置项 `AGENT_LLM_MAX_RETRIES`、`AGENT_PLANNING_MAX_RETRIES`

### 📁 新增文件

```
agent/
└── llm_guard.py    # LLM 输出守护（重试/校验/降级/兜底）
```

### ⚙️ 新增配置项

| 配置项                       | 默认值 | 说明                            |
| ---------------------------- | ------ | ------------------------------- |
| `AGENT_LLM_MAX_RETRIES`      | `2`    | 问题重写/回答生成的最大重试次数 |
| `AGENT_PLANNING_MAX_RETRIES` | `2`    | 工具规划步骤的最大重试次数      |

### 🔑 设计要点

1. **全链路覆盖** — 每个 LLM 调用环节都有独立的守护策略
2. **优雅降级** — 层层 fallback：重试 → 输出修正 → 兜底回复，绝不崩溃
3. **人设一致** — 所有兜底回复均保持邓紫棋人设，用户无法感知系统异常
4. **幻觉修正** — 工具名模糊匹配自动修正 LLM 的拼写/格式错误
5. **零侵入** — 守护层完全可选，通过 `GuardConfig` 控制，设 `max_retries=0` 即退化为原始行为
6. **可观测** — 所有重试和降级事件均有详细日志记录，配合 AgentTrace 可定位瓶颈

### 🔙 向后兼容

- API 接口完全不变
- `GuardConfig` 默认值与 v2.5.0 行为一致
- 传统 RAG 模式不受影响

---

## v2.5.0 — Strategy-Enhanced Agent

**发布日期：** 2026-03-19  
**核心变更：** 新增 Prompt 策略池（零延迟 SkillsBank）和链路追踪（AgentTrace），从 v2.0 升级为策略增强的 Agent 管线

> **注意：** v2.5.0 之前已完成 Skill 架构重构（见下方 v2.1.0），工具层从扁平的 `tools/` 升级为按领域分组的 `skills/` 体系。v2.5.0 在此基础上新增策略池和追踪能力。

### ✨ 新增功能

#### Prompt 策略池 (`agent/prompt_strategy.py`) — 新建
借鉴微信 AI 分身系统的 S-Model + SkillsBank 架构，用**零延迟规则引擎**替代额外的 LLM 调用：

- `PromptStrategySelector` — 关键词 + 正则匹配引擎，微秒级完成
- `PromptStrategyResult` — 匹配结果封装，携带合并后的注入文本
- **8 个预定义策略**：

| 策略 ID             | 名称         | 触发场景示例                   | 优先级 |
| ------------------- | ------------ | ------------------------------ | ------ |
| `lyrics_deep_dive`  | 歌词深度解读 | "歌词含义"、"创作背景"、"灵感" | 0.8    |
| `concert_fan_vibe`  | 演唱会热情   | "演唱会"、"巡演"、"live"       | 0.7    |
| `emotional_support` | 温暖共情     | "难过"、"加油"、"失恋"、"焦虑" | 0.9    |
| `fun_chat`          | 轻松幽默     | "哈哈"、"好玩"、"搞笑"         | 0.5    |
| `fact_precise`      | 精确事实     | "什么时候"、"几岁"、"YYYY年"   | 0.85   |
| `song_recommend`    | 歌曲推荐     | "推荐"、"好听的歌"、"歌单"     | 0.6    |
| `fan_love`          | 粉丝互动     | "喜欢你"、"棋士"、"支持你"     | 0.7    |
| `career_story`      | 成长故事     | "出道"、"经历"、"困难"、"蜂鸟" | 0.7    |

- 最多同时匹配 3 个策略（按优先级排序）
- 匹配结果以 `## 🎯 本次回答策略增强` 区块注入 system prompt
- 扩展方式：向 `PROMPT_STRATEGIES` 列表添加条目即可

#### 链路追踪 (`agent/trace.py`) — 新建
每次请求自动记录各阶段耗时，输出格式化日志：

- `AgentTrace` — 请求级追踪容器
- `StageRecord` — 单阶段计时记录
- `stage()` — 上下文管理器，自动计时，异常时标记失败
- `print_summary()` — 输出格式化表格日志
- `set_extra()` — 附加任意元数据（如 tools_called、strategies）
- 仅日志输出，不持久化

输出示例：
```
┌─ AgentTrace ─────────────────────────────────
│ Question : 邓紫棋什么时候生日
│ Total    : 3,241.0 ms
├─ rewrite    :   812.3 ms ✓
├─ strategy   :     0.2 ms ✓  精确事实
├─ planning   :   923.1 ms ✓
├─ execute    :   456.7 ms ✓  lookup_artist_profile
├─ generate   : 1,034.2 ms ✓
│ tools_called: ['lookup_artist_profile']
│ strategies : ['精确事实']
└──────────────────────────────────────────────
```

### 🔄 架构变更

- **`agent/gem_agent.py`** — 核心改造：
  - `__init__()` 新增 `PromptStrategySelector` 初始化
  - `run()` / `run_stream()` 每个阶段用 `trace.stage()` 包裹计时
  - `_generate_answer()` / `_generate_answer_stream()` 新增 `strategy_result: Optional[PromptStrategyResult]` 参数
  - 策略结果注入 system prompt（`GEM_SYSTEM_PROMPT` + 策略增强区块）
  - 模块文档从 v2.0 更新为 v2.5
- **`agent/__init__.py`** — 新增导出 `PromptStrategySelector`、`PromptStrategyResult`、`AgentTrace`

### 📁 新增文件

```
agent/
├── prompt_strategy.py          # Prompt 策略池（零延迟规则引擎）
└── trace.py                    # 链路追踪（AgentTrace）
```

### 🔑 设计要点

1. **策略池完全零延迟** — 纯 Python 关键词/正则匹配，微秒级完成，不消耗任何 LLM token
2. **向后兼容** — `strategy_result` 参数默认 `None`，不传则行为与 v2.0 完全一致
3. **链路追踪仅日志输出** — 不写数据库不写文件，随时可观察各阶段耗时瓶颈
4. **高扩展性** — 新增策略只需向列表添加条目；新增追踪维度只需 `trace.set_extra()`
5. **API 零改动** — `/chat_gem` 和 `/chat_gem/stream` 接口不变，前后端无需修改

### 🔙 向后兼容

- API 接口完全不变
- 不传 `strategy_result` 时，`_generate_answer()` 和 `_generate_answer_stream()` 行为与 v2.0 完全一致
- 传统 RAG 模式不受任何影响

---

## v2.1.0 — Skill 架构重构 + 结构化数据 + 内容安全

**发布日期：** 2026-03-05  
**核心变更：** 将扁平的 `tools/` 架构重构为按领域分组的 `skills/` 体系；新增结构化数据查询（零幻觉）；新增内容安全检查

### 🏗️ 架构重构：`tools/` → `skills/`

原有 `agent/tools/` 中的 5 个独立函数重新组织为 4 个 Skill（领域分组），引入两级分派：

```
SkillRegistry
  ├── KnowledgeSkill   → search_knowledge_base, search_concert_schedule, search_song_info
  ├── RecommendSkill   → get_hot_songs_recommendation
  ├── ProfileSkill     → lookup_artist_profile, lookup_milestones （新增）
  └── UtilitySkill     → get_current_datetime
```

- **`BaseSkill`** 抽象基类 — 统一生命周期：`configure()` → `_finalize()` → `invoke()`
- **`SkillRegistry`** — 聚合所有 Skill，LLM 仍看到扁平的函数列表（零感知变化）
- **`FunctionResult`** 替代 `ToolResult` — 增加 `skill_name` 字段，便于追踪

### ✨ 新增功能

#### 结构化档案查询（ProfileSkill）

| 函数名                  | 功能                       | 数据源                |
| ----------------------- | -------------------------- | --------------------- |
| `lookup_artist_profile` | 精确查询邓紫棋个人资料字段 | `artist_profile.json` |
| `lookup_milestones`     | 按年份/类别查询里程碑事件  | `milestones.json`     |

- 内存缓存 JSON 数据，零检索延迟，零幻觉
- 支持嵌套字段点号导航（如 `family.father`）
- 支持年份范围筛选（如 `2019-2020`）

#### 内容安全检查（`rag_modules/content_safety.py`）

- 关键词匹配 + 正则模式匹配，检测涉黄/涉政/暴力/仇恨言论
- 违规问题自动替换为随机邓紫棋安全话题，保持角色沉浸
- 检查在 Agent/RAG 流程之前执行

### 📁 文件变更

**新增：**
```
agent/skills/
├── __init__.py             # Skills 包入口
├── base_skill.py           # Skill 抽象基类 + FunctionResult
├── skill_registry.py       # Skill 注册中心
├── knowledge_skill.py      # 知识检索 Skill（3 个函数）
├── profile_skill.py        # 结构化档案 Skill（2 个函数，新增）
├── recommend_skill.py      # 推荐 Skill
└── utility_skill.py        # 工具 Skill

gem_data/structured/
├── artist_profile.json     # 艺人档案（零幻觉数据源）
└── milestones.json         # 里程碑事件

rag_modules/content_safety.py   # 内容安全检查
```

**废弃（保留但不再使用）：**
```
agent/tools/                # v2.0 旧工具层，已被 skills/ 完全取代
```

### 🔙 向后兼容

- API 接口完全不变
- LLM 规划提示词中的函数 Schema 格式不变
- 新增 2 个函数（`lookup_artist_profile`、`lookup_milestones`），不影响已有函数

---

## v2.0.0 — Tool-Augmented Agent（Phase 1）

**发布日期：** 2026-03-03  
**核心变更：** 从硬编码 if-else 路由升级为 LLM 自主决策的 Tool-Augmented Agent

> **注意：** v2.0 的 `agent/tools/` 架构已在 v2.1.0 中被 `agent/skills/` 取代。以下记录保留为历史参考。

### ✨ 新增功能

#### Agent 编排层 (`agent/gem_agent.py`)
- 实现 `GemAgent` 类，包含完整的 Agent 循环：
  - **问题重写** → **工具规划（LLM 决策）** → **工具执行** → **回答生成**
- LLM 通过 prompt-based tool calling 自主选择调用哪些工具
- 支持单次调用多个工具（如同时查时间 + 查演唱会）
- 完整的错误处理和降级机制

#### 工具注册中心 (`agent/tools/tool_registry.py`)
- `ToolRegistry` 类：集中管理工具定义、Schema 生成、调度执行
- `ToolResult` 数据类：统一的工具执行结果封装
- 支持动态注册新工具，无需修改 Agent 核心逻辑

#### 5 个内置工具

| 工具名称                       | 文件                 | 功能                                     |
| ------------------------------ | -------------------- | ---------------------------------------- |
| `search_knowledge_base`        | `knowledge_tools.py` | 通用知识库检索（生涯、个人信息、奖项等） |
| `search_concert_schedule`      | `knowledge_tools.py` | 演唱会日程检索（带时间排序）             |
| `search_song_info`             | `knowledge_tools.py` | 歌曲信息检索（歌词、创作背景）           |
| `get_current_datetime`         | `time_tools.py`      | 获取当前时间（含星期几）                 |
| `get_hot_songs_recommendation` | `recommend_tools.py` | 热门歌曲推荐（支持偏好筛选）             |

### 🔄 架构变更

- **新增 `agent/` 模块**，包含 `gem_agent.py` 和 `tools/` 子包
- **`rag_server.py`** 新增双模式运行：
  - `--mode agent`（默认）：使用 GemAgent，LLM 自主决策
  - `--mode rag`：使用传统 RAG 链（保持向后兼容）
  - 也可通过环境变量 `RUNTIME_MODE=agent|rag` 控制
- **`config.py`** 新增 Agent 配置项：
  - `AGENT_PLANNING_TEMPERATURE`：工具规划步骤的温度（默认 0.1）
  - `AGENT_MAX_TOOL_CALLS`：单轮最大工具调用数（默认 3）

### 📁 新增文件

```
agent/
├── __init__.py              # Agent 包入口
├── gem_agent.py             # Agent 编排核心
└── tools/
    ├── __init__.py          # Tools 包入口
    ├── tool_registry.py     # 工具注册中心
    ├── knowledge_tools.py   # 知识检索工具（3个）
    ├── time_tools.py        # 时间工具
    └── recommend_tools.py   # 推荐工具
```

### ⚡ 对比：Agent 模式 vs 传统 RAG 模式

| 方面       | 传统 RAG (v1.x)                  | Agent (v2.0)                            |
| ---------- | -------------------------------- | --------------------------------------- |
| 路由决策   | 硬编码 `is_time_related_query()` | LLM 自主分析问题选择工具                |
| 时间感知   | 仅检测关键词列表                 | LLM 理解语义 + 主动调用时间工具         |
| 歌曲查询   | 统一走通用检索                   | 专用 `search_song_info` 工具            |
| 推荐能力   | 无专门逻辑                       | `get_hot_songs_recommendation` 支持偏好 |
| 扩展性     | 需修改核心代码                   | 新增 Tool 只需注册，无需改 Agent        |
| API 兼容性 | `/chat_gem`                      | `/chat_gem`（完全兼容，零改动）         |

### 🔙 向后兼容

- API 接口 `/chat_gem` 保持不变，前端和后端无需任何修改
- 传统 RAG 链完整保留，随时可通过 `--mode rag` 切换回去
- 所有已有配置项保持不变

---

## v1.x — 增强型 RAG（已完成）

### v1.0.0 — 初始版本
- 基础 RAG：向量检索 + BM25 混合检索 + RRF 重排序
- 邓紫棋角色扮演 Prompt
- 问题重写优化
- 时间感知路由

### v1.1.0 — 质量改进（已完成）
- `_clean_text` 过度清洗修复
- `hot_song.json` 集成到知识库
- Prompt 模板无效空行清理
- `_enhance_metadata` 不覆盖已有 category
- RestTemplate 超时配置
- AI 接口 IP 限流保护
- 前端超时 / history bug 修复
- `manageContext` / `shouldTerminateConversation` 矛盾修复
- 死代码清理、print→logger、时间关键词扩展
- 去重策略改用 chunk_id
- 检索 K 值调优（30→10）
- 对话历史截断轮数配置化
- 后端 JPA ddl-auto 冲突修复

---

## 未来版本规划

### v2.6.0 — Agent 性能优化（规划中）
- [ ] `asyncio.gather` 并行化（问题重写 ∥ BM25 预检索），延迟降低 20-30%
- [ ] ProbeSkill（追问检测）— 复用上轮检索结果，追问场景延迟降低 50%
- [ ] Skill 添加 `get_prompt_hint()` 方法（告诉 LLM 如何最佳使用 Skill 输出）
- [ ] 工具调用结果缓存（相同查询不重复执行）

### v3.0.0 — ReAct Agent + 自进化闭环（规划中）
- [ ] ReAct 多步推理（LLM 观察工具结果后决定是否继续调用）
- [ ] 对话日志持久化 + 反馈收集（👍/👎）
- [ ] 离线分析脚本（发现知识库空缺、沉淀 Prompt 模式）
- [ ] 前端打字机效果优化

### v4.0.0 — Multi-Agent + 多媒体（远期规划）
- [ ] 路由 Agent → 专用子 Agent 协作
- [ ] 微博爬虫 Skill（定时抓取最新动态）
- [ ] 音频片段预览（歌词关联音频）
- [ ] 图片展示（微博图片转发）
