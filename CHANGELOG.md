# GEM Fan Club — CHANGELOG & 版本迭代计划

---

## 版本命名规范

- **v1.x** — 增强型 RAG（已完成）
- **v2.x** — Tool-Augmented Agent（当前阶段）
- **v3.x** — ReAct Agent + 流式响应（规划中）
- **v4.x** — Multi-Agent + 多媒体能力（远期规划）

---

## v2.0.0 — Tool-Augmented Agent（Phase 1）

**发布日期：** 2026-03-03  
**核心变更：** 从硬编码 if-else 路由升级为 LLM 自主决策的 Tool-Augmented Agent

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

### v2.1.0 — Agent 增强（规划中）
- [ ] 工具规划使用独立的低温 LLM 实例（当前共用同一个）
- [ ] 工具调用结果缓存（相同查询不重复执行）
- [ ] 增加 `search_album_info` 工具（专辑查询）
- [ ] 增加 `search_awards` 工具（奖项查询）

### v3.0.0 — ReAct Agent + 流式响应（规划中）
- [ ] ReAct 多步推理（LLM 观察工具结果后决定是否继续调用）
- [ ] SSE 流式响应（`/chat_gem/stream` 端点）
- [ ] 前端打字机效果

### v4.0.0 — Multi-Agent + 多媒体（远期规划）
- [ ] 路由 Agent → 专用子 Agent 协作
- [ ] 微博爬虫 Skill（定时抓取最新动态）
- [ ] 音频片段预览（歌词关联音频）
- [ ] 图片展示（微博图片转发）
