# 🤖 GEM Fan Club RAG - 邓紫棋 AI 聊天机器人

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.20-purple.svg)
![智谱AI](https://img.shields.io/badge/智谱AI-GLM--4.7-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.27-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

基于 **RAG + Tool-Augmented Agent** 的邓紫棋粉丝智能问答系统，能够以邓紫棋的语气与粉丝自然对话。

[功能特性](#-功能特性) · [快速开始](#-快速开始) · [架构设计](#-架构设计) · [API 文档](#-api-文档) · [版本历史](#-版本历史)

</div>

---

## ✨ 功能特性

### 🎭 智能角色扮演
- **100% 邓紫棋人设**：完全模拟说话风格和语气
- **多轮对话**：上下文感知，对话历史自动截断（可配置轮数）
- **时间感知**：LLM 自主调用时间工具，准确判断事件时间线
- **情感表达**：根据话题调整回答的情感色彩

### 🧠 v2.0 Tool-Augmented Agent（当前版本）

LLM 不再依赖硬编码路由，而是**自主决策**调用哪些工具：

| 工具名称                       | 功能                                   | 文件                 |
| ------------------------------ | -------------------------------------- | -------------------- |
| `search_knowledge_base`        | 通用知识库检索（生涯、个人信息、奖项） | `knowledge_tools.py` |
| `search_concert_schedule`      | 演唱会日程检索（带时间排序）           | `knowledge_tools.py` |
| `search_song_info`             | 歌曲信息检索（歌词、创作背景）         | `knowledge_tools.py` |
| `get_current_datetime`         | 获取当前时间（含星期几）               | `time_tools.py`      |
| `get_hot_songs_recommendation` | 热门歌曲推荐（支持偏好筛选）           | `recommend_tools.py` |

### 🔍 先进检索技术
- **自适应分块**：生涯按时间线、演唱会保持表格结构、歌词整首为一块
- **混合检索**：向量检索 (ChromaDB) + BM25 关键词检索 (jieba 分词)
- **RRF 重排序**：Reciprocal Rank Fusion 科学融合检索结果
- **问题重写**：LLM 智能重写用户问题，提高检索准确性

### 📚 知识库
- **生涯数据** — 成长历程、重要成就
- **演唱会信息** — 巡演记录、演出详情 (I AM Gloria 等)
- **歌词库** — 完整歌词、创作背景
- **热门歌曲** — hot_song.json 推荐数据

## 🛠️ 技术栈

| 技术        | 版本    | 用途                           |
| ----------- | ------- | ------------------------------ |
| Python      | 3.8+    | 主语言                         |
| FastAPI     | 0.116.1 | 异步 Web 框架                  |
| ChromaDB    | 1.0.20  | 向量数据库                     |
| 智谱 AI     | GLM-4.7 | 大语言模型（对话 + Embedding） |
| LangChain   | 0.3.27  | RAG 编排框架                   |
| jieba       | Latest  | 中文分词 (BM25)                |
| uvicorn     | 0.35.0  | ASGI 服务器                    |
| Pydantic    | 2.10.4  | 数据验证                       |
| python-docx | 1.2.0   | Word 文档解析                  |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 智谱 AI API Key
- 8GB+ 内存（推荐）

### 安装与运行

```bash
# 克隆项目
git clone https://github.com/Reyotsed/GEM_FAN_CLUB_RAG.git
cd GEM_FAN_CLUB_RAG

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥（任选一种）
export ZHIPUAI_API_KEY="your_api_key"    # 环境变量
# 或创建 .env 文件: ZHIPUAI_API_KEY=your_api_key

# 数据准备（首次运行，创建向量数据库）
python load_data.py

# 启动服务
python rag_server.py                 # 默认 Agent 模式
python rag_server.py --mode rag      # 传统 RAG 模式
```

### 运行模式

| 模式              | 启动方式                                 | 说明                 |
| ----------------- | ---------------------------------------- | -------------------- |
| **Agent**（默认） | `python rag_server.py` 或 `--mode agent` | LLM 自主决策调用工具 |
| **RAG**           | `--mode rag` 或 `RUNTIME_MODE=rag`       | 传统检索增强生成     |

### 访问地址

- API 服务: `http://localhost:8000`
- Swagger 文档: `http://localhost:8000/docs`
- 健康检查: `http://localhost:8000/`

## 🏗️ 架构设计

```
GEM_FAN_CLUB_RAG/
├── rag_server.py                   # FastAPI 主程序（双模式运行）
├── config.py                       # 全局配置（环境变量驱动）
├── load_data.py                    # 数据加载 & 向量化入口
├── requirements.txt                # 依赖清单
├── hot_song.json                   # 热门歌曲推荐数据
│
├── agent/                          # v2.0 Agent 模块
│   ├── __init__.py
│   ├── gem_agent.py                # Agent 编排核心
│   │   └── 问题重写 → 工具规划(LLM) → 工具执行 → 回答生成
│   └── tools/
│       ├── __init__.py
│       ├── tool_registry.py        # 工具注册中心 + ToolResult
│       ├── knowledge_tools.py      # 知识检索工具 ×3
│       ├── time_tools.py           # 时间工具
│       └── recommend_tools.py      # 推荐工具
│
├── rag_modules/                    # RAG 核心模块
│   ├── data_preparation.py         # 数据清洗、元数据增强
│   ├── adaptive_splitter.py        # 自适应分块器
│   ├── hybrid_retriever.py         # 混合检索 + RRF 重排序
│   └── prompts.py                  # Prompt 模板
│
├── gem_data/                       # 邓紫棋知识库
│   ├── career/gem.txt              # 生涯数据
│   ├── concert/I AM Gloria.json    # 演唱会数据
│   └── lyrics/*.txt                # 歌词库
│
└── scripts/
    └── get_lyrics.py               # 歌词抓取脚本
```

### 系统流程

```mermaid
graph TB
    %% ═══════════ 入口层 ═══════════
    A["👤 用户提问<br/>(question + history)"] --> B["FastAPI /chat_gem<br/>参数校验: 1~500字, ≤20轮历史"]
    B --> C{RUNTIME_MODE?}

    %% ═══════════ Agent 模式 ═══════════
    C -->|"agent (默认)"| D["GemAgent.run()"]

    D --> D1["Step1: 问题重写<br/>_rewrite_question()"]
    D1 -->|"LLM + QUESTION_REWRITE_PROMPT<br/>语义增强/指代消解"| D1a["重写后的问题"]
    D1 -.->|"❌ 失败回退"| D1b["使用原始问题"]
    D1b --> D1a

    D1a --> D2["Step2: 工具规划<br/>_plan_tools()"]
    D2 -->|"LLM + TOOL_PLANNING_PROMPT<br/>注入7个工具的JSON Schema<br/>temperature=0.1"| D2a["解析工具调用列表<br/>JSON: tool + arguments"]
    D2 -.->|"❌ 解析失败回退"| D2b["默认调用<br/>search_knowledge_base"]
    D2b --> D2a

    D2a --> D3["Step3: 执行工具<br/>_execute_tools()<br/>最多3个工具调用"]

    %% ═══════════ 7个工具 ═══════════
    D3 --> T1["🔍 search_knowledge_base<br/>通用知识库搜索<br/>(生涯/个人/奖项)"]
    D3 --> T2["📅 search_concert_schedule<br/>演唱会日程搜索"]
    D3 --> T3["🎵 search_song_info<br/>特定歌曲搜索<br/>(歌词/创作背景)"]
    D3 --> T4["⏰ get_current_datetime<br/>获取当前日期时间"]
    D3 --> T5["🔥 get_hot_songs_recommendation<br/>热门歌曲推荐"]
    D3 --> T6["📋 lookup_artist_profile<br/>结构化档案查询<br/>(零幻觉)"]
    D3 --> T7["🏆 lookup_milestones<br/>里程碑事件查询<br/>(按年/类别筛选)"]

    %% ═══════════ 工具 → 检索链 ═══════════
    T1 --> HR["混合检索<br/>HybridRetriever"]
    T2 --> HR_T["时间检索<br/>time_retriever (K=60)"]
    T3 --> HR
    T5 --> HR

    HR_T --> SORT["sort_docs_by_date()<br/>按日期降序排列"]
    SORT --> CTX

    T4 --> DT["datetime.now()<br/>格式化时间信息"]
    T6 --> SD1["内存缓存读取<br/>artist_profile.json"]
    T7 --> SD2["内存缓存读取<br/>milestones.json"]

    %% ═══════════ 混合检索详细流程 ═══════════
    HR --> V["向量检索<br/>ChromaDB + ZhipuAI Embedding-3<br/>语义相似度 (K=30)"]
    HR --> BM["BM25 检索<br/>jieba 分词 + 关键词匹配<br/>(K=30)"]
    V & BM --> RRF["RRF 重排序<br/>Score = Σ 1/(k+rank), k=60<br/>双路命中文档得分最高"]
    RRF --> TOP["取 Top-20 文档"]
    TOP --> CTX

    %% ═══════════ 汇聚生成回答 ═══════════
    DT --> CTX["汇聚工具结果<br/>拼接为 context"]
    SD1 --> CTX
    SD2 --> CTX

    CTX --> D4["Step4: 生成回答<br/>_generate_answer()"]
    D4 --> PROMPT["构造消息列表<br/>[GEM_SYSTEM_PROMPT,<br/>...history (≤5轮),<br/>GEM_USER_PROMPT<br/>(current_time + context + question)]"]
    PROMPT --> LLM["智谱 AI GLM-4-air"]
    LLM --> ANS["💬 邓紫棋风格回答"]

    %% ═══════════ RAG 模式 ═══════════
    C -->|"rag"| E["传统 RAG Chain<br/>enhanced_rag_chain()"]

    E --> E1["问题重写<br/>question_rewrite_prompt | llm"]
    E1 -.->|"❌ 失败回退"| E1b["使用原始问题"]
    E1b --> E2
    E1 --> E2{"is_time_related_query()?<br/>检测时间关键词<br/>(最近/最新/即将/20XX年)"}

    E2 -->|"是"| E3["time_retriever.retrieve()<br/>→ sort_docs_by_date()"]
    E2 -->|"否"| E4["retriever.retrieve()"]
    E3 --> E5
    E4 --> E5["检索结果"]
    E4 -.->|"❌ 混合检索失败"| E4b["降级: 纯向量检索<br/>db.as_retriever()"]
    E4b --> E5
    E4b -.->|"❌ 向量检索也失败"| ERR["raise RuntimeError"]

    E5 --> E6["build_messages_with_history()<br/>→ [system, ...history, user]"]
    E6 --> LLM

    %% ═══════════ 样式定义 ═══════════
    classDef entryStyle fill:#4A90D9,stroke:#2C5F8A,color:#fff,stroke-width:2px
    classDef agentStyle fill:#7B68EE,stroke:#5A4FCF,color:#fff,stroke-width:2px
    classDef toolStyle fill:#FF8C42,stroke:#CC6B2E,color:#fff,stroke-width:1.5px
    classDef retrievalStyle fill:#2ECC71,stroke:#1FA855,color:#fff,stroke-width:2px
    classDef llmStyle fill:#E74C3C,stroke:#C0392B,color:#fff,stroke-width:2px
    classDef ragStyle fill:#9B59B6,stroke:#7D3C98,color:#fff,stroke-width:1.5px
    classDef resultStyle fill:#F39C12,stroke:#D68910,color:#fff,stroke-width:2px
    classDef errorStyle fill:#E74C3C,stroke:#C0392B,color:#fff,stroke-dasharray:5 5

    class A,B entryStyle
    class D,D1,D1a,D2,D2a,D3,D4,PROMPT agentStyle
    class T1,T2,T3,T4,T5,T6,T7 toolStyle
    class HR,HR_T,V,BM,RRF,TOP,SORT,CTX retrievalStyle
    class LLM llmStyle
    class E,E1,E2,E3,E4,E5,E6 ragStyle
    class ANS resultStyle
    class D1b,D2b,E1b,E4b,ERR errorStyle
    class DT,SD1,SD2 toolStyle
```

## 📡 API 文档

### POST `/chat_gem` — 聊天

```json
// Request
{
    "question": "你最近在忙什么？",
    "history": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！很高兴见到你！"}
    ]
}

// Response
{
    "answer": "我最近在准备新专辑呢！虽然很忙，但看到你们支持我就充满动力！"
}
```

### GET `/` — 健康检查

```json
{
    "message": "欢迎使用 G.E.M. AI Chat API，服务运行正常！"
}
```

## ⚙️ 配置项

所有配置均通过环境变量或 `.env` 文件管理：

| 配置项                         | 默认值             | 说明                             |
| ------------------------------ | ------------------ | -------------------------------- |
| `ZHIPUAI_API_KEY`              | （必填）           | 智谱 AI 密钥                     |
| `ZHIPUAI_MODEL`                | `glm-4.7`          | 对话模型                         |
| `ZHIPUAI_EMBEDDING_MODEL`      | `embedding-3`      | 向量化模型                       |
| `ZHIPUAI_TEMPERATURE`          | `0.8`              | 生成温度                         |
| `ZHIPUAI_TIMEOUT`              | `120`              | API 超时（秒）                   |
| `VECTOR_RETRIEVER_K`           | `10`               | 向量检索 Top-K                   |
| `BM25_RETRIEVER_K`             | `10`               | BM25 检索 Top-K                  |
| `HYBRID_RETRIEVER_NUM_RESULTS` | `5`                | 混合检索最终结果数               |
| `HISTORY_MAX_TURNS`            | `5`                | 对话历史最大轮数                 |
| `AGENT_PLANNING_TEMPERATURE`   | `0.1`              | Agent 工具规划温度（低温更确定） |
| `AGENT_MAX_TOOL_CALLS`         | `3`                | 单轮最大工具调用数               |
| `HOST` / `PORT`                | `0.0.0.0` / `8000` | 服务地址                         |

## 📋 版本历史

| 版本       | 日期       | 核心变更                                   |
| ---------- | ---------- | ------------------------------------------ |
| **v2.0.0** | 2026-03-03 | Tool-Augmented Agent，LLM 自主决策工具调用 |
| **v1.1.0** | —          | 质量改进：限流、去重、K 值调优、日志优化   |
| **v1.0.0** | —          | 初始版本：混合检索 + RRF + 角色扮演        |

> 详细变更日志见 [CHANGELOG.md](CHANGELOG.md)

### 未来规划

- **v2.1** — 工具缓存、专辑/奖项查询工具
- **v3.0** — ReAct 多步推理 + SSE 流式响应
- **v4.0** — Multi-Agent + 微博爬虫 + 多媒体

## 🔗 关联项目

| 项目                 | 说明                 | 仓库                                                   |
| -------------------- | -------------------- | ------------------------------------------------------ |
| **GEM-FAN-CLUB-VUE** | Vue 3 前端应用       | [GitHub](https://github.com/Reyotsed/GEM-FAN-CLUB-VUE) |
| **GEM-FAN-CLUB-WEB** | Spring Boot 后端 API | [GitHub](https://github.com/Reyotsed/GEM-FAN-CLUB-WEB) |

## 🙏 致谢

- [智谱 AI](https://www.zhipuai.cn/) — 中文大语言模型
- [LangChain](https://langchain.com/) — RAG 框架
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
- [FastAPI](https://fastapi.tiangolo.com/) — Web 框架

## 📄 许可证

MIT License

---

<div align="center">

⭐ **如果这个项目对你有帮助，请给一个 Star！**

Made with ❤️ for G.E.M. fans

</div>
