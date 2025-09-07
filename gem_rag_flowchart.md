# 🎵 GEM Fan Club RAG 系统流程图

```mermaid
flowchart TD
    %% 数据源阶段
    A[("🎵 原始数据源")] --> B1["📚 生涯数据<br/>(career)"]
    A --> B2["🎤 演唱会数据<br/>(concert)"]
    A --> B3["🎶 歌词数据<br/>(lyrics)"]
    
    %% 数据预处理阶段
    B1 --> C["🔧 数据加载模块<br/>(DataPreparationModule)"]
    B2 --> C
    B3 --> C
    
    C1[("🧹 数据清洗<br/>文本清理")] --> C
    C2[("🏷️ 元数据增强<br/>分类标注")] --> C
    
    %% 智能分块阶段
    C --> D["⚙️ 自适应分块器<br/>(AdaptiveSplitter)"]
    
    D1[("📅 生涯分块策略<br/>按时间线分割")] --> D
    D2[("📊 演唱会分块策略<br/>按表格结构分割")] --> D
    D3[("🎵 歌词分块策略<br/>一首歌一个chunk")] --> D
    
    D --> E["📄 文档块<br/>(chunks)"]
    
    %% 向量化存储阶段
    E --> F["🔤 中文分词<br/>(jieba)"]
    E --> G["🧮 向量化处理<br/>(ZhipuAI Embeddings)"]
    
    F --> H["🔍 BM25检索器<br/>(关键词检索)"]
    G --> I["🗄️ ChromaDB<br/>向量数据库"]
    
    %% 用户交互阶段
    J[("👤 用户问题")] --> K["🔄 问题重写模块<br/>(Question Rewrite)"]
    L[("💬 对话历史")] --> K
    
    %% 混合检索阶段
    K --> M["🔀 混合检索器<br/>(HybridRetriever)"]
    I --> N["🎯 向量检索<br/>(Vector Search)"]
    H --> O["🔎 BM25检索<br/>(Keyword Search)"]
    
    N --> M
    O --> M
    
    P[("⚖️ RRF重排序<br/>+ 智能去重")] --> M
    
    %% 生成回答阶段
    M --> Q["📋 相关文档<br/>(Retrieved Docs)"]
    
    Q --> R["🎭 邓紫棋角色模板<br/>(GEM Template)"]
    L --> R
    S[("⏰ 当前时间")] --> R
    
    R --> T["🤖 智谱AI<br/>(GLM-3-Turbo)"]
    T --> U["💝 邓紫棋风格回答<br/>(GEM Response)"]
    
    %% 服务接口阶段
    U --> V["🌐 API响应<br/>(FastAPI)"]
    V --> W["🖥️ 前端展示<br/>(Vue.js)"]
    
    %% 样式定义
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef retrieval fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef generation fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000
    classDef interface fill:#e0f2f1,stroke:#004d40,stroke-width:2px,color:#000
    
    %% 应用样式
    class A,B1,B2,B3 dataSource
    class C,C1,C2,D,D1,D2,D3,E processing
    class F,G,I,H storage
    class J,K,L,M,N,O,P,Q retrieval
    class R,S,T,U generation
    class V,W interface
```

## 🎯 系统流程详解

### 1. 📊 数据源阶段
- **生涯数据**：邓紫棋的成长历程和重要成就
- **演唱会数据**：巡演记录和演出详情  
- **歌词数据**：完整的歌曲歌词和创作背景

### 2. 🔧 数据预处理阶段
- **数据加载**：`DataPreparationModule`统一加载所有数据
- **数据清洗**：清理文本格式，移除特殊字符
- **元数据增强**：根据文件路径自动分类和标注

### 3. ⚙️ 智能分块阶段
- **自适应分块器**：根据文档类型采用不同策略
  - **生涯数据**：按时间线和重要事件分割
  - **演唱会数据**：保持表格结构完整性
  - **歌词数据**：一首歌作为一个完整块

### 4. 🧮 向量化存储阶段
- **中文分词**：使用jieba进行中文分词处理
- **向量化**：使用智谱AI的embedding模型
- **存储**：保存到ChromaDB向量数据库

### 5. 🔍 混合检索阶段
- **问题重写**：将用户问题重写为更完整的查询
- **向量检索**：语义相似度匹配
- **BM25检索**：关键词精确匹配
- **RRF重排序**：科学融合两种检索结果

### 6. 🎭 生成回答阶段
- **角色模板**：邓紫棋的角色扮演模板
- **上下文构建**：结合检索文档、对话历史、当前时间
- **AI生成**：通过智谱AI生成符合邓紫棋风格的回答

### 7. 🌐 服务接口阶段
- **API服务**：FastAPI提供RESTful接口
- **前端展示**：Vue.js前端应用展示结果

## 🚀 技术亮点

- **🎯 自适应分块**：根据内容类型智能选择分块策略
- **🔀 混合检索**：结合语义检索和关键词检索
- **🎭 角色扮演**：完全模拟邓紫棋的说话风格
- **🧠 上下文感知**：支持对话历史和问题重写
- **⚖️ 科学重排序**：使用RRF算法优化检索结果
