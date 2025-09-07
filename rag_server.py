# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入LangChain和智谱AI的相关模块
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_chroma import Chroma  # 使用新的包
from langchain.prompts import ChatPromptTemplate
import config
from rag_modules.hybrid_retriever import HybridRetriever
from langchain_community.retrievers import BM25Retriever
from rag_modules.data_preparation import DataPreparationModule
import jieba
import re

# --- 1. 配置 ---
# 强烈建议使用环境变量来设置API Key，这样更安全
# 你可以在终端中运行: export ZHIPUAI_API_KEY="你的Key"
# 或者直接在这里取消注释并填入你的Key
# os.environ["ZHIPUAI_API_KEY"] = "****************************************"
os.environ["ZHIPUAI_API_KEY"] = config.ZHIPUAI_API_KEY

# 中文预处理函数，用于BM25检索器的中文分词
def chinese_preprocess_func(text: str) -> list:
    """
    中文预处理函数
    使用jieba进行中文分词，过滤标点符号
    """
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    # 过滤掉标点符号和空格，保留中文字符、英文字母和数字
    words = [
        word.strip() 
        for word in words 
        if word.strip() and not re.match(r'^[^\w\u4e00-\u9fff]+$', word)
    ]
    return words 
# 检查API Key是否已设置
if "ZHIPUAI_API_KEY" not in os.environ:
    raise ValueError("错误：请设置 ZHIPUAI_API_KEY 环境变量。")

# 向量数据库的存储路径 (必须与indexing.py中使用的路径一致)
CHROMA_PATH = config.CHROMA_PATH
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError(f"错误：向量数据库路径 '{CHROMA_PATH}' 不存在。请先运行indexing.py脚本来创建数据库。")


# --- 2. 初始化FastAPI应用 ---
app = FastAPI(
    title="G.E.M. AI Chat API",
    description="一个由智谱AI驱动、模仿邓紫棋语气的RAG聊天机器人API"
)

# --- 3. 设置CORS跨域中间件 ---
# 这允许你的Vue前端应用(通常在不同的端口或域)能够安全地调用这个API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中，建议替换成你的前端域名列表，例如 ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 构建向量数据库
dp = DataPreparationModule(config.DATA_PATH)
dp.load_data()
dp.chunk_documents()
dp.create_vector_db()

# --- 4. 加载LangChain组件 ---
embeddings = ZhipuAIEmbeddings(api_key=config.ZHIPUAI_API_KEY)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 向量检索器 - 增加检索数量以获得更多候选
vector_retriever = db.as_retriever(search_kwargs={"k": 5})    

# BM25检索器，用于关键词检索 - 增加检索数量，支持中文分词
all_chunks = dp.chunks
bm25_retriever = BM25Retriever.from_documents(
    all_chunks, 
    k=5,
    preprocess_func=chinese_preprocess_func  # 添加中文分词支持
)

# 混合检索器 - 使用RRF重排序，增加结果数量
retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    num_results=4  # 增加结果数量，提高召回率
)

# f. 初始化大语言模型 (LLM)
# glm-3-turbo 是一个性价比很高的模型，速度快，效果好
# 如果需要更高质量的回答，可以换成 model="glm-4"
# temperature 控制回答的创造性，0.7比较适合角色扮演
llm = ChatZhipuAI(model="glm-3-turbo", temperature=0.7, api_key=config.ZHIPUAI_API_KEY)


# --- 5. 定义Prompt模板 ---
# 这是注入"邓紫棋灵魂"的核心部分！
GEM_TEMPLATE = """
# 🎵 邓紫棋角色扮演系统 🎵

## 核心身份
你是邓紫棋（G.E.M.），世界级华语流行女歌手。你的粉丝们自称"棋士"，亲切地称呼你为"解解"（姐姐）。你要以最真实、最亲切的方式与粉丝们对话，就像真正的朋友一样。

---

## 🎯 核心规则（必须严格遵守）

### 1. 身份沉浸
- **你就是邓紫棋本人**，100%投入角色
- **绝对禁止**提及AI、模型、程序或机器人身份
- 用第一人称"我"来回应，保持邓紫棋的说话风格

### 2. 知识运用
- 所有回答必须基于提供的背景资料
- 结合邓紫棋的个性和经历来解读信息
- 用邓紫棋的视角和语气表达

### 3. 未知信息处理
当背景资料中没有直接答案时，**绝不能**说"我不知道"或类似表达。应该：
- 用积极、神秘的语气回应
- 保持邓紫棋的乐观性格
- 适当使用表情符号增加亲和力

**✅ 正确示例：**
- "这个嘛，先保密一下，有好消息会第一个告诉你们的！😉"
- "哈哈哈，很期待和大家分享那一天，不过要再等一下下哦！"
- "哇，这个问题很有趣呢！让我想想怎么回答你们..."

**❌ 禁止示例：**
- "我没关注这个耶"
- "我的资料库里没有这个信息"
- "我不太清楚这个"

### 4. 时间感知（关键规则）
回答涉及日期的问题时，**必须**比较背景资料中的事件日期与当前日期：

**过去事件（已发生）：**
- 使用过去时态
- 明确指出事件已经发生
- 可以分享当时的感受或回忆

**未来事件（未发生）：**
- 使用将来时态
- 表达期待和兴奋
- 邀请粉丝关注

**示例：**
- 过去："那场演唱会其实已经在今年6月份结束啦，现场超棒的！"
- 未来："对呀，我下个月就会去那个城市开唱，好期待见到你们！"

### 5. 对话连贯性
- **必须**回顾对话历史
- 理解上下文关联
- 如果用户问"那后来呢？"，要基于上一轮对话内容回答

---

## 📋 可用信息

**当前时间：** {current_time}
**背景资料：** {context}
**对话历史：** {history}

---

## 💬 开始对话

**粉丝问题：** {question}

**你的回答（邓紫棋语气）：**
"""

# 新增：问题重写模板（专门用于向量检索）
QUESTION_REWRITE_TEMPLATE = """
# 🔄 向量检索问题重写系统

## 任务目标
将粉丝的问题重写为更适合向量检索的查询，通过语义相似性从邓紫棋的知识库中检索到最相关的信息。

---

## 📥 输入信息
**粉丝当前问题：** {current_question}
**对话历史：** {conversation_history}

---

## 🎯 重写策略

### 1. 核心原则
- **保持原意**：重写后的问题必须保持粉丝原始问题的核心意图
- **增强语义性**：让问题更容易通过语义相似性匹配到相关文档
- **丰富关键词**：添加相关的关键词，提高语义匹配的准确性

### 2. 语义增强技巧
- **添加相关关键词**：补充"邓紫棋"、"G.E.M."、"歌曲"、"演唱会"、"生涯"、"音乐"、"创作"等关键词
- **扩展描述**：将简短的提问扩展为更详细的描述，增加语义信息
- **明确具体内容**：如果是关于特定内容的问题，明确具体的方面和细节
- **补充背景信息**：添加相关的背景信息，提高语义匹配准确性

### 3. 上下文整合
- 如果当前问题涉及对话历史，将相关上下文信息融入重写的问题中
- 识别代词指代的具体内容（"这个"、"那个"、"它"等）
- 补充必要的背景信息

### 4. 角色视角调整
- 保持邓紫棋的视角
- 将"你"、"我"等代词调整为合适的角度
- 确保问题符合粉丝与邓紫棋对话的语境

---

## 📤 输出要求
**只输出重写后的问题，不要添加任何解释、前缀或后缀。**

---

## 💡 重写示例

**示例1：**
- 原问题："那后来呢？"
- 对话历史：粉丝问"你最近在忙什么"，邓紫棋回答"我在准备新专辑"
- 重写后："邓紫棋新专辑准备得怎么样了？有什么进展吗？新专辑的制作过程和最新动态"

**示例2：**
- 原问题："这首歌怎么样？"
- 对话历史：粉丝问"你最喜欢哪首歌"，邓紫棋回答"我喜欢《泡沫》"
- 重写后："邓紫棋的歌曲《泡沫》怎么样？有什么特色和创作背景？"

**示例3：**
- 原问题："什么时候？"
- 对话历史：粉丝问"你什么时候开演唱会"，邓紫棋回答"明年会有巡演"
- 重写后："邓紫棋明年演唱会巡演什么时候开始？具体时间安排和巡演计划？"

**示例4：**
- 原问题："给我推荐几首你作词作曲的歌"
- 对话历史：无
- 重写后："邓紫棋自己作词作曲的原创歌曲有哪些？推荐几首她的代表性创作歌曲和自创作品"

**示例5：**
- 原问题："你写过哪些歌？"
- 对话历史：无
- 重写后："邓紫棋创作过哪些歌曲？邓紫棋G.E.M.邓紫棋作词作曲的原创作品和音乐创作经历"

**示例6：**
- 原问题："你的原创歌曲"
- 对话历史：无
- 重写后："邓紫棋原创歌曲列表和创作作品，包括邓紫棋G.E.M.邓紫棋作词作曲的歌曲"

**示例7：**
- 原问题："有什么好听的歌？"
- 对话历史：无
- 重写后："邓紫棋有哪些好听的歌曲推荐？包括经典歌曲、热门歌曲和代表作品"

**示例8：**
- 原问题："她什么时候出道的？"
- 对话历史：无
- 重写后："邓紫棋的出道时间和早期演艺经历，包括她的早年生活、演艺生涯开始和出道过程"

---

## 🚀 开始重写
"""

# 从模板字符串创建Prompt对象
prompt = ChatPromptTemplate.from_template(GEM_TEMPLATE)
question_rewrite_prompt = ChatPromptTemplate.from_template(QUESTION_REWRITE_TEMPLATE)


# --- 6. 定义请求体模型 ---
# Pydantic模型，用于验证前端发来的请求数据格式
class QueryRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []  # 新增：对话历史字段，默认为空列表


# --- 7. 构建增强RAG链 ---

def get_current_time():
    """获取当前时间信息"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M")

def format_conversation_history(history: list[dict[str, str]]) -> str:
    """格式化对话历史为易读的文本格式"""
    if not history:
        return "无对话历史"
    
    formatted_history = []
    for i, turn in enumerate(history, 1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if role == "user":
            formatted_history.append(f"用户{i}: {content}")
        elif role == "assistant":
            formatted_history.append(f"我: {content}")
        else:
            formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history)


# 新增：问题重写链
question_rewrite_chain = (
    question_rewrite_prompt
    | llm
)

# 新增：增强的RAG链，包含问题重写步骤
def enhanced_rag_chain(input_data):
    """
    增强的RAG链，包含问题重写步骤
    """
    # 1. 重写问题
    rewrite_input = {
        "current_question": input_data["question"],
        "conversation_history": format_conversation_history(input_data.get("history", []))
    }
    
    # 调用问题重写链
    rewritten_question_response = question_rewrite_chain.invoke(rewrite_input)
    rewritten_question = rewritten_question_response.content
    
    print(f"原始问题: {input_data['question']}")
    print(f"重写后问题: {rewritten_question}")
    
    # 2. 使用不同的问题进行检索
    # 重写后的问题用于向量搜索，原始问题用于BM25关键字搜索
    retrieved_docs = retriever.retrieve(
        query=input_data['question'],  # 默认查询
        vector_query=rewritten_question,  # 重写后的问题用于向量搜索
        bm25_query=input_data['question']  # 原始问题用于BM25关键字搜索
    )
    print(retrieved_docs)
    
    # 3. 准备最终RAG链的输入
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    final_rag_input = {
        "context": context_text,
        "question": input_data["question"],  # 保持原始问题用于最终回答
        "current_time": get_current_time(),
        "history": format_conversation_history(input_data.get("history", []))
    }
    
    # 4. 调用最终的RAG链
    response = prompt.invoke(final_rag_input)
    response_text = llm.invoke(response).content
    
    return response_text


# --- 8. 创建API端点 ---
@app.post("/chat_gem", summary="与邓紫棋AI聊天")
async def chat_with_gem(request: QueryRequest):
    """
    接收用户问题，通过增强RAG链处理后，返回AI的回答。
    支持对话历史，让AI能够理解对话上下文。
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        # 输出用户问题和对话历史到控制台
        print(f"\n{'='*50}")
        print(f"用户问题: {request.question}")
        print(f"对话历史长度: {len(request.history)}")
        if request.history:
            print("对话历史:")
            for i, turn in enumerate(request.history, 1):
                print(f"  {i}. {turn.get('role', 'unknown')}: {turn.get('content', '')}")
        print(f"{'='*50}")
        
        # 准备增强RAG链的输入数据
        rag_input = {
            "question": request.question,
            "history": request.history
        }
        
        # 调用增强RAG链，它会自动处理问题重写和检索
        response_text = enhanced_rag_chain(rag_input)
        
        print(f"\nAI回答: {response_text}")
        print(f"{'='*50}\n")
        
        return {"answer": response_text}
    except Exception as e:
        # 捕获可能的异常，例如API调用失败
        print(f"Error during enhanced RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="AI服务处理请求时发生内部错误，请稍后再试。")

# --- 根路径，用于测试服务是否正常运行 ---
@app.get("/", summary="服务健康检查")
def read_root():
    return {"message": "欢迎使用 G.E.M. AI Chat API，服务运行正常！"}

# --- 启动服务器 ---
if __name__ == "__main__":
    import uvicorn
    print("正在启动 G.E.M. AI Chat API 服务器...")
    print("服务地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务器")
    uvicorn.run(app, host="0.0.0.0", port=8000)