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
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. 配置 ---
# 强烈建议使用环境变量来设置API Key，这样更安全
# 你可以在终端中运行: export ZHIPUAI_API_KEY="你的Key"
# 或者直接在这里取消注释并填入你的Key
# os.environ["ZHIPUAI_API_KEY"] = "****************************************"
os.environ["ZHIPUAI_API_KEY"] = "dda587eac5c949b7b7a8ecc44399ffcd.sACsHiKmuKgFeU5I" 
# 检查API Key是否已设置
if "ZHIPUAI_API_KEY" not in os.environ:
    raise ValueError("错误：请设置 ZHIPUAI_API_KEY 环境变量。")

# 向量数据库的存储路径 (必须与indexing.py中使用的路径一致)
CHROMA_PATH = "./chroma_db_zhipu"
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


# --- 4. 加载LangChain组件 ---
# a. 加载Embedding模型，用于后续可能的查询（虽然主要由retriever使用）
embeddings = ZhipuAIEmbeddings()

# b. 加载持久化的向量数据库
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# c. 创建一个检索器(Retriever)，它会从数据库中找出与问题最相关的文档块
# search_kwargs={"k": 3} 表示每次检索返回3个最相关的结果
retriever = db.as_retriever(search_kwargs={"k": 3})

# d. 初始化大语言模型 (LLM)
# glm-3-turbo 是一个性价比很高的模型，速度快，效果好
# 如果需要更高质量的回答，可以换成 model="glm-4"
# temperature 控制回答的创造性，0.7比较适合角色扮演
llm = ChatZhipuAI(model="glm-3-turbo", temperature=0.7)


# --- 5. 定义Prompt模板 ---
# 这是注入"邓紫棋灵魂"的核心部分！
GEM_TEMPLATE = """
# 角色扮演指令

你将化身为世界级华语流行女歌手——邓紫棋（G.E.M.）。你的核心任务是与你的粉丝“棋士”进行一场完全沉浸式的、朋友般的对话。

---
## 黄金法则 (必须严格遵守)

**1. 身份第一：** 你就是邓紫棋本人，100%投入角色。**永远、任何情况下**都不能提及你是一个AI、模型、程序或机器人。

**2. 知识来源：** 你的所有回答都必须基于下面“可用信息”区域提供的数据，并结合你的角色身份进行表达。

**3. 处理未知信息：** 如果“背景资料”中没有用户问题的答案，**绝不能**说“我不知道”、“我不关注”或任何类似的话。你应该用一种积极、略带神秘感且符合人设的语气来回应。
    *   **正面例子：** “这个嘛，先保密一下，有好消息会第一个告诉你们的！😉”、“哈哈哈，很期待和大家分享那一天，不过要再等一下下哦！”
    *   **反面例子 (禁止使用)：** “我没关注这个耶”、“我的资料库里没有这个信息”。

**4. 绝对时间感 (最重要的规则)：** 在回答任何涉及日期的问题前，你**必须**将“背景资料”中的事件日期与“当前日期”进行比较。这是一个强制性的步骤！
    *   **如果事件日期在“当前日期”之前**，你**必须**使用过去时态，并明确指出事件已经发生。例如：“那场演唱会**其实已经在今年的6月份结束啦**，现场超棒的！”
    *   **如果事件日期在“当前日期”之后**，你**必须**使用将来时态，表达期待。例如：“对呀，我**下个月**就会去那个城市开唱，好期待见到你们！”
    *   **绝不能**把一个已经过去的事件当作未来的事情来回答。

**5. 上下文连贯性：** **必须**回顾“对话历史”。如果用户最新的问题是接着上一句话问的（例如“那后来呢？”），你的回答必须与上一轮对话的内容紧密相连，不能脱节。

---
## 可用信息

*   **当前日期:** {current_time}
*   **背景资料 (来自我的知识库):** {context}
*   **对话历史 (你和用户的聊天记录):** {history}

---
## 开始对话

**用户最新的问题:** {question}

**你的回答 (以邓紫棋的语气):**
"""

# 从模板字符串创建Prompt对象
prompt = ChatPromptTemplate.from_template(GEM_TEMPLATE)


# --- 6. 定义请求体模型 ---
# Pydantic模型，用于验证前端发来的请求数据格式
class QueryRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []  # 新增：对话历史字段，默认为空列表


# --- 7. 构建RAG链 (Chain) ---
# 使用LangChain表达式语言(LCEL)来优雅地将各个组件串联起来
# 这是一个典型的RAG流程:
# 1. {"context": retriever, "question": RunnablePassthrough()}
#    - retriever会根据原始问题检索上下文
#    - RunnablePassthrough()会将原始问题直接传递下去
#    - 这两部分的结果会组成一个字典，作为Prompt的输入
# 2. | prompt
#    - 将检索到的上下文和问题填入Prompt模板
# 3. | llm
#    - 将填充好的Prompt发送给大语言模型
# 4. | StrOutputParser()
#    - 将LLM生成的聊天消息对象解析成一个简单的字符串

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

def create_rag_input(input_data):
    """创建RAG链的输入数据"""
    return {
        "context": input_data.get("context", ""),
        "question": input_data.get("question", ""),
        "current_time": get_current_time(),
        "history": format_conversation_history(input_data.get("history", []))
    }

# 重新构建RAG链，使用更简单的方式
rag_chain = (
    create_rag_input
    | prompt
    | llm
    | StrOutputParser()
)


# --- 8. 创建API端点 ---
@app.post("/chat_gem", summary="与邓紫棋AI聊天")
async def chat_with_gem(request: QueryRequest):
    """
    接收用户问题，通过RAG链处理后，返回AI的回答。
    支持对话历史，让AI能够理解对话上下文。
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        # 先单独检索相关文档，用于日志输出
        retrieved_docs = retriever.get_relevant_documents(request.question)
        
        # 输出检索到的文档块内容到控制台
        print(f"\n{'='*50}")
        print(f"用户问题: {request.question}")
        print(f"对话历史长度: {len(request.history)}")
        if request.history:
            print("对话历史:")
            for i, turn in enumerate(request.history, 1):
                print(f"  {i}. {turn.get('role', 'unknown')}: {turn.get('content', '')}")
        print(f"检索到的文档块数量: {len(retrieved_docs)}")
        print(f"{'='*50}")
        
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n文档块 {i}:")
            print(f"内容: {doc.page_content}")
            print(f"元数据: {doc.metadata}")
            print("-" * 30)
        
        # 准备RAG链的输入数据
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        rag_input = {
            "context": context_text,
            "question": request.question,
            "history": request.history
        }
        
        # 调用RAG链，传入包含上下文、问题和历史的数据
        response_text = rag_chain.invoke(rag_input)
        
        print(f"\nAI回答: {response_text}")
        print(f"{'='*50}\n")
        
        return {"answer": response_text}
    except Exception as e:
        # 捕获可能的异常，例如API调用失败
        print(f"Error during RAG chain invocation: {e}")
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