# main.py
import os
import argparse
import logging
from typing import List, Dict, Optional
from datetime import datetime
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
from rag_modules.prompts import GEM_TEMPLATE, QUESTION_REWRITE_TEMPLATE
import jieba
import re

# 配置日志 - 从config读取配置
logging.basicConfig(
    level=config.LOG_LEVEL_VALUE,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 过滤第三方库的调试信息，只保留业务相关日志
# 将 httpcore、httpx、urllib3 等 HTTP 库的日志级别设置为 WARNING
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
# 过滤 LangChain 相关的详细日志
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
logging.getLogger("langchain_chroma").setLevel(logging.WARNING)

# 设置jieba缓存目录到项目目录，避免权限问题
jieba_cache_dir = os.path.join(os.getcwd(), 'jieba_cache')
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, 'jieba.cache')

# --- 1. 配置 ---
# 配置已从config.py加载，使用环境变量管理

# 中文预处理函数，用于BM25检索器的中文分词
def chinese_preprocess_func(text: str) -> List[str]:
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


# --- 2. 初始化FastAPI应用 ---
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION
)

# --- 3. 设置CORS跨域中间件 ---
# 这允许你的Vue前端应用(通常在不同的端口或域)能够安全地调用这个API
# 使用config中的CORS配置，支持通过环境变量设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,  # 从配置文件读取，支持环境变量设置
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS_LIST,
    allow_headers=config.CORS_ALLOW_HEADERS_LIST,
)
logger.info(f"CORS已配置，允许的来源: {config.CORS_ORIGINS}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='G.E.M. AI Chat API 服务器')
    parser.add_argument('--update-db', action='store_true', 
                       help='更新向量数据库（默认：如果数据库不存在则创建，存在则跳过）')
    return parser.parse_args()

# 全局变量，用于存储初始化的组件
dp: Optional[DataPreparationModule] = None
embeddings: Optional[ZhipuAIEmbeddings] = None
db: Optional[Chroma] = None
vector_retriever = None
bm25_retriever = None
retriever: Optional[HybridRetriever] = None
llm: Optional[ChatZhipuAI] = None

def initialize_components(update_db: bool = False):
    """初始化所有组件"""
    global dp, embeddings, db, vector_retriever, bm25_retriever, retriever, llm
    
    logger.info("正在初始化组件...")
    
    # 检查向量数据库路径
    CHROMA_PATH = config.CHROMA_PATH
    if not os.path.exists(CHROMA_PATH) and not update_db:
        raise FileNotFoundError(
            f"错误：向量数据库路径 '{CHROMA_PATH}' 不存在。\n"
            "请先运行 'python load_data.py' 创建数据库，或使用 'python rag_server.py --update-db' 创建数据库。"
        )
    
    # 构建向量数据库（根据命令行参数决定是否更新）
    dp = DataPreparationModule(config.DATA_PATH)
    dp.load_data()
    dp.chunk_documents()
    
    # 根据命令行参数决定是否更新数据库
    if update_db:
        logger.info("重建向量数据库...")
        dp.create_vector_db()
    elif not os.path.exists(CHROMA_PATH):
        logger.info("向量数据库不存在，正在创建...")
        dp.create_vector_db()
    
    # 加载LangChain组件
    embeddings = ZhipuAIEmbeddings(api_key=config.ZHIPUAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 向量检索器 - 从配置读取检索数量
    vector_retriever = db.as_retriever(search_kwargs={"k": config.VECTOR_RETRIEVER_K})
    
    # BM25检索器，用于关键词检索 - 从配置读取检索数量，支持中文分词
    all_chunks = dp.chunks
    bm25_retriever = BM25Retriever.from_documents(
        all_chunks, 
        k=config.BM25_RETRIEVER_K,
        preprocess_func=chinese_preprocess_func  # 添加中文分词支持
    )
    
    # 混合检索器 - 使用RRF重排序，从配置读取参数
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        num_results=config.HYBRID_RETRIEVER_NUM_RESULTS,
        rrf_k=config.RRF_K
    )
    
    # 初始化大语言模型 (LLM) - 从配置读取温度参数
    llm = ChatZhipuAI(
        model=config.ZHIPUAI_MODEL, 
        temperature=config.ZHIPUAI_TEMPERATURE, 
        api_key=config.ZHIPUAI_API_KEY
    )
    
    logger.info("组件初始化完成")

# 组件将在启动时初始化（见main块）


# --- 5. 定义Prompt模板 ---
# Prompt模板已从 rag_modules.prompts 模块导入
# 从模板字符串创建Prompt对象
prompt = ChatPromptTemplate.from_template(GEM_TEMPLATE)
question_rewrite_prompt = ChatPromptTemplate.from_template(QUESTION_REWRITE_TEMPLATE)


# --- 6. 定义请求体模型 ---
# Pydantic模型，用于验证前端发来的请求数据格式
class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []  # 新增：对话历史字段，默认为空列表


# --- 7. 构建增强RAG链 ---

def get_current_time() -> str:
    """获取当前时间信息"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M")

def format_conversation_history(history: List[Dict[str, str]]) -> str:
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


# 新增：增强的RAG链，包含问题重写步骤
def enhanced_rag_chain(input_data: Dict[str, any]) -> str:
    """
    增强的RAG链，包含问题重写步骤
    
    Args:
        input_data: 包含question和history的字典
        
    Returns:
        str: AI生成的回答
        
    Raises:
        Exception: 如果处理过程中发生错误
    """
    try:
        # 检查组件是否已初始化
        if llm is None or retriever is None:
            raise RuntimeError("组件未初始化，请先启动服务器")
        
        # 1. 重写问题
        question_rewrite_chain = question_rewrite_prompt | llm
        rewrite_input = {
            "current_question": input_data["question"],
            "conversation_history": format_conversation_history(input_data.get("history", []))
        }
        
        # 调用问题重写链
        rewritten_question_response = question_rewrite_chain.invoke(rewrite_input)
        rewritten_question = rewritten_question_response.content
        
        logger.info(f"原始问题: {input_data['question']}")
        logger.info(f"重写后问题: {rewritten_question}")
        
        # 2. 使用不同的问题进行检索
        # 重写后的问题用于向量搜索，原始问题用于BM25关键字搜索
        retrieved_docs = retriever.retrieve(
            query=input_data['question'],  # 默认查询
            vector_query=rewritten_question,  # 重写后的问题用于向量搜索
            bm25_query=rewritten_question  # 原始问题用于BM25关键字搜索
        )
        
        # 输出检索到的文档详情
        logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
        for i, doc in enumerate(retrieved_docs, 1):
            logger.debug(f"\n{'='*60}")
            logger.debug(f"文档 {i}/{len(retrieved_docs)}:")
            logger.debug(f"来源: {doc.metadata.get('source', '未知')}")
            logger.debug(f"类别: {doc.metadata.get('category', '未知')}")
            if 'rrf_score' in doc.metadata:
                logger.debug(f"RRF分数: {doc.metadata.get('rrf_score', 0):.4f}")
            if 'vector_rank' in doc.metadata and doc.metadata['vector_rank'] is not None:
                logger.debug(f"向量检索排名: {doc.metadata['vector_rank'] + 1}")
            if 'bm25_rank' in doc.metadata and doc.metadata['bm25_rank'] is not None:
                logger.debug(f"BM25检索排名: {doc.metadata['bm25_rank'] + 1}")
            logger.debug(f"文档内容预览 (前200字符):")
            content_preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            logger.debug(f"{content_preview}")
            logger.debug(f"完整文档内容:")
            logger.debug(f"{doc.page_content}")
            logger.debug(f"{'='*60}\n")
        
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
    except Exception as e:
        logger.error(f"RAG链处理失败: {str(e)}", exc_info=True)
        raise


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
        # 输出用户问题和对话历史到日志
        logger.info(f"\n{'='*50}")
        logger.info(f"用户问题: {request.question}")
        logger.info(f"对话历史长度: {len(request.history)}")
        if request.history:
            logger.debug("对话历史:")
            for i, turn in enumerate(request.history, 1):
                logger.debug(f"  {i}. {turn.get('role', 'unknown')}: {turn.get('content', '')}")
        logger.info(f"{'='*50}")
        
        # 准备增强RAG链的输入数据
        rag_input = {
            "question": request.question,
            "history": request.history
        }
        
        # 调用增强RAG链，它会自动处理问题重写和检索
        response_text = enhanced_rag_chain(rag_input)
        
        logger.info(f"\nAI回答: {response_text}")
        logger.info(f"{'='*50}\n")
        
        return {"answer": response_text}
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 捕获可能的异常，例如API调用失败
        logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"AI服务处理请求时发生内部错误: {str(e)}"
        )

# --- 根路径，用于测试服务是否正常运行 ---
@app.get("/", summary="服务健康检查")
def read_root():
    return {"message": "欢迎使用 G.E.M. AI Chat API，服务运行正常！"}

# --- 启动服务器 ---
if __name__ == "__main__":
    import uvicorn
    
    # 解析命令行参数
    args = parse_args()
    
    # 初始化组件
    try:
        initialize_components(update_db=args.update_db)
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise
    
    logger.info("正在启动 G.E.M. AI Chat API 服务器...")
    logger.info(f"服务地址: http://{config.HOST}:{config.PORT}")
    logger.info(f"API文档: http://{config.HOST}:{config.PORT}/docs")
    logger.info("按 Ctrl+C 停止服务器")
    logger.info("\n启动选项说明：")
    logger.info("  python rag_server.py                    # 默认模式：使用现有数据库")
    logger.info("  python rag_server.py --update-db        # 更新模式：重建数据库")
    
    uvicorn.run(app, host=config.HOST, port=config.PORT)