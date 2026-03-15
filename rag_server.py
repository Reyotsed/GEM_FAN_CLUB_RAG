# main.py
import os
import asyncio
import argparse
import logging
from typing import List, Dict, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from langchain.schema import Document
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import config
from rag_modules.hybrid_retriever import HybridRetriever
from langchain_community.retrievers import BM25Retriever
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.prompts import (
    GEM_SYSTEM_PROMPT,
    GEM_USER_PROMPT,
    QUESTION_REWRITE_SYSTEM_PROMPT,
    QUESTION_REWRITE_USER_PROMPT
)
from agent.gem_agent import GemAgent
from rag_modules.content_safety import check_content_safety
import jieba
import re

logging.basicConfig(
    level=config.LOG_LEVEL_VALUE,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 过滤第三方库的调试信息
for lib in ["httpcore", "httpx", "urllib3", "requests", "langchain", "langchain_community", "langchain_chroma"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# 设置jieba缓存目录
jieba_cache_dir = os.path.join(os.getcwd(), 'jieba_cache')
os.makedirs(jieba_cache_dir, exist_ok=True)
jieba.dt.cache_file = os.path.join(jieba_cache_dir, 'jieba.cache')

# --- 工具函数 ---

def chinese_preprocess_func(text: str) -> List[str]:
    """中文预处理函数：使用jieba分词，过滤标点符号"""
    words = jieba.cut(text)
    words = [
        word.strip() 
        for word in words 
        if word.strip() and not re.match(r'^[^\w\u4e00-\u9fff]+$', word)
    ]
    return words


# --- FastAPI应用初始化 ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: initialize components on startup"""
    args = parse_args()
    try:
        initialize_components(update_db=args.update_db)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        raise
    yield
    # Cleanup on shutdown (if needed in the future)
    logger.info("Application shutdown")

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS_LIST,
    allow_headers=config.CORS_ALLOW_HEADERS_LIST,
)
logger.info(f"CORS已配置，允许的来源: {config.CORS_ORIGINS}")

# 解析命令行参数
def parse_args():
    global RUNTIME_MODE
    parser = argparse.ArgumentParser(description='G.E.M. AI Chat API 服务器')
    parser.add_argument('--update-db', action='store_true', 
                       help='更新向量数据库（默认：如果数据库不存在则创建，存在则跳过）')
    parser.add_argument('--mode', choices=['agent', 'rag'], default=None,
                       help='运行模式：agent（默认，LLM自主决策工具调用）或 rag（传统RAG链）')
    # Use parse_known_args to avoid errors when launched by uvicorn with its own arguments
    args, _ = parser.parse_known_args()
    # CLI --mode takes precedence over env var
    if args.mode is not None:
        RUNTIME_MODE = args.mode
    return args

# 全局变量，用于存储初始化的组件
dp: Optional[DataPreparationModule] = None
embeddings: Optional[ZhipuAIEmbeddings] = None
db: Optional[Chroma] = None
vector_retriever = None
bm25_retriever = None
retriever: Optional[HybridRetriever] = None
llm: Optional[ChatZhipuAI] = None
all_chunks = None  # 存储所有文档块，用于创建临时检索器
time_retriever: Optional[HybridRetriever] = None  # Cached retriever for time-related queries
gem_agent: Optional[GemAgent] = None  # Tool-augmented agent (Phase 1)

# Runtime mode: 'agent' (default) or 'rag' (legacy fallback)
RUNTIME_MODE: str = os.getenv("RUNTIME_MODE", "agent")

def initialize_components(update_db: bool = False):
    """初始化所有组件"""
    global dp, embeddings, db, vector_retriever, bm25_retriever, retriever, llm, all_chunks, time_retriever, gem_agent
    
    logger.info("正在初始化组件...")
    
    CHROMA_PATH = config.CHROMA_PATH
    if not os.path.exists(CHROMA_PATH) and not update_db:
        raise FileNotFoundError(
            f"错误：向量数据库路径 '{CHROMA_PATH}' 不存在。\n"
            "请先运行 'python load_data.py' 创建数据库，或使用 'python rag_server.py --update-db' 创建数据库。"
        )
    
    # 加载和分块文档
    dp = DataPreparationModule(config.DATA_PATH)
    dp.load_data(hot_song_path=config.HOT_SONG_PATH)
    dp.chunk_documents()
    
    # 创建或更新向量数据库
    if update_db or not os.path.exists(CHROMA_PATH):
        logger.info("创建向量数据库...")
        dp.create_vector_db()
    
    # 初始化Embedding和向量数据库
    embeddings = ZhipuAIEmbeddings(
        api_key=config.ZHIPUAI_API_KEY,
        model=config.ZHIPUAI_EMBEDDING_MODEL,
        timeout=config.ZHIPUAI_TIMEOUT
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # 初始化检索器
    vector_retriever = db.as_retriever(search_kwargs={"k": config.VECTOR_RETRIEVER_K})
    all_chunks = dp.chunks
    bm25_retriever = BM25Retriever.from_documents(
        all_chunks, k=config.BM25_RETRIEVER_K, preprocess_func=chinese_preprocess_func
    )
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        num_results=config.HYBRID_RETRIEVER_NUM_RESULTS,
        rrf_k=config.RRF_K
    )
    
    # 初始化LLM
    llm = ChatZhipuAI(
        model=config.ZHIPUAI_MODEL,
        temperature=config.ZHIPUAI_TEMPERATURE,
        api_key=config.ZHIPUAI_API_KEY,
        timeout=config.ZHIPUAI_TIMEOUT
    )
    
    # Pre-build time-query retriever with expanded search scope
    # Use larger K values for time queries to ensure we capture enough date-relevant documents
    time_k = min(config.VECTOR_RETRIEVER_K * 2, len(all_chunks))
    time_vector_retriever = db.as_retriever(search_kwargs={"k": time_k})
    time_bm25_retriever = BM25Retriever.from_documents(
        all_chunks, k=time_k, preprocess_func=chinese_preprocess_func
    )
    time_retriever = HybridRetriever(
        vector_retriever=time_vector_retriever,
        bm25_retriever=time_bm25_retriever,
        num_results=time_k,
        rrf_k=config.RRF_K
    )
    logger.info(f"时间查询检索器预构建完成 (K={time_k})")
    
    # Initialize the Agent (Phase 1: Tool-augmented RAG)
    gem_agent = GemAgent(llm=llm, app_config=config)
    gem_agent.initialize(
        retriever=retriever,
        time_retriever=time_retriever,
        sort_docs_fn=sort_docs_by_date,
    )
    logger.info(f"GemAgent 初始化完成 (运行模式: {RUNTIME_MODE})")
    
    logger.info("组件初始化完成")


# --- Prompt模板 ---
# 问题重写Prompt（用于向量检索优化）
question_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", QUESTION_REWRITE_SYSTEM_PROMPT),
    ("user", QUESTION_REWRITE_USER_PROMPT)
])


# --- 请求体模型 ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question, max 500 characters")
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=20, description="Conversation history, max 20 turns")


# --- RAG核心功能 ---

def get_current_time() -> str:
    """获取当前时间信息"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M")


def is_time_related_query(question: str) -> bool:
    """
    检测问题是否包含时间相关的关键词
    
    Args:
        question: 用户问题
        
    Returns:
        bool: 如果包含时间相关关键词返回True
    """
    time_keywords = [
        '最近', '最新', '即将', '马上', '接下来', '未来', '以后',
        '最近的', '最新一场', '最近一场', '下一场', '即将举办',
        '最新演唱会', '最近演唱会', '即将开始的', '最近的演出',
        'latest', 'recent', 'upcoming', 'next', 'last',
    ]
    # Also match 4-digit year numbers (e.g. "2025", "2026")
    if re.search(r'\b20[2-3]\d\b', question):
        return True
    return any(keyword in question for keyword in time_keywords)


def parse_chinese_date(date_str: str) -> Optional[datetime]:
    """
    解析中文日期格式，支持以下格式：
    - "2023年12月7-9日" -> 返回开始日期 (2023-12-07)
    - "2024年1月6-8日" -> 返回开始日期 (2024-01-06)
    - "2023年12月7日" -> 返回日期 (2023-12-07)
    - "2025年12月26-28日，12月31日" -> 返回最早日期 (2025-12-26)
    - "2026年1月2-4日，1月9-11日" -> 返回最早日期 (2026-01-02)
    
    Args:
        date_str: 中文日期字符串
        
    Returns:
        datetime对象，如果解析失败返回None
    """
    if not date_str:
        return None
    
    try:
        # 匹配格式：YYYY年MM月DD-DD日 或 YYYY年MM月DD日
        # 支持多个日期段，取最早的日期
        pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})(?:-(\d{1,2}))?日'
        matches = re.findall(pattern, date_str)
        
        if matches:
            # 找到所有匹配的日期，取最早的
            dates = []
            for match in matches:
                year = int(match[0])
                month = int(match[1])
                day = int(match[2])
                dates.append(datetime(year, month, day))
            
            if dates:
                return min(dates)  # 返回最早的日期
    except (ValueError, AttributeError) as e:
        logger.debug(f"日期解析失败: {date_str}, 错误: {str(e)}")
    
    return None


def sort_docs_by_date(docs: List[Document], reverse: bool = True) -> List[Document]:
    """
    按日期对文档进行排序
    
    Args:
        docs: 文档列表
        reverse: True表示降序（最新的在前），False表示升序（最旧的在前）
        
    Returns:
        排序后的文档列表
    """
    def get_sort_key(doc: Document) -> datetime:
        # 优先从metadata中获取concert_date
        concert_date = doc.metadata.get('concert_date', '')
        if concert_date:
            parsed_date = parse_chinese_date(concert_date)
            if parsed_date:
                return parsed_date
        
        # 如果metadata中没有，尝试从内容中提取
        # 在内容中查找日期模式
        content = doc.page_content
        date_pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})(?:-(\d{1,2}))?日'
        match = re.search(date_pattern, content)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            try:
                return datetime(year, month, day)
            except ValueError:
                pass
        
        # 如果没有找到日期，返回一个很旧的日期（这样会被排到最后）
        return datetime(1900, 1, 1)
    
    # 分离有日期和无日期的文档
    docs_with_date = []
    docs_without_date = []
    
    for doc in docs:
        date = get_sort_key(doc)
        if date.year > 1900:  # 有有效日期
            docs_with_date.append((doc, date))
        else:
            docs_without_date.append(doc)
    
    # 对有日期的文档按日期排序
    docs_with_date.sort(key=lambda x: x[1], reverse=reverse)
    
    # 合并结果：有日期的在前（已排序），无日期的在后（保持原顺序）
    sorted_docs = [doc for doc, _ in docs_with_date] + docs_without_date
    
    return sorted_docs

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """格式化对话历史为文本格式（用于问题重写）"""
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


def build_messages_with_history(
    question: str,
    history: List[Dict[str, str]],
    context: str,
    current_time: str
) -> List[tuple]:
    """构建包含对话历史的消息列表"""
    messages = [("system", GEM_SYSTEM_PROMPT)]
    
    # 添加对话历史（保留最近N轮，由配置项控制）
    for turn in history[-config.HISTORY_MAX_TURNS:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ["user", "assistant"]:
            messages.append((role, content))
    
    # 添加当前问题
    current_user_message = GEM_USER_PROMPT.format(
        current_time=current_time,
        context=context,
        question=question
    )
    messages.append(("user", current_user_message))
    
    return messages


def enhanced_rag_chain(input_data: Dict[str, any]) -> str:
    """Enhanced RAG chain: question rewrite -> retrieval -> generation with staged error handling"""
    if llm is None or retriever is None:
        raise RuntimeError("组件未初始化，请先启动服务器")
    
    original_question = input_data["question"]
    history = input_data.get("history", [])
    
    # Stage 1: Rewrite question (with fallback to original question on failure)
    try:
        rewritten_question = (question_rewrite_prompt | llm).invoke({
            "current_question": original_question,
            "conversation_history": format_conversation_history(history)
        }).content
        logger.info(f"原始问题: {original_question}")
        logger.info(f"重写后问题: {rewritten_question}")
    except Exception as e:
        logger.warning(f"问题重写失败，使用原始问题进行检索: {str(e)}")
        rewritten_question = original_question
    
    # Stage 2: Retrieve documents (with fallback to vector-only retrieval on failure)
    try:
        is_time_query = is_time_related_query(original_question)
        
        if is_time_query:
            logger.info("检测到时间相关查询，使用时间检索器")
            
            if time_retriever is None:
                raise RuntimeError("时间查询检索器未初始化")
            
            # Use pre-built time retriever instead of creating a new one each time
            retrieved_docs = time_retriever.retrieve(
                query=original_question,
                vector_query=rewritten_question,
                bm25_query=original_question  # BM25 uses original question for better keyword matching
            )
            
            retrieved_docs = sort_docs_by_date(retrieved_docs, reverse=True)
            retrieved_docs = retrieved_docs[:config.HYBRID_RETRIEVER_NUM_RESULTS]
            logger.info(f"时间排序完成，返回前 {len(retrieved_docs)} 个文档")
        else:
            retrieved_docs = retriever.retrieve(
                query=original_question,
                vector_query=rewritten_question,
                bm25_query=original_question  # BM25 uses original question for better keyword matching
            )
    except Exception as e:
        logger.warning(f"混合检索失败，降级为向量检索: {str(e)}")
        try:
            retrieved_docs = db.as_retriever(
                search_kwargs={"k": config.HYBRID_RETRIEVER_NUM_RESULTS}
            ).invoke(rewritten_question)
        except Exception as fallback_e:
            logger.error(f"向量检索也失败: {str(fallback_e)}", exc_info=True)
            raise RuntimeError("检索服务不可用") from fallback_e
    
    logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
    
    # Stage 3: Generate response
    try:
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        messages = build_messages_with_history(
            question=original_question,
            history=history,
            context=context_text,
            current_time=get_current_time()
        )
        
        response_text = llm.invoke(ChatPromptTemplate.from_messages(messages).invoke({})).content
        return response_text
    except Exception as e:
        logger.error(f"LLM生成回答失败: {str(e)}", exc_info=True)
        raise RuntimeError("AI回答生成失败，请稍后重试") from e


# --- API端点 ---
@app.post("/chat_gem", summary="与邓紫棋AI聊天")
async def chat_with_gem(request: QueryRequest):
    """接收用户问题，根据运行模式选择 Agent 或传统 RAG 链处理"""
    try:
        logger.info(f"用户问题: {request.question} (模式: {RUNTIME_MODE})")

        # ── 内容安全检查 ──────────────────────────────────────
        safety_result = check_content_safety(request.question)
        if safety_result.is_violation:
            logger.warning(
                "用户问题被安全检查拦截 [类型: %s, 命中: %s]，已替换为安全话题",
                safety_result.violation_type,
                safety_result.matched_keyword,
            )
            # 将违规问题替换为安全的邓紫棋相关话题
            safe_question = safety_result.safe_replacement
            logger.info(f"安全替代问题: {safe_question}")
        else:
            safe_question = request.question
        # ─────────────────────────────────────────────────────

        if RUNTIME_MODE == "agent" and gem_agent is not None:
            # Agent mode: LLM autonomously decides which tools to call
            response_text = await asyncio.to_thread(
                gem_agent.run,
                safe_question,
                request.history,
            )
        else:
            # Legacy RAG mode: hardcoded if-else retrieval pipeline
            response_text = await asyncio.to_thread(
                enhanced_rag_chain,
                {
                    "question": safe_question,
                    "history": request.history,
                }
            )

        logger.debug(f"AI回答: {response_text[:100]}...")  # Only log first 100 chars at DEBUG level
        return {"answer": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI服务处理请求时发生内部错误: {str(e)}")


@app.post("/chat_gem/stream", summary="与邓紫棋AI聊天（流式响应）")
async def chat_with_gem_stream(request: QueryRequest):
    """SSE streaming endpoint — streams tokens as they are generated."""
    import json as _json

    logger.info(f"[Stream] 用户问题: {request.question} (模式: {RUNTIME_MODE})")

    # ── 内容安全检查 ──
    safety_result = check_content_safety(request.question)
    if safety_result.is_violation:
        logger.warning(
            "[Stream] 用户问题被安全检查拦截 [类型: %s, 命中: %s]",
            safety_result.violation_type, safety_result.matched_keyword,
        )
        safe_question = safety_result.safe_replacement
    else:
        safe_question = request.question

    async def event_generator():
        """Yield SSE-formatted events with true token-level streaming."""
        try:
            if RUNTIME_MODE == "agent" and gem_agent is not None:
                # Use an asyncio.Queue to bridge the sync generator → async stream
                queue: asyncio.Queue = asyncio.Queue()
                _SENTINEL = object()  # marks end of generator
                loop = asyncio.get_event_loop()

                def _producer():
                    """Run the sync generator in a worker thread, push events to queue."""
                    try:
                        for evt in gem_agent.run_stream(safe_question, request.history):
                            # asyncio.Queue is NOT thread-safe; must use call_soon_threadsafe
                            loop.call_soon_threadsafe(queue.put_nowait, evt)
                    except Exception as exc:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {"event": "error", "data": {"message": str(exc)}},
                        )
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

                # Start the blocking producer in a background thread
                loop.run_in_executor(None, _producer)

                # Consume from the queue asynchronously
                while True:
                    evt = await queue.get()
                    if evt is _SENTINEL:
                        break
                    event_type = evt.get("event", "token")
                    data = _json.dumps(evt.get("data", {}), ensure_ascii=False)
                    yield f"event: {event_type}\ndata: {data}\n\n"
            else:
                # Legacy RAG mode — no streaming, fall back to single-shot
                yield f"event: status\ndata: {_json.dumps({'stage': 'generating', 'message': '正在生成回答...'}, ensure_ascii=False)}\n\n"
                response_text = await asyncio.to_thread(
                    enhanced_rag_chain,
                    {"question": safe_question, "history": request.history},
                )
                yield f"event: token\ndata: {_json.dumps({'content': response_text}, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {{}}\n\n"

        except Exception as exc:
            logger.error(f"[Stream] 处理请求时发生错误: {str(exc)}", exc_info=True)
            error_data = _json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/", summary="服务健康检查")
def read_root():
    return {"message": "欢迎使用 G.E.M. AI Chat API，服务运行正常！"}

# --- 启动服务器 ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"正在启动服务器: http://{config.HOST}:{config.PORT}")
    logger.info(f"API文档: http://{config.HOST}:{config.PORT}/docs")
    # Components are now initialized via lifespan manager,
    # so uvicorn can be started directly with "uvicorn rag_server:app" as well.
    uvicorn.run(app, host=config.HOST, port=config.PORT)