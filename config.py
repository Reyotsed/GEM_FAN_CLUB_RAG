import os
import logging
from typing import Optional, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== 智谱AI配置 ====================
# 优先从环境变量读取，如果没有则使用默认值（仅用于开发测试）
ZHIPUAI_API_KEY: str = os.getenv(
    "ZHIPUAI_API_KEY", 
    ""  # 默认值为空，强制使用环境变量
)
ZHIPUAI_MODEL: str = os.getenv("ZHIPUAI_MODEL", "glm-3-turbo")
ZHIPUAI_TEMPERATURE: float = float(os.getenv("ZHIPUAI_TEMPERATURE", "0.7"))

# 验证API密钥是否已设置
if not ZHIPUAI_API_KEY:
    raise ValueError(
        "错误：请设置 ZHIPUAI_API_KEY 环境变量。\n"
        "方法1：创建 .env 文件并添加：ZHIPUAI_API_KEY=your_api_key\n"
        "方法2：在终端中运行：export ZHIPUAI_API_KEY=your_api_key"
    )

# ==================== 数据路径配置 ====================
DATA_PATH: str = os.getenv("DATA_PATH", "./gem_data")
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db_zhipu")

# ==================== 日志配置 ====================
# 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE: str = os.getenv("LOG_FILE", "rag_server.log")
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 将字符串日志级别转换为logging常量
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
LOG_LEVEL_VALUE: int = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)

# ==================== 检索器配置 ====================
# 向量检索器返回的文档数量
VECTOR_RETRIEVER_K: int = int(os.getenv("VECTOR_RETRIEVER_K", "5"))
# BM25检索器返回的文档数量
BM25_RETRIEVER_K: int = int(os.getenv("BM25_RETRIEVER_K", "5"))
# 混合检索器最终返回的文档数量
HYBRID_RETRIEVER_NUM_RESULTS: int = int(os.getenv("HYBRID_RETRIEVER_NUM_RESULTS", "4"))
# RRF (Reciprocal Rank Fusion) 常数，用于混合检索器重排序
RRF_K: int = int(os.getenv("RRF_K", "60"))

# ==================== 数据准备配置 ====================
# 向量数据库创建时的批次大小
VECTOR_DB_BATCH_SIZE: int = int(os.getenv("VECTOR_DB_BATCH_SIZE", "50"))

# ==================== CORS配置 ====================
# 从环境变量读取允许的来源，如果没有则使用默认值（仅开发环境）
CORS_ORIGINS: List[str] = os.getenv(
    "CORS_ORIGINS", 
    "http://localhost:3000,http://localhost:8080,http://localhost:5173"
).split(",")
# CORS是否允许凭证
CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
# CORS允许的HTTP方法
CORS_ALLOW_METHODS: str = os.getenv("CORS_ALLOW_METHODS", "*")
CORS_ALLOW_METHODS_LIST: List[str] = (
    ["*"] if CORS_ALLOW_METHODS == "*" else CORS_ALLOW_METHODS.split(",")
)
# CORS允许的HTTP头
CORS_ALLOW_HEADERS: str = os.getenv("CORS_ALLOW_HEADERS", "*")
CORS_ALLOW_HEADERS_LIST: List[str] = (
    ["*"] if CORS_ALLOW_HEADERS == "*" else CORS_ALLOW_HEADERS.split(",")
)

# ==================== FastAPI配置 ====================
API_TITLE: str = os.getenv("API_TITLE", "G.E.M. AI Chat API")
API_DESCRIPTION: str = os.getenv(
    "API_DESCRIPTION",
    "一个由智谱AI驱动、模仿邓紫棋语气的RAG聊天机器人API"
)

# ==================== 服务器配置 ====================
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
