push # GEM Fan Club RAG

这是一个基于RAG（Retrieval-Augmented Generation）技术的GEM粉丝俱乐部问答系统。

## 项目结构

- `rag_server.py` - RAG服务器主程序
- `load_data.py` - 数据加载和处理脚本
- `word_to_txx.py` - Word文档转换脚本
- `gem_data/` - GEM相关文本数据
- `gem_data_pdf/` - PDF格式数据
- `gem_data_word/` - Word文档数据

## 功能特性

- 智能问答系统
- 支持多种文档格式（PDF、Word、TXT）
- 基于向量数据库的检索增强生成
- 中文自然语言处理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行数据加载脚本：
```bash
python load_data.py
```

2. 启动RAG服务器：
```bash
python rag_server.py
```

## 技术栈

- Python
- ChromaDB (向量数据库)
- 智谱AI (ZhipuAI)
- RAG技术

## 许可证

MIT License
