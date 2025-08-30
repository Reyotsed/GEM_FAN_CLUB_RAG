# indexing.py
import os
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
# 导入智谱的Embedding
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_chroma import Chroma  # 使用新的包

# --- 1. 配置 ---
# 设置你的智谱AI API Key
os.environ["ZHIPUAI_API_KEY"] = "dda587eac5c949b7b7a8ecc44399ffcd.sACsHiKmuKgFeU5I" 
DATA_PATH = "./gem_data" 
CHROMA_PATH = "./chroma_db_zhipu" # 建议用一个新的路径，避免和旧的混淆

def clean_text(text):
    """清理文本，移除可能导致API错误的特殊字符，但保留换行符"""
    # 移除或替换可能导致问题的字符，但保留换行符
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\n]', ' ', text)
    # 确保文本不为空且长度合适
    text = text.strip()
    return text

def my_text_splitter(text):
    """按照换行符分割文本，每个段落作为一个chunk"""
    # 按换行符分割，并过滤掉空行
    lines = text.split('\n\n')
    chunks = []
    
    print(f"原始文本长度: {len(text)} 字符")
    print(f"分割后得到 {len(lines)} 行")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line:  # 只保留非空行
            chunks.append(line)
            print(f"第 {i+1} 行: {line[:50]}...")
    
    print(f"最终得到 {len(chunks)} 个有效chunk")
    return chunks

def create_vector_db():
    print("开始加载文档...")
    
    # 检查是否已存在向量数据库
    if os.path.exists(CHROMA_PATH):
        print(f"发现已存在的向量数据库: {CHROMA_PATH}")
        response = input("是否要重新创建？(y/N): ").strip().lower()
        if response != 'y':
            print("跳过创建，使用现有数据库。")
            return
        else:
            # 删除旧的数据库目录
            import shutil
            shutil.rmtree(CHROMA_PATH)
            print("已删除旧数据库，将重新创建。")
    
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    if not documents:
        print("在指定目录下未找到任何文档。")
        return

    print("开始进行文本分块...")
    
    # 使用自定义的文本分割器
    all_chunks = []
    
    for doc in documents:
        print(f"\n处理文档: {doc.metadata.get('source', 'unknown')}")
        # 清理文本内容
        cleaned_text = clean_text(doc.page_content)
        
        # 使用自定义分割器按换行符分割
        text_chunks = my_text_splitter(cleaned_text)
        
        # 为每个分割后的文本块创建Document对象
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) > 5:  # 过滤掉过短的文本块
                # 创建新的Document对象，保持原有的metadata
                new_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'chunk_type': 'line_based'
                    }
                )
                all_chunks.append(new_doc)
    
    print(f"\n文档分块完成，共生成 {len(all_chunks)} 个有效文本块")
    
    # 显示前几个chunk作为示例
    print("\n前3个文本块示例:")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"Chunk {i+1}: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)

    print("正在创建向量数据库 (使用智谱Embedding)...")
    # 使用智谱的Embedding模型
    embeddings = ZhipuAIEmbeddings()
    
    # 分批处理文档，避免超过智谱AI的64条限制
    batch_size = 50  # 设置更小的批次大小以确保安全
    db = None
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        print(f"正在处理第 {i//batch_size + 1} 批，包含 {len(batch)} 个文档...")
        
        try:
            if db is None:
                # 第一批：创建新的向量数据库
                db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
                print(f"第 {i//batch_size + 1} 批处理成功")
            else:
                # 后续批次：添加到现有数据库
                db.add_documents(batch)
                print(f"第 {i//batch_size + 1} 批添加成功")
        except Exception as e:
            print(f"第 {i//batch_size + 1} 批处理失败: {str(e)}")
            # 尝试逐个处理文档
            print("尝试逐个处理文档...")
            for j, doc in enumerate(batch):
                try:
                    if db is None:
                        db = Chroma.from_documents([doc], embeddings, persist_directory=CHROMA_PATH)
                    else:
                        db.add_documents([doc])
                    print(f"  文档 {i+j+1} 处理成功")
                except Exception as doc_error:
                    print(f"  文档 {i+j+1} 处理失败: {str(doc_error)}")
                    continue
    
    if db:
        print(f"向量数据库创建成功，并已保存至 '{CHROMA_PATH}'。")
        print(f"总共处理了 {len(all_chunks)} 个文档块。")
    else:
        print("向量数据库创建失败。")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    else:
        # 重新运行此脚本，用智谱的Embedding模型生成新的数据库
        create_vector_db()
