from inspect import cleandoc
from typing import List, Dict
import re
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
# 导入智谱的Embedding
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_chroma import Chroma  # 使用新的包
import config
import os
import uuid
from pathlib import Path
from .adaptive_splitter import AdaptiveSplitter

class DataPreparationModule:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整文档）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
        self.adaptive_splitter = AdaptiveSplitter()  # 自适应分块器


    def load_data(self) -> List[Document]:
        documents = []
        data_path_obj = Path(self.data_path)
        print(data_path_obj)

        # 加载TXT文件
        for md_file in data_path_obj.rglob("*.txt"):
            # 读取文件内容，保持Markdown格式
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 为每个父文档分配唯一ID
            parent_id = str(uuid.uuid4())

            content = self._clean_text(content)

            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent"  # 标记为父文档
                }
            )
            documents.append(doc)

        # 加载JSON文件（演唱会数据）
        for json_file in data_path_obj.rglob("*.json"):
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 为每个父文档分配唯一ID
            parent_id = str(uuid.uuid4())
            
            # 创建Document对象，将JSON转换为文本格式
            # JSON文件作为父文档，但内容会被分块器处理
            doc = Document(
                page_content=json.dumps(json_data, ensure_ascii=False, indent=2),
                metadata={
                    "source": str(json_file),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                    "file_type": "json"  # 标记为JSON文件
                }
            )
            documents.append(doc)

        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        return documents


    def _enhance_metadata(self, doc: Document):
        """增强文档元数据"""
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts

        # 提取数据类别
        category_mapping = {
            'career': '生涯',
            'concert': '演唱会',
            'lyrics': '歌曲'
        }

        # 从文件路径推断分类
        doc.metadata['category'] = '其他'        
        for key, value in category_mapping.items():
            if key in file_path.parts:
                doc.metadata['category'] = value
                if key != 'career':
                    # 处理文件名，移除扩展名
                    file_name = file_path.stem
                    if file_name.endswith('.txt'):
                        file_name = file_name[:-4]
                    elif file_name.endswith('.json'):
                        file_name = file_name[:-5]
                    doc.metadata[key + '_name'] = file_name
                break


    def _clean_text(self, text) -> str:
        # 移除或替换可能导致问题的字符，但保留换行符
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\n]', ' ', text)
        text = text.strip()
        return text


    def chunk_documents(self) -> List[Document]:
        """Markdown结构感知分块"""
        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._text_splitter()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        return chunks


    def _text_splitter(self) -> List[Document]:
        """
        统一使用 AdaptiveSplitter 处理所有文档的分块
        """
        all_chunks = []

        for doc in self.documents:
            parent_id = doc.metadata["parent_id"]
            
            # 检查是否是JSON文件（演唱会数据）
            if doc.metadata.get('file_type') == 'json':
                # 使用 AdaptiveSplitter 处理 JSON 格式的演唱会数据
                json_chunks = self.adaptive_splitter.split_json_concert(doc)
                
                # 为每个子块建立与父文档的关系
                for chunk in json_chunks:
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                    })
                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id
                
                all_chunks.extend(json_chunks)
            else:
                # 处理TXT文件（生涯、歌词）
                type_map = {
                    '生涯': 'career',
                    '歌曲': 'lyrics',
                }
                doc_type = type_map.get(doc.metadata.get('category', '其他'), '其他')
                
                # 准备元数据，传递给分块器
                base_metadata = {
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "source": doc.metadata.get('source', ''),
                    "category": doc.metadata.get('category', '其他'),
                }
                
                # 使用自适应分块器根据文档类型进行分割
                md_chunks = self.adaptive_splitter.split_document(
                    doc.page_content, 
                    doc_type,
                    metadata=base_metadata
                )

                # 为每个子块分配唯一ID并建立父子关系
                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "chunk_index": i  # 在父文档中的位置
                    })
                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id
                
                all_chunks.extend(md_chunks)

        return all_chunks
    

    
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据子块获取对应的父文档（智能去重）"""
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序并构建去重后的父文档列表
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                key=lambda x: parent_relevance[x], reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        return parent_docs



    def create_vector_db(self):
        """
        重建向量数据库
        """
        # 如果指定重建，删除现有数据库
        if os.path.exists(config.CHROMA_PATH):
            import shutil
            shutil.rmtree(config.CHROMA_PATH)
            print("已删除旧数据库，将重新创建。")
        
        # 使用智谱的Embedding模型
        # 注意：需要指定 model 参数，智谱AI的embedding模型通常是 "embedding-2"
        embeddings = ZhipuAIEmbeddings(
            api_key=config.ZHIPUAI_API_KEY,
            model=config.ZHIPUAI_EMBEDDING_MODEL  # 从配置读取embedding模型
        )
        
        # 分批处理文档，避免超过智谱AI的64条限制
        batch_size = config.VECTOR_DB_BATCH_SIZE  # 从配置读取批次大小
        db = None
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            print(f"正在处理第 {i//batch_size + 1} 批，包含 {len(batch)} 个文档...")
            
            try:
                if db is None:
                    # 第一批：创建新的向量数据库
                    db = Chroma.from_documents(batch, embeddings, persist_directory=config.CHROMA_PATH)
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
                            db = Chroma.from_documents([doc], embeddings, persist_directory=config.CHROMA_PATH)
                        else:
                            db.add_documents([doc])
                        print(f"  文档 {i+j+1} 处理成功")
                    except Exception as doc_error:
                        print(f"  文档 {i+j+1} 处理失败: {str(doc_error)}")
                        continue
        
        if db:
            print(f"向量数据库创建成功，并已保存至 '{config.CHROMA_PATH}'。")
            print(f"总共处理了 {len(self.chunks)} 个文档块。")
        else:
            print("向量数据库创建失败。")