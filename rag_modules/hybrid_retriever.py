#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索器
结合向量检索和BM25关键词检索，提供更全面的检索能力
"""

from typing import List
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    """
    混合检索器
    结合向量检索和BM25关键词检索，使用RRF (Reciprocal Rank Fusion) 进行重排序
    RRF是一种更科学的重排序方法，能够更好地融合不同检索系统的结果
    """
    
    def __init__(self, vector_retriever, bm25_retriever, num_results: int = 3):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            num_results: 返回的结果数量
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.num_results = num_results
    
    def retrieve(self, query: str, vector_query: str = None, bm25_query: str = None) -> List[Document]:
        """
        混合检索
        结合向量检索和BM25检索的结果，使用RRF算法重排序返回最相关的文档
        
        Args:
            query: 默认查询（如果vector_query和bm25_query都未提供，则使用此查询）
            vector_query: 用于向量检索的查询（可选）
            bm25_query: 用于BM25检索的查询（可选）
        """
        # 确定实际使用的查询
        vector_query = vector_query if vector_query is not None else query
        bm25_query = bm25_query if bm25_query is not None else query
        
        # 向量检索
        vector_docs = self.vector_retriever.invoke(vector_query)
        
        # BM25检索
        bm25_docs = self.bm25_retriever.invoke(bm25_query)
        
        # 合并结果
        combined_docs = self._combine_results(vector_docs, bm25_docs)
        
        return combined_docs[:self.num_results]  # 返回前3个结果
    
    def _combine_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 合并向量检索和BM25检索结果
        RRF公式: score = 1/(k + rank)，其中k是常数(通常为60)，rank是排名(从0开始)
        """
        # 创建文档内容到文档的映射（使用内容作为唯一标识，避免重复）
        doc_map = {}
        
        # 处理向量检索结果 - 记录排名
        for i, doc in enumerate(vector_docs):
            # 使用文档内容的前200个字符作为唯一标识，避免重复内容
            doc_key = doc.page_content[:200] + str(doc.metadata.get('chunk_id', ''))
            doc_map[doc_key] = {
                'doc': doc,
                'vector_rank': i,
                'bm25_rank': None
            }
        
        # 处理BM25检索结果 - 记录排名
        for i, doc in enumerate(bm25_docs):
            doc_key = doc.page_content[:200] + str(doc.metadata.get('chunk_id', ''))
            if doc_key in doc_map:
                # 文档已存在，更新BM25排名
                doc_map[doc_key]['bm25_rank'] = i
            else:
                # 新文档
                doc_map[doc_key] = {
                    'doc': doc,
                    'vector_rank': None,
                    'bm25_rank': i
                }
        
        # 计算RRF分数
        scored_docs = []
        k = 60  # RRF常数，通常设为60
        
        for doc_info in doc_map.values():
            rrf_score = 0.0
            
            # 计算向量检索的RRF分数
            if doc_info['vector_rank'] is not None:
                rrf_score += 1.0 / (k + doc_info['vector_rank'])
            
            # 计算BM25检索的RRF分数
            if doc_info['bm25_rank'] is not None:
                rrf_score += 1.0 / (k + doc_info['bm25_rank'])
            
            # 添加RRF分数到元数据
            doc = doc_info['doc']
            doc.metadata = doc.metadata.copy()
            doc.metadata['rrf_score'] = rrf_score
            doc.metadata['vector_rank'] = doc_info['vector_rank']
            doc.metadata['bm25_rank'] = doc_info['bm25_rank']
            
            scored_docs.append((doc, rrf_score))
        
        # 按RRF分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
