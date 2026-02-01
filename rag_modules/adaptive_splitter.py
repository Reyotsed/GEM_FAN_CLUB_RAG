#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应文档分块器
根据文档类型采用不同的分块策略
支持 TXT 格式（生涯、歌词）和 JSON 格式（演唱会）数据
"""

import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.schema import Document


class AdaptiveSplitter:
    """
    自适应文档分块器
    统一处理所有类型文档的分块逻辑：
    - 生涯数据（TXT）：按时间线和重要事件分割
    - 歌词数据（TXT）：一首歌一个chunk
    - 演唱会数据（JSON）：每个场次一个chunk
    """
    
    def __init__(self):
        # 不同类型文档的分块参数
        self.splitter_config = {
            "career": {
                "max_chunk_size": 1000,
                "chunk_type": "career_section"
            },
            "lyrics": {
                "max_chunk_size": 600,
                "chunk_type": "lyrics_info"
            },
            "concert": {
                "max_chunk_size": 800,
                "chunk_type": "concert_session"
            }
        }
    
    def split_document(self, text: str, doc_type: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        根据文档类型分割文档
        
        Args:
            text: 文档内容（TXT 格式的文本）
            doc_type: 文档类型 ('career', 'lyrics')
            metadata: 可选的元数据，用于传递给子块
            
        Returns:
            List[Document]: 分割后的文档块列表
        """
        if doc_type == "career":
            chunks = self._career_splitter(text)
        elif doc_type == "lyrics":
            chunks = self._lyrics_splitter(text)
        else:
            # 默认使用通用分块策略
            chunks = self._default_splitter(text)
        
        # 如果提供了元数据，更新所有 chunk 的元数据
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        
        return chunks
    
    def split_json_concert(self, doc: Document) -> List[Document]:
        """
        处理JSON格式的演唱会数据
        每个场次作为一个独立的chunk
        
        Args:
            doc: 包含 JSON 内容的 Document 对象
            
        Returns:
            List[Document]: 分割后的文档块列表
        """
        chunks = []
        try:
            # 解析JSON内容
            json_data = json.loads(doc.page_content)
            
            # 获取演唱会名称
            concert_name = doc.metadata.get('concert_name', '')
            if not concert_name:
                # 从文件名提取
                file_path = Path(doc.metadata.get('source', ''))
                concert_name = file_path.stem.replace('.json', '')
            
            parent_id = doc.metadata.get("parent_id", "")
            
            # 确保json_data是列表
            if not isinstance(json_data, list):
                json_data = [json_data]
            
            # 为每个场次创建一个chunk
            for i, concert_item in enumerate(json_data):
                # 过滤掉created_at和updated_at字段
                filtered_item = {k: v for k, v in concert_item.items() 
                               if k not in ['created_at', 'updated_at']}
                
                # 构建文本内容
                content = self._build_concert_content(filtered_item, concert_name)
                
                # 创建chunk
                chunk = Document(
                    page_content=content,
                    metadata={
                        "chunk_type": "concert_session",
                        "parent_id": parent_id,
                        "chunk_index": i,
                        "source": doc.metadata.get('source', ''),
                        "category": "演唱会",
                        "concert_name": concert_name,
                        "session_id": filtered_item.get('id'),
                        "sequence_range": filtered_item.get('sequence_range', ''),
                        "tour_phase": filtered_item.get('tour_phase', ''),
                        "city": filtered_item.get('city', ''),
                        "venue": filtered_item.get('venue', ''),
                        "concert_date": filtered_item.get('concert_date', ''),
                    }
                )
                
                chunks.append(chunk)
                
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {doc.metadata.get('source', '')} - {str(e)}")
        except Exception as e:
            raise ValueError(f"处理JSON文件时出错: {doc.metadata.get('source', '')} - {str(e)}")
        
        return chunks
    
    def _build_concert_content(self, concert_item: Dict[str, Any], concert_name: str) -> str:
        """
        构建演唱会场次的文本内容
        
        Args:
            concert_item: 场次数据字典
            concert_name: 演唱会名称
            
        Returns:
            str: 格式化的文本内容
        """
        city = concert_item.get('city', '')
        concert_date = concert_item.get('concert_date', '')
        venue = concert_item.get('venue', '').strip() if concert_item.get('venue') else ''
        tour_phase = concert_item.get('tour_phase', '')
        sequence_range = concert_item.get('sequence_range', '')
        country = concert_item.get('country', '')
        notes = concert_item.get('notes', '')
        estimated_audience = concert_item.get('estimated_audience', '')
        status = concert_item.get('status', '')
        
        # 构建更自然、语义化的文本内容
        content_parts = []
        
        # 第一部分：核心信息（自然语言描述）
        if city and concert_date:
            content_parts.append(f"邓紫棋于{concert_date}在{city}举办{concert_name}演唱会")
        elif city:
            content_parts.append(f"邓紫棋在{city}举办{concert_name}演唱会")
        
        # 第二部分：详细信息（结构化描述）
        details = []
        if tour_phase:
            details.append(f"巡演阶段: {tour_phase}")
        if sequence_range:
            details.append(f"场次编号: {sequence_range}")
        if concert_date:
            details.append(f"演出日期: {concert_date}")
        if country:
            details.append(f"国家/地区: {country}")
        if city:
            details.append(f"演出城市: {city}")
        if venue:
            details.append(f"演出场地: {venue}")
        if estimated_audience:
            details.append(f"预计观看人次: {estimated_audience}")
        if notes:
            details.append(f"备注: {notes}")
        if status:
            details.append(f"状态: {status}")
        
        if details:
            content_parts.append("\n详细信息：")
            content_parts.extend(details)
        
        # 第三部分：增强关键词匹配（用于向量检索）
        keyword_phrases = []
        if city:
            keyword_phrases.extend([
                f"{city}演唱会",
                f"{city}站",
                f"邓紫棋{city}演唱会",
                f"{concert_name}{city}演唱会"
            ])
        if venue:
            keyword_phrases.append(f"{venue}演唱会")
            if city and city in venue:
                keyword_phrases.append(f"{city}{venue}")
        
        if keyword_phrases:
            content_parts.append("\n关键词: " + "、".join(keyword_phrases))
        
        return "\n".join(content_parts)
    
    def _career_splitter(self, text: str) -> List[Document]:
        """
        生涯文档分块器 - 按时间线和重要事件分割
        生涯文档包含邓紫棋的成长历程、重要成就等，按时间线分割更合理
        """
        chunks = []
        
        # 按段落分割（双换行符）
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_size = 0
        max_chunk_size = self.splitter_config["career"]["max_chunk_size"]
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 检查是否是时间标题（如"早年生活"、"2008年－2009年：香港出道"）
            if self._is_time_section_header(paragraph):
                # 如果当前块不为空，先保存
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), "career_section"))
                    current_chunk = ""
                    chunk_size = 0
                
                # 时间标题单独成块
                chunks.append(self._create_chunk(paragraph, "career_timeline_header"))
                continue
            
            # 检查段落长度
            if chunk_size + len(paragraph) > max_chunk_size and current_chunk:
                # 当前块已满，保存并开始新块
                chunks.append(self._create_chunk(current_chunk.strip(), "career_section"))
                current_chunk = paragraph
                chunk_size = len(paragraph)
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                chunk_size += len(paragraph)
        
        # 保存最后一个块
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk.strip(), "career_section"))
        
        return chunks

    def _lyrics_splitter(self, text: str) -> List[Document]:
        """
        歌词文档分块器 - 一首歌一个chunk
        每首歌曲作为一个完整的文档块，包含歌曲信息和歌词内容
        """
        chunks = []
        
        # 提取歌曲名称 - 修复解析逻辑
        song_name = ""
        if '歌曲名' in text:
            try:
                # 更健壮的歌曲名提取
                song_line = text.split('歌曲名')[1].split('\n')[0].strip()
                # 移除可能的冒号
                song_name = song_line.replace(':', '').strip()
            except:
                song_name = "未知歌曲"
        else:
            song_name = "未知歌曲"
        
        # 提取歌名、作词作曲作为一个chunk
        song_info = f"歌曲名: {song_name} \n"
        for line in text.split('\n'):
            if '作词' in line:
                song_info += line.replace('歌词', '') + '\n'
            elif '作曲' in line:
                song_info += line.replace('歌词', '') + '\n'
            elif not len(line.strip()) == 0 and line[0] == ' ':
                song_info += line.replace('歌词', '') + '\n'

        chunks.append(self._create_chunk(song_info, "song_info"))
        # 整个文档作为一个chunk，包含歌曲信息和歌词
        if text.strip():
            # 创建包含详细元数据的chunk
            chunk = Document(
                page_content=text.strip(),
                metadata={
                    "chunk_type": "lyrics_info",
                    "song_name": song_name,
                }
            )
            chunks.append(chunk)
        return chunks

    def _default_splitter(self, text: str) -> List[Document]:
        """
        默认分块器 - 通用分块策略
        """
        chunks = []
        paragraphs = text.split('\n\n')
        max_chunk_size = 800
        
        current_chunk = ""
        chunk_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if chunk_size + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk.strip(), "default_section"))
                current_chunk = paragraph
                chunk_size = len(paragraph)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                chunk_size += len(paragraph)
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk.strip(), "default_section"))
        
        return chunks

    def _is_time_section_header(self, text: str) -> bool:
        """判断是否是时间段落标题"""
        # 检查是否包含年份或时间关键词
        time_patterns = [
            r'\d{4}年',  # 年份
            r'早年生活',  # 早年生活
            r'演艺生涯',  # 演艺生涯
            r'早年',      # 早年
            r'童年',      # 童年
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _create_chunk(self, content: str, chunk_type: str) -> Document:
        """创建文档块"""
        return Document(
            page_content=content,
            metadata={
                "chunk_type": chunk_type
            }
        )
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        获取分块统计信息
        
        Args:
            chunks: 文档块列表
            
        Returns:
            dict: 统计信息
        """
        if not chunks:
            return {}
        
        # 统计块类型
        chunk_types = {}
        total_size = 0
        sizes = []
        
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            chunk_size = len(chunk.page_content)  # 直接从内容计算大小
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_size += chunk_size
            sizes.append(chunk_size)
        
        return {
            "total_chunks": len(chunks),
            "total_size": total_size,
            "average_size": total_size / len(chunks) if chunks else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "chunk_types": chunk_types
        }

