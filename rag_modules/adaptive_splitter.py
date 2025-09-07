#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应文档分块器
根据文档类型采用不同的分块策略
"""

import re
from typing import List
from langchain.schema import Document


class AdaptiveSplitter:
    """
    自适应文档分块器
    根据文档类型（生涯、演唱会、歌词）采用不同的分块策略
    """
    
    def __init__(self):
        # 不同类型文档的分块参数
        self.splitter_config = {
            "career": {
                "max_chunk_size": 1000,
                "chunk_type": "career_section"
            },
            "concert": {
                "max_chunk_size": 800,
                "chunk_type": "concert_info"
            },
            "lyrics": {
                "max_chunk_size": 600,
                "chunk_type": "lyrics_info"
            }
        }
    
    def split_document(self, text: str, doc_type: str) -> List[Document]:
        """
        根据文档类型分割文档
        
        Args:
            text: 文档内容
            doc_type: 文档类型 ('career', 'concert', 'lyrics')
            
        Returns:
            List[Document]: 分割后的文档块列表
        """
        if doc_type == "career":
            return self._career_splitter(text)
        elif doc_type == "concert":
            return self._concert_splitter(text)
        elif doc_type == "lyrics":
            return self._lyrics_splitter(text)
        else:
            # 默认使用通用分块策略
            return self._default_splitter(text)
    
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

    def _concert_splitter(self, text: str) -> List[Document]:
        """
        演唱会文档分块器 - 按表格和重要信息分割
        演唱会文档包含巡演信息、场次列表等，保持表格完整性
        """
        chunks = []
        
        # 按行分割，保持表格结构
        lines = text.split('\n')
        
        current_chunk = ""
        chunk_size = 0
        max_chunk_size = self.splitter_config["concert"]["max_chunk_size"]
        in_table = False
        table_chunk = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_chunk:
                    current_chunk += "\n"
                continue
            
            # 检查是否是表格开始
            if self._is_table_start(line):
                # 保存之前的非表格内容
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), "concert_info"))
                    current_chunk = ""
                
                in_table = True
                table_chunk = line + "\n"
                continue
            
            # 检查是否是表格结束
            if in_table and self._is_table_end(line):
                table_chunk += line
                chunks.append(self._create_chunk(table_chunk.strip(), "concert_table"))
                in_table = False
                table_chunk = ""
                continue
            
            # 在表格中
            if in_table:
                table_chunk += line + "\n"
                continue
            
            # 检查是否是重要信息段落
            if self._is_important_concert_info(line):
                # 保存当前块
                if current_chunk.strip():
                    chunks.append(self._create_chunk(current_chunk.strip(), "concert_info"))
                    current_chunk = ""
                
                # 重要信息单独成块
                chunks.append(self._create_chunk(line, "concert_highlight"))
                continue
            
            # 普通内容处理
            if chunk_size + len(line) > max_chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk.strip(), "concert_info"))
                current_chunk = line
                chunk_size = len(line)
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
                chunk_size += len(line)
        
        # 保存最后一个块
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk.strip(), "concert_info"))
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

    def _is_table_start(self, line: str) -> bool:
        """判断是否是表格开始"""
        table_start_patterns = [
            r'== .* ==',  # 标题：== 巡演地點 ==
            r'\{.*class="wikitable"',  # 表格开始
            r'\|.*\|.*\|',  # 表格行开始
        ]
        
        for pattern in table_start_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _is_table_end(self, line: str) -> bool:
        """判断是否是表格结束"""
        # 简单的表格结束判断：空行或非表格内容
        if not line.strip():
            return True
        
        # 检查是否是表格行
        if re.search(r'\|.*\|', line):
            return False
        
        return True

    def _is_important_concert_info(self, text: str) -> bool:
        """判断是否是重要演唱会信息"""
        important_keywords = [
            '巡演', '演唱会', '首场', '加场', '创下', '记录', '突破',
            '体育场', '体育馆', '连开', '票房', '观看人次'
        ]
        
        for keyword in important_keywords:
            if keyword in text:
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

