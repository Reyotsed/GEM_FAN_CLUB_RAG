# get_lyrics.py
# -*- coding: utf-8 -*-
from pyncm import apis
import re
import os

def fetch_lyrics_from_netease(song_id):
    """
    使用PyNCM从网易云音乐获取指定ID歌曲的歌词。
    
    :param song_id: 歌曲的ID (纯数字)
    :return: 包含歌词信息的字典，如果失败则返回None
    """
    print(f"正在获取歌曲ID为 {song_id} 的歌词...")
    
    try:
        # 调用PyNCM的核心函数
        lyrics_data = apis.track.GetTrackLyrics(song_id)
        print(lyrics_data)

        if not lyrics_data or 'lrc' not in lyrics_data or not lyrics_data['lrc']['lyric']:
            print("错误：未能获取到歌词信息，可能是歌曲没有歌词或ID错误。")
            return None
        
        # 提取纯文本歌词
        original_lyric = lyrics_data['lrc']['lyric']
        
        # （可选）清洗歌词，去掉时间戳等信息
        clean_lyric = re.sub(r'\[\d{2}:\d{2}\.\d{2,3}\]', '', original_lyric).strip()

        print("成功获取并清洗歌词！")

        # 获取歌曲名
        song_detail = apis.track.GetTrackDetail(song_id)
        print(song_detail)

        return {
            "song_name": song_detail['songs'][0]['name'],
            "original": original_lyric,
            "clean": clean_lyric,
        }

    except Exception as e:
        print(f"请求过程中发生错误: {e}")
        return None


def generate_rag_data(song_name, lyric):
    # 去除空行：
    lyric = re.sub(r'\n\n', '\n', lyric)
    # 去除行首空格：
    lyric = re.sub(r'^\s+', '', lyric)
    # 去除行尾空格：
    lyric = re.sub(r'\s+$', '', lyric)
    # 拼接：
    data = f"歌曲名: {song_name}\n歌词: {lyric}"
    return data


def get_song_id_list():
    file_path = "./hot_song.json"
    import json
    song_id_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 假设json结构为 {"songs": [ {...}, {...}, ... ]}
            for song in data.get("songs", []):
                song_id = song.get("songId")
                if song_id:
                    song_id_list.append(song_id)
    except Exception as e:
        print(f"读取歌曲ID列表时发生错误: {e}")
    return song_id_list
    
    

if __name__ == '__main__':
    gem_song_id_list = get_song_id_list()

    for gem_song_id in gem_song_id_list:
        
        results = fetch_lyrics_from_netease(gem_song_id)
        
        if results:
            lyric = results['clean']
            song_name = results['song_name']

            # 检查歌词文件是否已存在，若存在则跳过
            lyric_file_path = f"gem_data/lyrics/{song_name}.txt"
            if os.path.exists(lyric_file_path):
                print(f"已存在歌词文件: {lyric_file_path}，跳过。")
                continue

            print(f"歌曲名: {song_name}")
            print(f"歌词: {lyric}")

            data = generate_rag_data(song_name, lyric)
            print(f"RAG数据: {data}")

            if not os.path.exists(f"gem_data/lyrics"):
                os.makedirs(f"gem_data/lyrics")
            with open(f"gem_data/lyrics/{song_name}.txt", "w", encoding="utf-8") as f:
                f.write(data)
            print(f"\n歌词已保存到 gem_data/lyrics/{song_name}.txt")
