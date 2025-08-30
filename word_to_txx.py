# word_to_txt_converter.py
import os
import docx  # 导入python-docx库

def convert_docs_in_folder(word_folder_path, txt_output_path):
    """
    遍历指定文件夹中的所有 .docx 文件，并将它们转换为 .txt 文件。
    """
    # 确保输出文件夹存在
    if not os.path.exists(txt_output_path):
        os.makedirs(txt_output_path)
        print(f"创建输出文件夹: {txt_output_path}")

    # 遍历Word文件夹中的所有文件
    for filename in os.listdir(word_folder_path):
        if filename.lower().endswith('.docx'):
            word_path = os.path.join(word_folder_path, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(txt_output_path, txt_filename)
            
            print(f"正在处理: {word_path} -> {txt_path}")

            try:
                # --- 核心转换逻辑 ---
                # 打开Word文档
                doc = docx.Document(word_path)
                
                # 创建一个列表来存储每个段落的文本
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                
                # 使用换行符将所有段落连接成一个单一的字符串
                final_text = '\n'.join(full_text)

                # 将提取的文本写入TXT文件
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(final_text)
                
                print(f"成功转换: {filename}")

            except Exception as e:
                print(f"!!!!!! 处理文件 {filename} 时发生错误: {e}")

if __name__ == "__main__":
    # --- 配置你的路径 ---
    # 存放你所有 .docx 资料的文件夹
    WORD_SOURCE_FOLDER = "./gem_data_word" 
    # 转换后TXT文件的存放文件夹 (这就是你的RAG data source)
    TXT_OUTPUT_FOLDER = "./gem_data" 

    # 检查Word源文件夹是否存在
    if not os.path.exists(WORD_SOURCE_FOLDER):
         os.makedirs(WORD_SOURCE_FOLDER)
         print(f"创建了Word源文件夹 '{WORD_SOURCE_FOLDER}'。请将你的 .docx 文件放入其中再运行此脚本。")
    else:
        convert_docs_in_folder(WORD_SOURCE_FOLDER, TXT_OUTPUT_FOLDER)