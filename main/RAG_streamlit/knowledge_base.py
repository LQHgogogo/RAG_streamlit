import os.path

from datetime import datetime
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data
from langchain_chroma import Chroma
import hashlib

def check_md5(md5_str:str):
    if not os.path.exists(config_data.md5_path):
        open(config_data.md5_path,'w',encoding='utf-8').close() #创建文件
        return False  #未处理过
    else:
        for line in open(config_data.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip()
            if line==md5_str:
                return True
            else:
                return False

def save_md5(md5_str:str):
    with open(config_data.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')


def get_str_md5(input_md5:str,encoding='utf-8'):
    str_bytes = input_md5.encode(encoding=encoding)
    md5 = hashlib.md5()
    md5.update(str_bytes)   #更新md5对象,将字符串转换为字节序列后更新
    return md5.hexdigest()  #返回16进制字符串

class KnowledgeBaseService:
    def __init__(self):
        os.makedirs(config_data.persist_directory,exist_ok=True)  #如果目录不存在,创建目录

        self.chroma = Chroma(
            collection_name=config_data.collection_name,
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=config_data.KEY,
            ),
            persist_directory=config_data.persist_directory
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_data.chunk_size,
            chunk_overlap=config_data.chunk_overlap,
            separators=config_data.separators,
            length_function=len
        )

    def upload_by_str(self,data,filename):
        md5_hex = get_str_md5(data)
        if check_md5(md5_hex):
            return "已存在"

        if len(data) >config_data.max_split_char_num:
            knowledge_chunks=self.splitter.split_text(data)
        else:
            knowledge_chunks=[data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "myself"
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        save_md5(md5_hex)
        return "上传成功"
