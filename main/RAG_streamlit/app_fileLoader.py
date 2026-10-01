"""
基于streamlit的文件加载器
"""
import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("文件加载器")

uploader_files = st.file_uploader(
    "上传文件",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_files is not None:
    filename = uploader_files.name
    filetype = uploader_files.type
    filesize = uploader_files.size/1024/1024   # 单位：MB

    st.subheader(f"文件信息：{filename}，{filetype}，{filesize:.2f}MB")

    file_content = uploader_files.getvalue().decode("utf-8")

    result = st.session_state["service"].upload_by_str(file_content,filename)
    st.write(result)
