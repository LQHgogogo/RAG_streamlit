from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("BAILIAN_API_KEY"),
)

vector = embedding.embed_query("你好")
print(len(vector))

# openai写法：

# from langchain_openai import OpenAIEmbeddings
# import os

# KEY = os.getenv("DASHSCOPE_API_KEY")

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-v3",
#     api_key=KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# vector = embeddings.embed_query("你好")
# print(len(vector))

vector_store = Chroma(
    collection_name="test",          # 当前向量数据表明
    embedding_function=embedding,
    persist_directory="data/chroma_vectors",
)

loader = TextLoader(
    "data/agent介绍.text",
    encoding="utf-8"
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
# 分割document对象
split_docs = splitter.split_documents(docs)

vector_store.add_documents(
    documents=split_docs,
    ids=[f"agent_{i}" for i in range(1, len(split_docs)+1)]
)

result = vector_store.similarity_search(
    "Agent是什么",
    k=1
)

for doc in result:
    print(doc.page_content)
