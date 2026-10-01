from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("BAILIAN_API_KEY"),
)

vector_store = InMemoryVectorStore(embedding)

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
