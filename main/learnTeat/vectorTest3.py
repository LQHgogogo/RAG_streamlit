import os
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

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
split_docs = splitter.split_documents(docs)

embedding = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("BAILIAN_API_KEY"),
)

model = ChatDeepSeek(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    model="deepseek-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我给你提供的信息为基础，回答用户的问题，参考信息如下：{reference}。"),
        ("human", "用户提问：{input}")
    ]
)

vector_store = Chroma(
    collection_name="test",
    embedding_function=embedding,
    persist_directory="data/chroma_vectors2"
)

vector_store.add_documents(split_docs)

user_input = "agent是什么"

result = vector_store.similarity_search(user_input)

reference = [doc.page_content for doc in result]

def print_prompt(prompt):
    print(prompt.to_string())
    print("="*50)
    return prompt

def list_to_dict(lst):
    if not lst:
        return "无相关信息"
    return "\n".join(doc.page_content for doc in lst)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chain = (
    {"input": RunnablePassthrough(), "reference": retriever | list_to_dict}
    | prompt
    | print_prompt
    | model
    | StrOutputParser()
)

print(chain.invoke(user_input))
