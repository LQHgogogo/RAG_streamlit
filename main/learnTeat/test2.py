from langchain_community.embeddings import DashScopeEmbeddings
import os

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("BAILIAN_API_KEY"),
)

vector = embeddings.embed_query("你好")
print(len(vector))

# openai写法：

# from langchain_openai import OpenAIEmbeddings
# import os
#
# KEY = os.getenv("DASHSCOPE_API_KEY")
#
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-v3",
#     api_key=KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
#
# vector = embeddings.embed_query("你好")
# print(len(vector))