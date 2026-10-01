from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

from langchain_deepseek import ChatDeepSeek

KEY = os.getenv("DEEPSEEK_API_KEY")
model=ChatDeepSeek(
    model="deepseek-flash",
    api_key=KEY,
    base_url="https://api.deepseek.com",
)

chat_prompt=ChatPromptTemplate(
    [
        ("system", "你是一个边塞诗人"),
        MessagesPlaceholder("history"),
        ("human", "再写一首边塞诗")
    ]
)

history_mes =[
    ("human", "你来作一首唐诗"),
    ("ai", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
]

chain = chat_prompt | model
print(chain.invoke({"history":history_mes}).content)
