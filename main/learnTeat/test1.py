from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.messages import HumanMessage, AIMessage

KEY = os.getenv('DEEPSEEK_API_KEY')

model = ChatDeepSeek(
    model='deepseek-v4-flash',
    api_key=KEY,
)

messages = [
    AIMessage(content='你好'),
    HumanMessage(content='你是什么模型')
]

res = model.stream(input=messages)
for chunk in res:
    print(chunk.content, end='',flush=True)
