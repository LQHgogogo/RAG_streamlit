from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{last_name}，刚生了一个{gender}，你帮我取个名字"
)

KEY = os.getenv('DEEPSEEK_API_KEY')

model = ChatDeepSeek(
    api_key=KEY,
    model="deepseek-v4-flash",
)

# res_prompt = prompt_template.format(last_name="王", gender="男")
# response = model.invoke(res_prompt)

chain = prompt_template | model
response = chain.invoke(input={"last_name": "王", "gender": "男"})
print(response.content)
print(type(response))
