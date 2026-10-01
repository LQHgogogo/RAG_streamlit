from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.prompts import PromptTemplate

str_parser = StrOutputParser()
my_func = RunnableLambda(lambda ai_msg:{"name":ai_msg.content})

first_prompt_template = PromptTemplate.from_template(
    "我的邻居姓{last_name}，刚生了一个{gender}，你帮我取个名字,"
    "只需要返回姓名，不用其他内容"
)

second_prompt_template = PromptTemplate.from_template(
    "姓名是{name}，解析一下含义"
)

KEY = os.getenv('DEEPSEEK_API_KEY')

model = ChatDeepSeek(
    api_key=KEY,
    model="deepseek-flash",
)

# res_prompt = prompt_template.format(last_name="王", gender="男")
# response = model.invoke(res_prompt)

chain = first_prompt_template | model | my_func | second_prompt_template | model | str_parser
# chain = first_prompt_template | model | (lambda ai_msg:{"name":ai_msg.content}) | second_prompt_template | model | str_parser
response = chain.invoke(input={"last_name": "王", "gender": "男"})
print(response)
