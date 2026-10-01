from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.prompts import PromptTemplate

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

first_prompt_template = PromptTemplate.from_template(
    "我的邻居姓{last_name}，刚生了一个{gender}，你帮我取个名字,"
    "请返回一个json字符串，包含name字段和gender字段，gender字段取值为男或女"
)

second_prompt_template = PromptTemplate.from_template(
    "姓名是{name}性别是{gender}，解析一下含义"
)

KEY = os.getenv('DEEPSEEK_API_KEY')

model = ChatDeepSeek(
    api_key=KEY,
    model="deepseek-v4-flash",
)

# res_prompt = prompt_template.format(last_name="王", gender="男")
# response = model.invoke(res_prompt)

chain = first_prompt_template | model | json_parser | second_prompt_template | model | str_parser
response = chain.invoke(input={"last_name": "王", "gender": "男"})
print(response)
