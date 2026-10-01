from kubernetes.leaderelection import example
from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
KEY=os.getenv('DEEPSEEK_API_KEY')

model = ChatDeepSeek(
    api_key=KEY,
    model="deepseek-flash",
)

example_prompt = FewShotPromptTemplate.from_template(
    "单词：{word}，反义词：{antonym}"
)

examples_data = [
    {"word": "好", "antonym": "坏"},
    {"word": "大", "antonym": "小"},
]

example_text = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples_data,
    prefix="根据以下示例，填写单词的反义词：",
    suffix="单词：{word}的反义词是：",
    input_variables=["word"],
)

text = example_text.invoke(input={"word": "好"}).to_string()

response=model.invoke(text)
print(response.content)
