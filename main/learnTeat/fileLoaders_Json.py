from langchain_community.document_loaders import JoplinLoader, JSONLoader
from sympy import false

loader = JSONLoader(
    file_path="./data/stus_lines.json",
    jq_schema=".name",
    text_content=False,   # 是否将json内容作为文本加载
    json_lines=True
)

document = loader.load()
print(document)
