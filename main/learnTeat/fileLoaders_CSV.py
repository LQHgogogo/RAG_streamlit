from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="./data/test_data.csv",
    encoding="utf-8",
    csv_args={
        # 分隔符
        "delimiter": ",",
        # 指定带有分隔符的字段的引号
        "quotechar": '"',
        # 指定字段名
        "filenames": ["id", "name", "age", "email", "department", "salary"],
    }
)

documents = loader.load()

# 一次性加载
for document in documents:
    print(type(document), document)

# 懒加载
for document in loader.lazy_load():
    print(type(document), document)