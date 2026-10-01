import os

KEY = os.getenv('BAILIAN_API_KEY')

md5_path = "./md5.txt"

collection_name = "rag"

persist_directory = "./chroma_db"

chunk_overlap = 100

chunk_size = 1000

separators = ["\n\n", "\n", " ", "?","!",",",".","，","。"]

max_split_char_num = 1000

similarity_threshold = 2

embedding_model = "text-embedding-v4"

chatModel = "deepseek-flash"

chatKey = os.getenv('DEEPSEEK_API_KEY')
