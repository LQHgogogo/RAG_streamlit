from networkx.algorithms import similarity

KEY = "sk-ws-H.PHDXRMY.2Hg5.MEQCIDDbSDFlqZ6nPnAlw6_LiFw4HUMBsW9bELUpNh6vKYGtAiAA-m3ecXHhwHc34rEQnxM19umrnaSAIWc9lB-acA29kA"

md5_path = "./md5.txt"

collection_name = "rag"

persist_directory = "./chroma_db"

chunk_overlap = 100

chunk_size = 1000

separators = ["\n\n", "\n", " ", "?","!",",",".","，","。"]

max_split_char_num = 1000

similarity_threshold = 2
