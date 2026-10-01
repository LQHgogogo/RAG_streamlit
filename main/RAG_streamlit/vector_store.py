from langchain_chroma import Chroma
import config_data

class VectorStoreService:
    def __init__(self,embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config_data.collection_name,
            embedding_function=self.embedding,
            persist_directory=config_data.persist_directory
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config_data.similarity_threshold})
