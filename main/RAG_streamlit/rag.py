
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

import vector_store
import config_data


class RagService:
    def __init__(self):
        self.vector_store = vector_store.VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config_data.embedding_model,
                dashscope_api_key=config_data.KEY,
            )
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我给你提供的信息为基础，简洁专业地回答用户的问题，参考信息如下：{reference}。"),
                ("human", "用户提问：{input}")
            ]
        )

        self.chat_model = ChatDeepSeek(
            api_key=config_data.chatKey,
            model=config_data.chatModel
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_store.get_retriever()

        def print_prompt(prompt):
            print(prompt)
            print("="*50)
            return prompt

        def __format_doc(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            return "\n".join(doc.page_content+"\n"+doc.metadata for doc in docs)

        chain = (
            {
                "input": RunnablePassthrough(),
                "reference": retriever | self.__format_doc
            }
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        return chain

