import os

from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def print_prompt(full_prompt):
    print(full_prompt.to_string())
    return full_prompt

model = ChatDeepSeek(
    model="deepseek-flash",
    api_key=os.getenv('DEEPSEEK_API_KEY'),
)

prompt = PromptTemplate.from_template(
    "你需要根据用户的历史会话信息，回答用户的问题，"
    "对话历史：{history}。"
    "用户输入：{question}，请回答"
)

base_chain = prompt|print_prompt|model|StrOutputParser()

chat_history_store = {}
def get_history(session_id):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

if __name__ == "__main__":

    session_config = {
        "configurable": {
            "session_id": "123",
        }
    }

    response = conversation_chain.invoke(input={"question": "你好"}, config=session_config)
    print(response)
    print(chat_history_store)

    response = conversation_chain.invoke(input={"question": "你是什么模型"}, config=session_config)
    print(response)
    print(chat_history_store)
