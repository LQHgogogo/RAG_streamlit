import json
import os

from typing import Sequence, List
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class FileMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id
        self.storage_path = storage_path

        self.file_path = os.path.join(self.storage_path, self.session_id)

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]):
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = []
        for message in all_messages:
            d = message_to_dict(message)
            new_messages.append(d)

        # new_messages = [message_to_dict(message) for message in all_messages]

        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f,ensure_ascii=False,indent=2)

    @property  #property装饰器将messages方法变成成员属性
    def messages(self) -> List[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f)
                return messages_from_dict(messages_data)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self):
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f,ensure_ascii=False,indent=2)


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

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHAT_HISTORY_DIR = os.path.join(_CURRENT_DIR, "chat_history")

def get_history(session_id):
    return FileMessageHistory(session_id, _CHAT_HISTORY_DIR)

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

    response = conversation_chain.invoke(input={"question": "你是什么模型"}, config=session_config)

    res = conversation_chain.invoke(input={"question": "我上次问了你什么"}, config=session_config)