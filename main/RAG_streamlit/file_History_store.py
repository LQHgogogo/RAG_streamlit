import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
import os
import config_data

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

def get_history(session_id):
    return FileChatMessageHistory(session_id,config_data.history_storage_path)

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id
        self.storage_path = storage_path

        self.file_path = os.path.join(self.storage_path,self.session_id)

        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self,messages: Sequence[BaseMessage]):
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages =[]
        for msg in all_messages:
            d = message_to_dict(msg)
            new_messages.append(d)

        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f,ensure_ascii=False,indent=4)

    @property
    def messages(self):
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                data = json.load(f)
                return messages_from_dict(data)
        except(FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f,ensure_ascii=False,indent=4)
