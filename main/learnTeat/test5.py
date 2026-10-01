from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt=ChatPromptTemplate(
    [
        ("system", "你是一个边塞诗人"),
        MessagesPlaceholder("history"),
        ("human", "再写一首边塞诗")
    ]
)

history_mes =[
    ("human", "你来作一首唐诗"),
    ("ai", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
]

prompt_text=chat_prompt.invoke({"history":history_mes}).to_string()
print(prompt_text)
