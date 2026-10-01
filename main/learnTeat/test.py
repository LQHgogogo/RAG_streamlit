from openai import OpenAI
import os

API_KEY = os.getenv('DEEPSEEK_API_KEY')

client = OpenAI(
    api_key=API_KEY,
    base_url='https://api.deepseek.com'
)

response = client.chat.completions.create(
    model='deepseek-v4-flash',
    messages=[
        {"role": "user", "content": "你是什么模型，说出具体版号"}
    ]
)

print(response.choices[0].message.content)