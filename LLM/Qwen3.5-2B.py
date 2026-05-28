from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language models."},
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=messages,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    presence_penalty=0.0,
)
print(chat_response.choices[0].message.content)