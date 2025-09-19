import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(".env")
api_key = os.getenv("DASHSCOPE_API_KEY")
print(api_key)

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-30b-a3b-thinking-2507",
    temperature=0.3,
)