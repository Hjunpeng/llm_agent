from langchain_core.runnables import RunnableGenerator
from typing import Iterable
from langchain_core.messages import AIMessageChunk, AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain.output_parsers import XMLOutputParser
from langchain.prompts import PromptTemplate
import os
from langchain_core.callbacks import BaseCallbackHandler

# current_dir = os.path.dirname(os.path.abspath(__file__))
#
#
load_dotenv(".env")
api_key = os.getenv("DASHSCOPE_API_KEY")
print(api_key)

model = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-30b-a3b-thinking-2507",
    # openai_api_key=api_key,
    # openapi_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def parse(ai_message: AIMessage):
    return ai_message.content.swapcase()


xml_parser = XMLOutputParser()

def stream_to_parse(chunks: Iterable[AIMessageChunk]):
    for chunk in chunks:
        yield chunk.content.swapcase()


streaming_parse = RunnableGenerator(stream_to_parse)


format_instructions = """
以xml结构返回，使用如下xml 结构
```
<xml>
<movie> 电影1 </movie> 
<movie> 电影2 </movie> 
<xml>
```
"""

prompt = PromptTemplate(
    template="{input}\n{format_instructions}",
    input_variables=["input"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model | xml_parser

output = chain.invoke({"input": "请列出几个童话故事名"})
print(output)