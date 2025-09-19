from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.qwen_llm import llm


system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

model = llm

chain = prompt_template | model | parser

app = FastAPI(
    title='LangChain 服务',
    version='0.1',
    decription='LangChain 服务'
)

from langserve import add_routes

add_routes(
    app,
    chain,
   path="/chain"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
