from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title='LangChain 服务',
    version='0.1',
    decription='LangChain 服务'
)

add_routes(app,
           ChatOpenAI())

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)

from langchain.schema.runnable import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable


openai = RemoteRunnable("http://127.0.0.1:8000")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个喜欢写故事的助手。"),
        ("system", "写一个故事，主题是{topic}")
    ]
)

chain = prompt | RunnableMap({
    "topic": openai
})
response = chain.batch([{'topic':'猫'}])
print(response)