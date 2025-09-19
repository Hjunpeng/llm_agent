from core.qwen_llm import llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the meaning of life?"),
]
#
# print(llm.invoke(message))
# # result = llm.invoke(message)
#
# parser = StrOutputParser()
#
# # parser.invoke(result)
#
# chain = llm | parser
# chain.invoke(message)

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("human", "{text}"),
    ]
)
result = prompt_template.invoke({"text": "I love programming.", "language": "Chinese"})
print(result)
print(result.to_messages())

chain = prompt_template | llm | StrOutputParser()
print(chain.invoke({"text": "I love programming.", "language": "Chinese"}))

