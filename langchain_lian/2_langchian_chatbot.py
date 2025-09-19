from core.qwen_llm import llm as model
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, trim_messages
from token_count import tiktoken_counter

# 存储
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=55,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
    allow_partial=True,
    start_on="human"

)


chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# response = chain.invoke(
#     {"messages": [HumanMessage(content="What's my name?")]}
# )

# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# response = chain.invoke(
#     {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
# )
# print(response.content)
#
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="Hi! I'm Jim")],
     "language": "Spanish"},
    config={"configurable": {"session_id": "abc"}},
)
print(response.content)

response = with_message_history.invoke(
    {"messages": [HumanMessage(content="What's my name?")],
     "language": "Spanish"},
    config={"configurable": {"session_id": "abc"}},
)

print(response.content)
for r in with_message_history.stream(
    {"messages": [HumanMessage(content="What's my name?")],
     "language": "Spanish"},
    config={"configurable": {"session_id": "abc"}},
):
    print(r.content, end="|")

