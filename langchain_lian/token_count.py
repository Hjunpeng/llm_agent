from typing import List

import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def str_token_counter(text: str) -> int:
    """Count the number of tokens in a string."""
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:

    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
                tokens_per_message
                + str_token_counter(role)
                + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


if __name__ == '__main__':
    print(tiktoken_counter([HumanMessage(content="hello world")]))