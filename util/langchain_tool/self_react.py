from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.json import parse_json_markdown
from langchain.agents.agent import AgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.json_chat.prompt import TEMPLATE_TOOL_RESPONSE
from agent_tool import get_word_length, seaxng_search_tool
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv(".env")
api_key = os.getenv("DASHSCOPE_API_KEY")
print(api_key)

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-30b-a3b-thinking-2507",
)


promptTemplate = """尽可能的帮助用户回答任何问题。
您可以使用以下工具来帮忙解决问题， 优先使用工具解答，若无法使用工具，也可以直接回答：
{tools}
回复格式说明
-————————————————————————
回复我时，请以以下两种格式之一输出回复：

选项1： 如果您希望人类使用工具， 请使用此选项。
采用以下json 模式格式化的回复内容：

```json
{{
    "reason": string, \\叙述使用工具的原因
    “action”: string, \\ 要使用的工具，必须是{tool_names}之一
    “action_input”: string \\工具的输入
}}
```

选项2： 如果您认为你已经有答案或者已经通过使用工具找到了答案，想直接对人类作出反应，采用以下json模式格式化回复内容：
```json
{{
    “action”: ”Final Answer“
    “answer”: string \\最终答复问题的答案放在这里
}}
```

用户的输入
——————————————————————————————————————
这是用户的输入（请记住通过单个选项，以JOSN模式格式化的回复内容，不要回复其它内容）：
{input}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", promptTemplate),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)


tools = [get_word_length, seaxng_search_tool]

prompt = prompt.partial(
    tools=render_text_description(list(tools)),
    tool_names=",".join([tool.name for tool in tools])
)

print(prompt)


TEMPLATE_TOOL_RESPONSE = """工具响应：
------------------
{observation}

用户输入：
------------------
请根据工具的响应判断，是否能够回答问题：
{input}
请根据工具响应的内容，思考接下来的回复，回复格式严格按照前面所说的2种json回复格式，选择其中1种进行回复，请记住通过单个选项，以JOSN模式格式化的回复内容，不要回复其它内容。
"""


def format_log_to_message(
        query,
        intermediate_steps,
        template_to_tool_response
):
    print(33333, query, intermediate_steps)
    thoughts = []
    for action, observation in intermediate_steps:
        thoughts.append(AIMessage(content=action.log))
        human_message = HumanMessage(content=template_to_tool_response.format(observation=observation, input=query))
        thoughts.append(human_message)
    return thoughts


class JSONAgentOutputParser(AgentOutputParser):
    def parse(self, text: str):
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                logging.warning("Got multiple action response:%s", response)
                response = response[0]
            if response.get("action") == "Final Answer":
                return AgentFinish(
                    return_values={"output": response.get("answer")},
                    log=text,
                )
            else:
                return AgentAction(
                    tool=response.get("action"),
                    tool_input=response.get("action_input"),
                    log=text,
                )
        except Exception as e:
            raise OutputParserException(
                f"Could not parse LLM output: {text} because {e}"
            )

    @property
    def _type(self):
        return "json-agent"


agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_message(x['input'],
                                                         x["intermediate_steps"],
                                                         template_to_tool_response=TEMPLATE_TOOL_RESPONSE)
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# print(agent_executor.invoke({"input":"刘德华的老婆多大年龄"}))


store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_agent = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",  # 与提示词中的变量名一致
    )


print("第一次对话:")
result1 = conversational_agent.invoke(
    {"input": "刘德华的老婆多大年龄"},
    config={"configurable": {"session_id": 1123}}
)
print(result1['output'])

print("\n第二次对话 (应该记得之前的对话):")
result2 = conversational_agent.invoke(
    {"input": "刚才问的是谁"},
    config={"configurable": {"session_id": 1123}}
)
print(result2['output'])
