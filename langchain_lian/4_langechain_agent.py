from core.my_search import seaxng_search
from langchain.agents import tool
from core.qwen_llm import llm
from langchain_core.messages import HumanMessage

@tool
def seaxng_search_tool(text):
    """输入搜索内容，使用SearXNG进行搜索"""
    return seaxng_search(text)

# res = seaxng_search_tool.invoke('python')
# print(res)

model_with_tools = llm.bind_tools([seaxng_search_tool])
res = model_with_tools.invoke([HumanMessage(content="python")])

print(f"ContentString: {res.content}")
print(f"ToolCalls: {res.tool_calls}")

print('-------------------------------')

from langgraph.prebuilt import create_react_agent

# agent_executor = create_react_agent(llm, [seaxng_search_tool])


# response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
#
# print(response["messages"])

# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="whats the weather in sf?")]}
# ):
#     print(chunk)
#     print("----")

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent_executor = create_react_agent(llm, [seaxng_search_tool], memory=memory)
config = {"configurable": {"thread_id": "abc123"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
