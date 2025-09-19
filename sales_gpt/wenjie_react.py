import os
import re
import logging
from dotenv import load_dotenv

from typing import Any, Callable, Dict, List, Union

from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.prompts.base import StringPromptTemplate
from langchain_community.llms import BaseLLM
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from ollama import Client
from typing import List, Union, Optional, Dict
from langchain_core.embeddings import Embeddings

from langchain_core.output_parsers.json import parse_json_markdown
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from my_llm import StageAnalyzerChain, SalesConversationChain, get_tools, llm
# 定义自定义提示模板


class CustomPromptTemplateForTools(StringPromptTemplate):
    # 要使用的模板
    template: str
    ############## NEW ######################
    # 可用工具列表
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction、Observation 元组）
        # 以特定方式格式化它们
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 将 agent_scratchpad 变量设置为该值
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # 从提供的工具列表创建一个工具变量
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # 为提供的工具创建工具名称列表
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


# 定义自定义输出解析器
class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # 更改 salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(text)
            print("-------")
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):
                # gpt Turbo 经常忽略发出单个操作的指令
                logging.warning("Got multiple action responses: %s", response)
                response = response[0]
            if response["isNeedTools"] == "False":
                return AgentFinish({"output": response["output"]}, text)
            else:
                return AgentAction(
                    response["action"], response.get("action_input", {}), text
                )
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "sales-agent"


SALES_AGENT_TOOLS_PROMPT = """
永远不要忘记您的名字是{salesperson_name}。 您担任{salesperson_role}。
您在名为 {company_name} 的公司工作。 {company_name} 的业务如下：{company_business}。
公司价值观如下。 {company_values}
您联系潜在客户是为了{conversation_purpose}
您联系潜在客户的方式是{conversation_type}

如果系统询问您从哪里获得用户的联系信息，请说您是从公共记录中获得的。
保持简短的回复以吸引用户的注意力。 永远不要列出清单，只给出答案。
只需打招呼即可开始对话，了解潜在客户的表现如何，而无需在您的第一回合中进行推销。
通话结束后，输出<END_OF_CALL>
在回答之前，请务必考虑一下您正处于对话的哪个阶段：

1：介绍：通过介绍您自己和您的公司来开始对话。 保持礼貌和尊重，同时保持谈话的语气专业。 你的问候应该是热情的。 请务必在问候语中阐明您打电话的原因。
2：资格：通过确认潜在客户是否是谈论您的产品/服务的合适人选来确定潜在客户的资格。 确保他们有权做出采购决定。
3：价值主张：简要解释您的产品/服务如何使潜在客户受益。 专注于您的产品/服务的独特卖点和价值主张，使其有别于竞争对手。
4：需求分析：提出开放式问题以揭示潜在客户的需求和痛点。 仔细聆听他们的回答并做笔记。
5：解决方案展示：根据潜在客户的需求，展示您的产品/服务作为可以解决他们痛点的解决方案。
6：异议处理：解决潜在客户对您的产品/服务可能提出的任何异议。 准备好提供证据或推荐来支持您的主张。
7：成交：通过提出下一步行动来要求出售。 这可以是演示、试验或与决策者的会议。 确保总结所讨论的内容并重申其好处。
8：结束对话：潜在客户必须离开去打电话，潜在客户不感兴趣，或者销售代理已经确定了下一步。

工具：
------

{salesperson_name} 有权使用以下工具：

{tools}

要使用工具，请使用以下JSON格式回复：

```
{{
    "isNeedTools":"True", //需要使用工具
    "action": str, //要采取操作的工具名称，应该是{tool_names}之一
    "action_input": str, // 使用工具时候的输入，始终是简单的字符串输入
}}

```

如果行动的结果是“我不知道”。 或“对不起，我不知道”，那么您必须按照下一句中的描述对用户说这句话。
当您要对人类做出回应时，或者如果您不需要使用工具，或者工具没有帮助，您必须使用以下JSON格式：

```
{{
    "isNeedTools":"False", //不需要使用工具
    "output": str, //您的回复，如果以前使用过工具，请改写最新的观察结果，如果找不到答案，请说出来
}}
```

您必须根据之前的对话历史记录以及当前对话的阶段进行回复。
一次仅生成一个响应并仅充当 {salesperson_name},响应的格式必须严格按照上面的JSON格式回复，不需要加上//后面的注释。

开始！

当前对话阶段：
{conversation_stage}

之前的对话记录：
{conversation_history}

回复：
{agent_scratchpad}
"""


# class SalesGPT(Chain, BaseModel):
class SalesGPT(Chain):
    """销售代理的控制器模型。"""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)

    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    conversation_stage_dict: Dict = {
        "1": "介绍：通过介绍您自己和您的公司来开始对话。 保持礼貌和尊重，同时保持谈话的语气专业。 你的问候应该是热情的。 请务必在问候语中阐明您联系潜在客户的原因。",
        "2": "资格：通过确认潜在客户是否是谈论您的产品/服务的合适人选来确定潜在客户的资格。 确保他们有权做出采购决定。",
        "3": "价值主张：简要解释您的产品/服务如何使潜在客户受益。 专注于您的产品/服务的独特卖点和价值主张，使其有别于竞争对手。",
        "4": "需求分析：提出开放式问题以揭示潜在客户的需求和痛点。 仔细聆听他们的回答并做笔记。",
        "5": "解决方案展示：根据潜在客户的需求，展示您的产品/服务作为可以解决他们的痛点的解决方案。",
        "6": "异议处理：解决潜在客户对您的产品/服务可能提出的任何异议。 准备好提供证据或推荐来支持您的主张。",
        "7": "结束：通过提出下一步行动来寻求销售。 这可以是演示、试验或与决策者的会议。 确保总结所讨论的内容并重申其好处。",
    }

    salesperson_name: str = "小陈"
    salesperson_role: str = "问界汽车销售经理"
    company_name: str = "赛力斯汽车"
    company_business: str = "问界是赛力斯发布的全新豪华新能源汽车品牌，华为从产品设计、产业链管理、质量管理、软件生态、用户经营、品牌营销、销售渠道等方面全流程为赛力斯的问界品牌提供了支持，双方在长期的合作中发挥优势互补，开创了联合业务、深度跨界合作的新模式。"
    company_values: str = "赛力斯汽车专注于新能源电动汽车领域的研发、制造和生产，旗下主要产品包括问界M5、问界M7、问界M9等车型，赛力斯致力于为全球用户提供高性能的智能电动汽车产品以及愉悦的智能驾驶体验。"
    conversation_purpose: str = "了解他们是否希望通过购买拥有智能驾驶的汽车来获得更好的驾乘体验"
    conversation_type: str = "电话"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        #第一步，初始化智能体
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    def determine_conversation_stage(self):
        if len(self.conversation_history) > 0:
            conversation_history = '"\n"'.join(self.conversation_history)
        else:
            conversation_history = '"\n暂无历史对话"'
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history=conversation_history,
            current_conversation_stage=self.current_conversation_stage,
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    def step(self):
        self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """运行销售代理的一步。"""

        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
            )

        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
            )

        # Add agent's response to conversation history
        print(f"{self.salesperson_name}: ", ai_message.rstrip("<END_OF_TURN>"))
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """初始化 SalesGPT 控制器。"""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:
            sales_agent_executor = None

        else:
            product_catalog = kwargs["product_catalog"]
            tools = get_tools(product_catalog)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # 这省略了“agent_scratchpad”、“tools”和“tool_names”变量，因为它们是动态生成的
                # 这包括“intermediate_steps”变量，因为这是需要的
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # 警告：此输出解析器尚不可靠
            ## 它对 LLM 的输出做出假设，这可能会破坏并引发错误
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )


# 设置您的代理

# 对话阶段 - 可以修改
conversation_stages = {
    "1": "介绍：通过介绍您自己和您的公司来开始对话。 保持礼貌和尊重，同时保持谈话的语气专业。 你的问候应该是热情的。 请务必在问候语中阐明您联系潜在客户的原因。",
    "2": "资格：通过确认潜在客户是否是谈论您的产品/服务的合适人选来确定潜在客户的资格。 确保他们有权做出采购决定。",
    "3": "价值主张：简要解释您的产品/服务如何使潜在客户受益。 专注于您的产品/服务的独特卖点和价值主张，使其有别于竞争对手。",
    "4": "需求分析：提出开放式问题以揭示潜在客户的需求和痛点。 仔细聆听他们的回答并做笔记。",
    "5": "解决方案展示：根据潜在客户的需求，展示您的产品/服务作为可以解决他们的痛点的解决方案。",
    "6": "异议处理：解决潜在客户对您的产品/服务可能提出的任何异议。 准备好提供证据或推荐来支持您的主张。",
    "7": "结束：通过提出下一步行动来要求出售。 这可以是演示、试验或与决策者的会议。 确保总结所讨论的内容并重申其好处。",
}


# 代理特征 - 可以修改
config = dict(
    salesperson_name="小陈",
    salesperson_role="问界汽车销售经理",
    company_name="赛力斯汽车",
    company_business="问界是赛力斯发布的全新豪华新能源汽车品牌，华为从产品设计、产业链管理、质量管理、软件生态、用户经营、品牌营销、销售渠道等方面全流程为赛力斯的问界品牌提供了支持，双方在长期的合作中发挥优势互补，开创了联合业务、深度跨界合作的新模式。",
    company_values="赛力斯汽车专注于新能源电动汽车领域的研发、制造和生产，旗下主要产品包括问界M5、问界M7、问界M9等车型，赛力斯致力于为全球用户提供高性能的智能电动汽车产品以及愉悦的智能驾驶体验。",
    conversation_purpose="了解他们是否希望通过购买拥有智能驾驶的汽车来获得更好的驾乘体验",
    conversation_history=["你好，我是来自问界汽车销售经理的小陈。","你好。"],
    conversation_type="电话",
    conversation_stage=conversation_stages.get(
        "1",
        "介绍：通过介绍您自己和您的公司来开始对话。 保持礼貌和尊重，同时保持谈话的语气专业。",
    ),
    use_tools=True,
    product_catalog="sample_product_catalog.txt",
)


sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)