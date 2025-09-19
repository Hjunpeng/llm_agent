from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("给我讲个关于{text}的笑话")

result = prompt_template.format(text='机器学习')

print(result)

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [("system", "你是一位人工智能助手，你的名字是{name}。"),
     ("human", "你好"), ("ai", "我很好，谢谢！"),
     ("human", "{user_input}"),
     ]
)

message = chat_template.format_messages(name="Bob", user_input="你的名字叫什么？")
print(message)

chat_template = ChatPromptTemplate.from_messages(
    [SystemMessage(content=( "你是一个乐于助人的助手，可以润色内容，使其看起来起来更简单易读。" ) ),
     HumanMessagePromptTemplate.from_template("{text}"),
     ]
)

message = chat_template.format_messages(text="这个句子的运行速度太慢了")


# MessagesPlaceholder

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}.")
        , MessagesPlaceholder(variable_name="chat_history"),
    ]
)

result = prompt_template.invoke({"input_language": "English", "output_language": "French", "chat_history": [HumanMessage(content="I love programming.")]})
print(result)

prompt_template = ChatPromptTemplate.from_messages([ ("system", "You are a helpful assistant"), ("placeholder", "{msgs}")]) # <-- 这是更改的部分 ])
result = prompt_template.invoke({"msgs": [HumanMessage(content="I love programming.")]})
print(result)

# 提示词追加示例(Few-shot prompt templates)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        'question':"你是谁？"
        , 'answer':"我是一个助手。"
    },
{
        'question':"讲个笑话？"
        , 'answer':"我是一个之恶能助手。"
    }
]

example_prompts = PromptTemplate(input_variables=["question", "answer"], template="Q: {question}\nA: {answer}")
print(example_prompts.format(question='测试把', answer='c'))
print('------------------------')
prompt = FewShotPromptTemplate(
    examples=examples
    , example_prompt=example_prompts
    , prefix="请将下面问题翻译成英文"
    , suffix="Q: {input}\nA:"
    , input_variables=["input"]
)
print(prompt.format(input="你是谁？"))

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

embeddings_path = "D:\\socar\\socar_agent\\bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_path)

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(

    # 这是可供选择的示例列表
    examples=examples,
    # 这是用于生成嵌入的嵌入类，该嵌入用于衡量语义相似性。
    embeddings= HuggingFaceEmbeddings(model_name=embeddings_path),
    # 这是向量数据库，用于存储嵌入和示例。
    vectorstore_cls=Chroma,
    # 这是用于选择示例的示例数。
    k=1,
    # 这是用于选择示例的示例选择器。
    example_selector_kwargs={"alpha": 0.0}
    # 这是用于选择示例的示例选择器参数。


)

question = "你是谁？"
selected_examples = example_selector.select_examples({"question": question})
print(f"最相似的示例：{question}")
for example in selected_examples:
    print(examples, "\\n")
    for k, v in example.items():
        print(f"{k}：{v}")

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Q: {question}\nA: {answer}")

prompt = FewShotPromptTemplate(example_selector=example_selector,
                               example_prompt=example_prompt,
                               suffix="问题：{input}", input_variables=["input"] )

print(prompt.format(input="乔治·华盛顿的父亲是谁？"))