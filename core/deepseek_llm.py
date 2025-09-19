from core import DEEPSEEK_API_KEY
from langchain_deepseek import ChatDeepSeek

import json
import os

# 初始化DeepSeek模型
# 基础对话模型（支持工具调用/结构化输出）
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",      # 或 "deepseek-reasoner"（专注推理，无工具调用）
    temperature=0,             # 控制输出随机性（0~1）
    max_tokens=1024,            # 响应最大长度
    timeout=30,                 # API超时时间
    max_retries=3  # 增加重试次数
)
