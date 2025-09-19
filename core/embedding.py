from langchain_core.embeddings import Embeddings
import os
from openai import OpenAI
from typing import List, Optional, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch


class BgeM3Embeddings(Embeddings):
    """BAAI通用嵌入模型BGE-M3的封装"""

    def __init__(
            self,
            model_name_or_path: str = "D:\\socar\\socar_agent\\bge-m3",
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
            use_fp16: bool = True
    ):
        """
        初始化BGE-M3模型

        参数：
        - model_name: 模型路径或HuggingFace ID
        - device: 指定设备 (None自动选择)
        - normalize_embeddings: 是否归一化输出向量
        - use_fp16: 是否使用半精度推理
        """
        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=True
        )

        if use_fp16:
            self.model = self.model.half()

        self.normalize = normalize_embeddings
        self.max_seq_length = self.model.get_max_seq_length()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档嵌入批量处理"""
        # 自动处理长文本分块
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """查询嵌入处理"""
        return self.embed_documents([text])[0]


class QwenEmbeddings(Embeddings):

    def __init__(
            self,
            model_name_or_path: str = "./bge-m3",
            device: Optional[str] = None,
            normalize_embeddings: bool = True,
            use_fp16: bool = True
    ):
        """
        初始化BGE-M3模型

        参数：
        - model_name: 模型路径或HuggingFace ID
        - device: 指定设备 (None自动选择)
        - normalize_embeddings: 是否归一化输出向量
        - use_fp16: 是否使用半精度推理
        """
        load_dotenv(".env")
        api_key = os.getenv("DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=api_key,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档嵌入批量处理"""

        completion = self.client.embeddings.create(
            model="text-embedding-v4",
            input=texts,
            dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format="float"
        )
        result = [d.embedding for d in completion.data]
        return result

    def embed_query(self, text: str) -> List[float]:
        """查询嵌入处理"""
        return self.embed_documents([text])[0]


if __name__ == '__main__':
    qwen_embedding = BgeM3Embeddings()
    print(qwen_embedding.embed_query("你好"))