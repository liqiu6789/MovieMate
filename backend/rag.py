from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.schema import Generation
from typing import Optional, List, Mapping, Any
import requests
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


class DeepSeekLLM(LLM):
    """自定义DeepSeek语言模型封装"""

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            ** kwargs: Any,
    ) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512)
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            print("调用deepseek成功")
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            raise ValueError(f"API调用失败: {str(e)}")


def query_knowledge_base(vectorstore, query):
    # 初始化自定义DeepSeek模型
    llm = DeepSeekLLM()

    # 构建问答链
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # 执行查询
    result = qa({"query": query})

    # 解析结果
    answer = result["result"]
    source_docs = result["source_documents"]
    print(f"上下文：{source_docs}")
    return answer