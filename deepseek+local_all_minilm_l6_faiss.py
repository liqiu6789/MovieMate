# 安装依赖 (运行前取消注释)
# !pip install sentence-transformers langchain faiss-cpu requests langchain_huggingface

import os
import logging
from pathlib import Path
from typing import List
from datetime import datetime

import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用新版库

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger("RAGSystem")

# 配置参数
LOCAL_MODEL_PATH = r"C:\Users\Administrator\PycharmProjects\MovieMate\sentence-transformers\all-MiniLM-L6-v2"  # 本地模型路径
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-e330538ce6644cb5aa04785fdd0c9839"  # 替换为你的API Key

class RAGSystem:
    def __init__(self, data_dir: str = "./data"):
        logger.info("Initializing RAG System...")
        start_time = datetime.now()

        # 检查本地模型是否存在
        if not Path(LOCAL_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"模型路径 {LOCAL_MODEL_PATH} 不存在，请先下载模型\n"
                "运行指令：python -c \"from sentence_transformers import SentenceTransformer as ST;"
                f"ST('sentence-transformers/all-MiniLM-L6-v2').save(r'{LOCAL_MODEL_PATH}')\""
            )

        # 初始化嵌入模型
        logger.info(f"Loading embedding model from: {LOCAL_MODEL_PATH}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        logger.info("Embedding model loaded successfully")

        # 加载和处理数据
        logger.info(f"Loading documents from: {data_dir}")
        self.documents = self.load_documents(data_dir)
        logger.info(f"Loaded {len(self.documents)} documents")

        logger.info("Splitting documents into chunks")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = self.split_documents(self.documents)
        logger.info(f"Split into {len(self.chunks)} chunks")

        # 创建向量数据库
        logger.info("Building FAISS vector database")
        self.vector_db = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embeddings
        )
        logger.info(f"VectorDB contains {self.vector_db.index.ntotal} vectors")
        logger.info(f"Initialization completed in {datetime.now() - start_time}")

    def load_documents(self, data_dir: str) -> List[Document]:
        """加载指定目录下的所有txt文件"""
        logger.debug(f"Scanning directory: {data_dir}")
        docs = []
        for p in Path(data_dir).glob("*.txt"):
            if not p.is_file():
                continue
            logger.debug(f"Processing file: {p.name}")
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    text = f.read()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": p.name}
                ))
            except Exception as e:
                logger.error(f"Error loading {p}: {str(e)}")
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为块"""
        return self.text_splitter.split_documents(documents)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        logger.info(f"Retrieving documents for query: '{query}'")
        start_time = datetime.now()

        try:
            query_embedding = self.embeddings.embed_query(query)
            docs = self.vector_db.similarity_search_by_vector(query_embedding, k=k)

            logger.info(f"Retrieved {len(docs)} documents in {datetime.now() - start_time}")
            for i, doc in enumerate(docs, 1):
                logger.debug(f"Doc {i} source: {doc.metadata['source']}")
                logger.debug(f"Doc {i} content: {doc.page_content[:100]}...")
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise

    def generate(self, query: str, context: List[Document]) -> str:
        """调用DeepSeek生成答案"""
        logger.info("Generating answer with DeepSeek API")
        start_time = datetime.now()

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        # 构造prompt
        context_str = "\n".join([doc.page_content for doc in context])
        prompt = f"请根据以下上下文回答问题：\n{context_str}\n\n问题：{query}\n答案："
        logger.debug(f"API Request Prompt:\n{prompt[:500]}...")  # 截断长文本

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        try:
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']

            logger.info(f"API call completed in {datetime.now() - start_time}")
            logger.debug(f"API Response: {result[:500]}...")  # 截断长文本
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    def query(self, question: str) -> str:
        """完整查询流程"""
        logger.info(f"Processing question: '{question}'")
        start_time = datetime.now()

        try:
            relevant_docs = self.retrieve(question)
            answer = self.generate(question, relevant_docs)

            logger.info(f"Query completed in {datetime.now() - start_time}")
            return answer
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return "抱歉，处理请求时出现错误。"

if __name__ == "__main__":
    try:
        logger.info("===== Starting RAG Application =====")
        # 初始化系统
        rag = RAGSystem(data_dir=r"C:\Users\Administrator\PycharmProjects\MovieMate\txt_data")

        # 示例查询
        question = "这段文本提到几个角色，分别是谁"
        logger.info("\n" + "=" * 50)
        logger.info(f"User Question: {question}")

        answer = rag.query(question)

        logger.info("\n" + "=" * 50)
        logger.info("Final Answer:")
        logger.info(answer)

        print(f"\n问题：{question}")
        print(f"答案：{answer}")
    except Exception as e:
        logger.exception("Application crashed with exception:")
        print("系统发生错误，请查看日志文件获取详细信息。")