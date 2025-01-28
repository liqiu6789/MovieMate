from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import pandas as pd
def load_csv(file_path):
    """
    加载 CSV 文件并提取文本内容。
    """
    df = pd.read_csv(file_path)
    texts = []

    for _, row in df.iterrows():
        text = (
            f"电影ID: {row['电影ID']}, 电影名: {row['电影名']}, 别名: {row['别名']}, 演员: {row['演员']}, "
            f"导演: {row['导演']}, 豆瓣评分: {row['豆瓣评分']}, 类型: {row['类型']}, 剧情简介: {row['剧情简介']}, "
            f"上映日期: {row['上映日期']}, 地区: {row['地区']}, 标签: {row['标签']}"
        )
        texts.append(text)

    return texts

def split_text(texts, chunk_size=1000, chunk_overlap=200):
    """
    将文本分割成适合处理的块。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.create_documents(texts)
    return chunks

def build_knowledge_base(file_path):
    """
    从 CSV 文件构建知识库。
    """
    # 加载 CSV 文件
    texts = load_csv(file_path)
    print(f"Loaded {len(texts)} text entries from CSV.")

    # 文本分块
    chunks = split_text(texts)
    print(f"Split into {len(chunks)} text chunks.")

    # 嵌入向量化并构建向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Knowledge base built successfully.")
    return vectorstore