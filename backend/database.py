from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import pandas as pd
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel
import torch
def split_large_csv(input_file, output_dir, chunksize=10000):
    """
    将大 CSV 文件分割成多个小 CSV 文件并保存到指定目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(tqdm(pd.read_csv(input_file, chunksize=chunksize), desc="Splitting large CSV")):
        chunk.to_csv(os.path.join(output_dir, f"chunk_{i}.csv"), index=False)
    print(f"Large CSV split into chunks and saved in {output_dir}.")

def load_csv_chunk(file_path):
    """
    加载单个小 CSV 文件并提取文本内容。
    """
    texts = []
    chunk = pd.read_csv(file_path)
    for _, row in chunk.iterrows():
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

def load_processed_files(record_file):
    """
    加载已处理的文件记录。
    """
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_processed_file(record_file, file_name):
    """
    将已处理的文件名保存到记录文件中。
    """
    with open(record_file, "a") as f:
        f.write(f"{file_name}\n")


def build_knowledge_base(input_file, cache_dir, vectorstore_dir="faiss_vectorstore", rebuild_cache=False):
    """
    增量构建知识库，支持缓存分块文件和向量数据库。
    """
    # 确保向量库目录存在
    os.makedirs(vectorstore_dir, exist_ok=True)
    record_file = os.path.join(vectorstore_dir, "processed_files.txt")

    # 分割大 CSV 的逻辑保持不变
    if rebuild_cache or not os.path.exists(cache_dir):
        split_large_csv(input_file, cache_dir)

    # 初始化 Embedding 模型
    model_path = r"C:\Users\Administrator\Downloads\MovieMate-main\MovieMate-main\sentence-transformers\all-MiniLM-L6-v2"  # 确保路径正确
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": device})

    # 加载现有向量库（如果存在）
    vectorstore = None
    index_path = os.path.join(vectorstore_dir, "index.faiss")  # FAISS 自动生成的文件
    if os.path.exists(index_path):
        print("Loading existing vectorstore...")
        print(index_path)
        vectorstore = FAISS.load_local(
            vectorstore_dir,  # 传入目录路径
            embeddings,
            allow_dangerous_deserialization=True
        )

    # 处理未处理的 CSV 分块
    processed_files = load_processed_files(record_file)
    for file_name in tqdm(os.listdir(cache_dir), desc="Processing cached CSV chunks"):
        if file_name.endswith(".csv") and file_name not in processed_files:
            file_path = os.path.join(cache_dir, file_name)
            try:
                texts = load_csv_chunk(file_path)
                chunks = split_text(texts)

                if vectorstore is None:
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                else:
                    vectorstore.add_documents(chunks)

                save_processed_file(record_file, file_name)
                print(f"Processed {file_name}")

                # 每次添加后保存
                vectorstore.save_local(vectorstore_dir)  # 保存到目录
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

    print("Knowledge base update complete.")
    return vectorstore
