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

def build_knowledge_base(input_file, cache_dir, vectorstore_path="vectorstore.faiss", rebuild_cache=False):
    """
    增量构建知识库，支持缓存分块文件和向量数据库。
    """
    # 记录已处理文件的文件名
    record_file = os.path.join(os.path.dirname(vectorstore_path), "processed_files.txt")

    # 如果缓存目录不存在或需要重建缓存，则分割大文件
    if rebuild_cache or not os.path.exists(cache_dir):
        split_large_csv(input_file, cache_dir)

    # 使用 Hugging Face 的嵌入模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检查是否有 GPU
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    # 如果向量数据库已存在，则加载；否则创建新的
    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True  # 允许反序列化
        )
    else:
        print("Creating new vectorstore...")
        vectorstore = None

    # 加载已处理的文件记录
    processed_files = load_processed_files(record_file)

    # 遍历缓存目录中的每个小 CSV 文件，增量构建向量数据库
    for file_name in tqdm(os.listdir(cache_dir), desc="Processing cached CSV chunks"):
        if file_name.endswith(".csv") and file_name not in processed_files:
            file_path = os.path.join(cache_dir, file_name)
            texts = load_csv_chunk(file_path)
            chunks = split_text(texts)

            # 增量添加向量到向量数据库
            if vectorstore is None:
                vectorstore = FAISS.from_documents(chunks, embeddings)
            else:
                vectorstore.add_documents(chunks)

            # 保存已处理的文件记录
            save_processed_file(record_file, file_name)
            print(f"Processed and added {file_name} to vectorstore.")

            # 处理完一个 CSV 文件后，保存向量数据库
            vectorstore.save_local(vectorstore_path)
            print(f"Vectorstore updated and saved after processing {file_name}.")

    print("Knowledge base built and saved successfully.")
    return vectorstore
