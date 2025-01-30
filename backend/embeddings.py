from langchain_community.embeddings import HuggingFaceEmbeddings  # 使用 Hugging Face 嵌入模型
from langchain_community.vectorstores.faiss import FAISS

def create_embeddings(chunks):
    """
    将文本块转换为向量表示。
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 选择一个轻量级的 Sentence Transformer 模型
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore