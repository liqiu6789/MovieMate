from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def create_embeddings(chunks):
    """
    将文本块转换为向量表示。
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore