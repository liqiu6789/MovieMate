from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
def query_knowledge_base(vectorstore, query):
    """
    从知识库中检索相关信息。
    """
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    result = qa.run(query)
    return result