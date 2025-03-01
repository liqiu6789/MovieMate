from fastapi import FastAPI
from pydantic import BaseModel
from rag import query_knowledge_base
from database import build_knowledge_base

app = FastAPI()

# 构建知识库，支持缓存分块文件和增量构建向量数据库
input_file = r"C:\Users\Administrator\Downloads\MovieMate-main\MovieMate-main\data\test1.csv"
cache_dir = r"C:\Users\Administrator\Downloads\MovieMate-main\MovieMate-main\cache"
vectorstore_path = r"C:\Users\Administrator\Downloads\MovieMate-main\MovieMate-main"
vectorstore = build_knowledge_base(input_file, cache_dir, vectorstore_path, rebuild_cache=False)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_movies(request: QueryRequest):
    response = query_knowledge_base(vectorstore, request.query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)