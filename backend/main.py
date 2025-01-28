from fastapi import FastAPI, Query
from pydantic import BaseModel
from rag import query_knowledge_base
from database import build_knowledge_base

app = FastAPI()

# 构建知识库
vectorstore = build_knowledge_base(r"C:\Users\Administrator\PycharmProjects\MovieMate\data\test1.csv")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_movies(request: QueryRequest):
    #print(11111111111111111111111111111)
    response = query_knowledge_base(vectorstore, request.query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)