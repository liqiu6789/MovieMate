# 电影推荐与问答系统

## 项目概述

这是一个基于 RAG（Retrieval-Augmented Generation）技术的电影推荐与问答系统。用户可以通过自然语言查询电影信息、获取推荐或解答相关问题。系统结合了 FastAPI 后端、Streamlit 前端、LangChain 和 OpenAI 的 GPT 模型，实现了高效的自然语言处理和电影推荐功能。

---

## 功能列表

1. **电影信息查询**：
   - 支持通过自然语言查询电影信息，例如：
     - “《盗梦空间》的导演是谁？”
     - “《泰坦尼克号》的上映时间是什么时候？”

2. **电影推荐**：
   - 根据用户输入的关键词或电影特征，推荐相关电影，例如：
     - “推荐几部类似《星际穿越》的电影。”
     - “有哪些高分科幻电影？”

3. **电影问答**：
   - 回答与电影相关的具体问题，例如：
     - “《阿甘正传》获得了哪些奖项？”
     - “《肖申克的救赎》的评分是多少？”

---

## 技术栈

### 后端
- **框架**：FastAPI
- **语言模型**：OpenAI GPT
- **向量存储**：FAISS
- **文本嵌入**：OpenAI Embeddings
- **数据处理**：Pandas
- **数据库**：SQLite

### 前端
- **框架**：Streamlit
- **UI 组件**：Streamlit 原生组件

### 工具
- **RAG 系统**：LangChain
- **环境管理**：Python + `dotenv`
- **容器化**：Docker

---

## 项目结构
```
movie-recommendation-system/
│
├── backend/
│ ├── main.py # FastAPI 后端服务
│ ├── database.py # 数据库操作
│ ├── embeddings.py # 文本嵌入与向量存储
│ ├── rag.py # RAG 系统实现
│ └── requirements.txt # 后端依赖
│
├── frontend/
│ ├── app.py # Streamlit 前端界面
│ └── requirements.txt # 前端依赖
│
├── data/
│ └── movies.csv # 电影数据文件
│
├── .env # 环境变量文件（存储 OpenAI API 密钥等）
├── Dockerfile # Docker 容器化配置
└── README.md # 项目说明文档
```


---

## 安装与运行

### 1. 克隆项目
```bash
git clone https://github.com/liqiu6789/MovieMate.git
cd MovieMate
```

### 2. 设置环境变量
在项目根目录下创建 .env 文件，并添加 deepseek API 密钥：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 3. 安装依赖
后端依赖
```bash
cd backend
pip install -r requirements.txt
```

前端依赖
```bash
cd ../frontend
pip install -r requirements.txt
```

### 4. 准备数据
将电影数据文件（如 movies.csv）放置在 data/ 目录下。

### 5. 运行后端服务
```bash
cd ../backend
uvicorn main:app --reload
```
- 后端服务默认运行在 http://localhost:8000。
- 访问 http://localhost:8000/docs 查看 API 文档。

### 6. 运行前端界面
```bash
cd ../frontend
streamlit run app.py
```
- 前端界面默认运行在 http://localhost:8501。
- 在界面中输入问题，系统会返回电影信息或推荐列表。

## Docker 部署
### 1. 构建 Docker 镜像
```bash
docker build -t movie-recommendation-system .
```
### 2. 运行 Docker 容器
```bash
docker run -d -p 8000:8000 movie-recommendation-system
```
## API 接口说明
### 电影查询与推荐
- URL: /query

- Method: POST

- Request Body:

```bash
{
  "query": "推荐几部类似《星际穿越》的电影。"
}
Response:


{
  "response": "以下是推荐的电影：1. 《盗梦空间》 2. 《火星救援》 3. 《地心引力》"
}
```
## 示例问题
### 1.电影信息查询：

- “《盗梦空间》的导演是谁？”

- “《泰坦尼克号》的上映时间是什么时候？”

### 2.电影推荐：

- “推荐几部类似《星际穿越》的电影。”

- “有哪些高分科幻电影？”

### 3.电影问答：

- “《阿甘正传》获得了哪些奖项？”

- “《肖申克的救赎》的评分是多少？”


## 联系信息
- 邮箱: liqiu6789@qq.com

- GitHub: liqiu123456123

