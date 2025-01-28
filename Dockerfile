# 使用 Python 3.9 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制后端代码
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# 复制前端代码
COPY frontend/requirements.txt .
RUN pip install -r requirements.txt

# 复制项目代码
COPY . .

# 启动后端服务
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]