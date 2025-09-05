# 使用 Python 3.11 作為基礎映像
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 設置時區
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴（增加超時時間和重試次數）
RUN pip install --no-cache-dir --timeout 100 --retries 3 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 複製應用程式碼
COPY app/ ./
COPY config/ ./config/ma

# 設置環境變數
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 設置默認命令（將由 docker-compose 覆蓋）
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
