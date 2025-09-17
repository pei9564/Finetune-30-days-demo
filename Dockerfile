# 構建階段：用於安裝依賴和編譯
FROM python:3.11-slim as builder

# 設置工作目錄
WORKDIR /app

# 安裝構建依賴
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 --retries 3 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 運行階段：最小化運行環境
FROM python:3.11-slim as runtime

# 設置工作目錄
WORKDIR /app

# 設置時區
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安裝運行時依賴
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 從構建階段複製 Python 環境
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# 複製整個專案目錄
COPY . .

# 創建必要的目錄
RUN mkdir -p results/config

# 設置環境變數
ENV PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1

# 設置默認命令（將由 k8s 或 docker-compose 覆蓋）
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]