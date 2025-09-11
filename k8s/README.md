# 🚀 LoRA Finetune Platform - Kubernetes 部署

本目錄包含將 LoRA 微調平台部署到 Kubernetes (minikube) 的完整配置。

## 📁 目錄結構

```
k8s/
├── manifests/           # Kubernetes 配置檔案
│   ├── namespace.yaml   # 命名空間定義
│   ├── configmap.yaml   # 環境變數配置
│   ├── pvc.yaml        # 持久化儲存
│   ├── redis.yaml      # Redis 服務
│   ├── worker.yaml     # Celery Worker
│   ├── api.yaml        # FastAPI 服務
│   └── ui.yaml         # Streamlit UI
├── Makefile            # 所有操作指令（無需額外腳本）
└── README.md           # 本說明文件
```

## 🚀 快速開始

### 方法 1: 一鍵部署（推薦）

```bash
cd k8s

# 1. 設定環境
make setup

# 2. 建構並部署
make quick-deploy

# 3. 驗證部署
make verify
```

### 方法 2: 分步執行

```bash
cd k8s

# 1. 設定 minikube
make setup

# 2. 建構映像
make build

# 3. 部署應用程式
make deploy

# 4. 驗證部署
make verify
```

## 🌐 訪問應用程式

部署完成後，您可以通過以下 URL 訪問：

- **API**: http://localhost:8000 (或 http://$(minikube ip):30080)
- **UI**: http://localhost:8501 (或 http://$(minikube ip):30501)

### 使用 port-forward 訪問（推薦）

```bash
# API
kubectl port-forward service/api 8000:8000 -n lora-system

# UI
kubectl port-forward service/ui 8501:8501 -n lora-system
```

## 🔧 常用指令

```bash
# 查看狀態
make status

# 查看日誌
make logs

# 擴展 Worker
make scale REPLICAS=3

# 重新啟動服務
make restart

# 清理部署
make cleanup
```

## 📊 服務說明

### Redis
- **映像**: redis:latest
- **端口**: 6379
- **健康檢查**: `redis-cli ping`
- **儲存**: 使用 PVC 持久化資料

### Celery Worker
- **映像**: finetune-app:latest
- **命令**: `python -m celery -A tasks worker --loglevel=info -P solo`
- **副本數**: 2 (可調整)
- **儲存**: 共用 results PVC

### FastAPI
- **映像**: finetune-app:latest
- **命令**: `python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
- **端口**: 8000 (NodePort: 30080)
- **健康檢查**: `GET /docs`
- **儲存**: 共用 results PVC

### Streamlit UI
- **映像**: finetune-app:latest
- **命令**: `python -m streamlit run stepper_ui.py`
- **端口**: 8501 (NodePort: 30501)
- **健康檢查**: `GET /_stcore/health`
- **儲存**: 共用 results PVC

## ⚙️ 環境變數

所有環境變數都通過 ConfigMap 管理：

- `CELERY_BROKER_URL`: redis://redis:6379/0
- `CELERY_RESULT_BACKEND`: redis://redis:6379/1
- `API_URL`: http://api:8000
- `REDIS_URL`: redis://redis:6379
- `TZ`: Asia/Taipei
- `PYTHONPATH`: /app
- `PYTHONUNBUFFERED`: 1

## 💾 儲存

- 使用 PersistentVolumeClaim `results-pvc`
- 儲存大小: 10Gi
- 所有服務共用同一個 PVC
- 資料持久化在 minikube 中

## 🐛 故障排除

### 1. Pod 無法啟動

```bash
# 查看 Pod 詳細資訊
kubectl describe pod <pod-name> -n lora-system

# 查看 Pod 日誌
kubectl logs <pod-name> -n lora-system
```

### 2. 映像建構失敗

```bash
# 檢查映像是否存在
docker images | grep finetune

# 重新建構映像
make build
```

### 3. 服務無法訪問

```bash
# 檢查服務狀態
kubectl get services -n lora-system

# 使用 port-forward
kubectl port-forward service/ui 8501:8501 -n lora-system
```

### 4. 儲存問題

```bash
# 檢查 PVC 狀態
kubectl get pvc -n lora-system

# 檢查 PV 狀態
kubectl get pv
```

## 📈 監控

### 查看資源使用

```bash
# 查看 Pod 資源使用
kubectl top pods -n lora-system

# 查看節點資源使用
kubectl top nodes
```

### 查看事件

```bash
kubectl get events -n lora-system --sort-by=.metadata.creationTimestamp
```

## 🧹 清理

```bash
# 清理 Kubernetes 資源
make cleanup

# 完整清理（包括映像）
make full-cleanup
```

## 🎯 部署成功指標

- ✅ 所有 Pod 狀態為 Running
- ✅ 服務可正常訪問
- ✅ Redis 連接正常
- ✅ 持久化儲存已配置
- ✅ 健康檢查通過

## ✨ 特色

- ✅ 清晰的目錄結構
- ✅ 完整的 Kubernetes 配置
- ✅ 自動化部署腳本
- ✅ 健康檢查和監控
- ✅ 持久化儲存
- ✅ 可擴展的架構
- ✅ 統一的 Makefile 介面
- ✅ 無需額外腳本檔案

## 🚀 下一步

1. **訪問應用程式**:
   - 打開瀏覽器訪問 http://localhost:8000 (API)
   - 打開瀏覽器訪問 http://localhost:8501 (UI)

2. **監控服務**:
   ```bash
   # 查看實時日誌
   make logs
   
   # 查看狀態
   make status
   ```

3. **調整資源**:
   - 編輯 `manifests/` 目錄中的 YAML 檔案來調整 CPU/記憶體限制
   - 使用 `kubectl apply -f manifests/<file>` 應用變更

**🎊 恭喜！您的 LoRA 微調平台已成功部署到 Kubernetes！**