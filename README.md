# Finetune-30-days — LoRA 訓練與實驗管理

此專案提供一個完整的 **LoRA 微調系統**，支援 M3 晶片 (MPS)、NVIDIA GPU (CUDA) 與 CPU。

**主要特點**：
- 🚀 支援同步（本地）與非同步（分散式）訓練
- 📊 完整的資料版本管理與驗證機制
- 🎯 實驗結果自動保存與追蹤
- 🌐 網頁界面支援任務提交與進度監控
- 🔄 基於 Celery + Redis 的非同步任務系統
- 📝 結構化的配置管理（Pydantic + YAML）
- ☸️ 完整的 Kubernetes 部署支援
- 🐳 優化的多階段 Docker 構建
- 📈 效能監控與分析工具
- 🔐 JWT 認證與 RBAC 權限控制
- 📋 完整的審計日誌與追蹤系統
- 🧪 全面的單元測試覆蓋

---

## 🔄 系統互動流程

```mermaid
sequenceDiagram
    participant U as 使用者
    participant UI as Streamlit UI
    participant API as FastAPI
    participant C as Celery Worker
    participant T as 訓練程式
    participant DB as SQLite DB
    participant R as Redis

    U->>UI: 填寫實驗參數並提交
    UI->>API: POST /train (config)
    API->>C: 提交 Celery 任務
    C->>R: 任務入隊 (task_id)
    API-->>UI: 回傳 task_id

    loop 任務輪詢
        UI->>API: GET /task/{task_id}
        API->>R: 查詢狀態
        R-->>API: 返回 PENDING/STARTED
        API-->>UI: 更新 UI Stepper 狀態
    end

    C->>T: 執行 train_lora_v2.py
    T->>DB: 寫入實驗記錄
    T->>R: 更新任務結果 (SUCCESS)
    UI->>API: 最後查詢 /task/{task_id}
    API-->>UI: 返回 SUCCESS + 實驗結果
```

## 🔄 系統架構

```mermaid
graph TB
    subgraph Training["訓練流程"]
        Train[train_lora_v2.py] --> |保存實驗結果| Results[(results/)]
        Results --> |包含| Config[config.yaml]
        Results --> |包含| Model[final_model/]
    end

    subgraph Inference["推論服務"]
        API[inference_api.py] --> |讀取配置| Config
        API --> |載入權重| Model
        API --> |回應請求| Response[JSON Response]
    end

    subgraph Client["客戶端"]
        CLI[make serve] --> |啟動| API
        Test[make predict-*] --> |測試| API
    end

    Results --> |實驗記錄| DB[(experiments.db)]
```

## 🔐 認證與授權機制

本系統採用 JWT (JSON Web Token) 進行認證和授權管理，實現了完整的 RBAC（基於角色的訪問控制）。

### 認證流程

```mermaid
sequenceDiagram
    participant C as 客戶端
    participant A as API
    participant J as JWT Utils
    
    C->>A: POST /login (username, password)
    A->>A: 驗證用戶憑證
    A->>J: create_token(user_id, role)
    J-->>A: JWT Token
    A-->>C: {token, user_id, role}
    
    Note over C,A: 後續請求
    C->>A: API 請求 + Bearer Token
    A->>J: get_current_user(token)
    J-->>A: user_info 或 401/403 錯誤
```

### 權限控制

系統實現了三層權限控制：

1. **基本認證** (`get_current_user`)：
   - 驗證 Bearer Token 的存在和有效性
   - 解析用戶身份和角色信息
   - 處理 token 過期和無效情況

2. **管理員權限** (`check_admin`)：
   - 限制只有管理員可以訪問的端點
   - 用於敏感操作（如查看所有實驗記錄）
   - 返回 403 錯誤給非管理員用戶

3. **資源所有權** (`check_task_owner`)：
   - 確保用戶只能訪問自己的資源
   - 管理員可以訪問所有資源
   - 基於資源 ID 前綴驗證所有權

### API 端點權限

| 端點 | 方法 | 權限要求 | 說明 |
|------|------|----------|------|
| `/login` | POST | 無 | 用戶登入，返回 JWT token |
| `/train` | POST | 已認證用戶 | 提交訓練任務 |
| `/task/{task_id}` | GET | 任務所有者 | 查詢任務狀態 |
| `/experiments` | GET | 管理員 | 列出所有實驗記錄 |
| `/experiments/stats` | GET | 管理員 | 獲取實驗統計信息 |
| `/experiments/{id}` | GET | 任務所有者 | 查詢單個實驗記錄 |
| `/audit/logs` | GET | 管理員 | 查詢審計日誌 |

### JWT 配置

```python
# JWT 相關配置
JWT_SECRET = "your-secret-key"  # 生產環境應使用環境變數
JWT_ALGORITHM = "HS256"         # 加密算法
TOKEN_EXPIRE_MINUTES = 30       # token 有效期
```

### 安全性考慮

1. **Token 管理**：
   - 自動過期機制（30分鐘）
   - 無狀態設計，不需要服務器存儲
   - 支援 token 刷新（待實現）

2. **錯誤處理**：
   - 401：未認證或 token 過期
   - 403：權限不足
   - 詳細的錯誤訊息

3. **最佳實踐**：
   - 使用環境變數管理敏感信息
   - HTTPS 傳輸（生產環境）
   - 請求頻率限制（待實現）

---

## 📂 專案結構

```
├── app/
│   ├── main.py                    # FastAPI 主應用程式
│   ├── config.py                  # 配置定義與驗證
│   ├── db.py                      # 資料庫模型與操作
│   ├── inference_api.py           # 情感分類推論服務
│   ├── logger_config.py           # 日誌配置與管理
│   ├── settings.py                # 系統設定與常數
│   ├── stepper_ui.py             # Streamlit UI 介面
│   ├── train_lora_v2.py          # LoRA 訓練主程式
│   ├── auth/                     # 認證與授權模組
│   │   ├── jwt_utils.py          # JWT Token 處理
│   │   └── audit_log.py          # 審計日誌功能
│   ├── api/                      # API 路由模組
│   │   └── routes/               # 路由定義
│   │       └── audit.py         # 審計日誌 API 路由
│   ├── data_management/          # 資料管理模組
│   │   ├── data_validator.py     # 資料集驗證與品質檢查
│   │   ├── dataset_analyzer.py   # 資料集分布與統計分析
│   │   └── version_manager.py    # 資料集版本與變更追蹤
│   ├── monitoring/              # 監控模組
│   │   └── performance.py       # 系統效能與資源監控
│   ├── tasks/                   # 任務處理模組
│   │   └── training.py          # Celery 訓練任務
│   └── tools/                   # 工具模組
│       ├── analyze_metrics.py   # 實驗指標分析工具
│       └── checkpoint_manager.py # 檢查點管理
├── config/                      # 配置文件目錄
│   ├── default.yaml            # 預設訓練配置
│   ├── albert.yaml             # Albert 模型配置
│   ├── bert_chinese.yaml       # 中文 BERT 配置
│   ├── distilbert_optimized.yaml # 優化的 DistilBERT 配置
│   └── roberta.yaml            # RoBERTa 模型配置
├── results/                     # 實驗結果目錄
│   ├── experiments.db          # SQLite 實驗記錄資料庫
│   └── {實驗名稱}/            # 獨立實驗目錄
│       ├── config.yaml        # 實驗完整配置
│       ├── metrics.json       # 訓練評估指標
│       ├── logs.txt          # 訓練日誌
│       └── artifacts/        # 模型與檢查點
│           ├── checkpoint-*/ # 訓練中間檢查點
│           └── final_model/  # 最終訓練模型
├── data/                       # 資料集目錄
│   ├── datasets/              # 訓練資料集
│   └── metadata/              # 資料集元資料
├── k8s/                       # Kubernetes 部署配置
│   ├── manifests/             # K8s 資源清單
│   └── k8s.sh                 # K8s 部署腳本
├── requirements.txt           # Python 依賴清單
├── docker-compose.yml         # Docker Compose 配置
├── Dockerfile                 # Docker 映像構建配置
├── Makefile                   # 專案管理指令
└── README.md                  # 專案說明文件
```

---

## 🚀 快速開始

### 環境設置

1. **複製環境變數文件**：
   ```bash
   cp .env.example .env
   ```

2. **編輯環境變數**（可選）：
   ```bash
   # 編輯 .env 文件來自定義配置
   nano .env
   ```

   主要配置項目：
   - `REDIS_PORT`: Redis 端口（預設：6379）
   - `API_PORT`: API 服務端口（預設：8000）
   - `UI_PORT`: UI 界面端口（預設：8501）
   - `TZ`: 時區設定（預設：Asia/Taipei）

### 部署方式

1. **本地直接訓練**：
```bash
make setup-conda   # 建立 Conda 環境（自動偵測 GPU/MPS/CPU）
make run-local     # 使用預設配置開始訓練
make logs-local    # 查看訓練進度
```

2. **Docker 容器部署**：
```bash
# 啟動所有服務
make start-services  # 啟動 Redis、Worker、API、UI 服務

# 使用網頁界面（推薦）
# 訪問 http://localhost:8501
```

3. **Kubernetes 部署**（新增）：
```bash
# 快速部署（開發環境）
make k8s-quick-deploy  # 一鍵部署（建構+部署）

# 開啟服務訪問
make k8s-port-forward  # 轉發服務端口到本地

# 監控與管理
make k8s-dashboard    # 開啟 K8s 儀表板
make k8s-status       # 查看部署狀態
make k8s-logs service=worker  # 查看特定服務日誌

# 擴展服務
make k8s-scale service=worker replicas=3  # 調整 worker 數量
```

> 💡 **服務說明**：
> - **Redis (localhost:6379)**：任務佇列與結果存儲
> - **FastAPI (localhost:8000)**：REST API 服務
> - **Streamlit UI (localhost:8501)**：網頁操作界面
>   - 支援所有 default.yaml 中的參數配置
>   - 直接傳遞訓練配置（不再使用臨時文件）
>   - 即時顯示訓練進度（每 2 秒更新）
>   - 實驗記錄查看與管理

### 測試開發

本專案使用 pytest 進行單元測試，主要測試以下功能：

1. **API 端點測試** (`app/tests/test_api.py`)：
   ```bash
   # 運行所有測試
   make test
   
   # 詳細模式（顯示測試過程）
   make test-v
   ```

   測試覆蓋：
   - 基本功能：訓練任務提交、狀態查詢
   - 錯誤處理：無效配置、無效任務 ID
   - 異常處理：記憶體不足、數據集錯誤

2. **訓練邏輯測試** (`app/tests/test_training.py`)：
   - 數據集處理：空數據集、缺失分割
   - 系統資源：記憶體監控、OOM 處理
   - 序列處理：超長文本截斷

3. **認證與授權測試** (`app/tests/test_auth.py`)：
   - JWT Token 創建與驗證
   - 權限檢查：管理員 vs 普通用戶
   - Token 過期處理
   - 錯誤處理：無效 token、權限不足

4. **審計日誌測試** (`app/tests/test_audit.py`)：
   - 審計日誌保存功能
   - 日誌查詢與篩選
   - 中間件功能測試
   - API 端點權限驗證

> 💡 **測試說明**：
> - 使用 mock 隔離外部依賴（Redis、Celery、資料庫）
> - 自動檢測系統環境（MPS/CUDA/CPU）
> - 包含完整的錯誤處理測試用例
> - 審計日誌功能在測試中被 mock，避免產生實際記錄

### 自定義訓練

1. **使用效能監控**：
   在訓練過程中自動啟用 `PerformanceMonitor` 以收集系統和訓練指標。

2. **分析實驗數據**：
   使用 `analyze_metrics.py` 來生成實驗比較報告。
   ```bash
   make analyze-by-model
   ```

3. **修改預設配置**：
   直接編輯 `config/default.yaml`

4. **使用命令列參數**：
   ```bash
   python app/train_lora_v2.py \
     --experiment_name "custom_test" \
     --learning_rate 0.001 \
     --epochs 3 \
     --train_samples 1000
   ```

### 常用參數

```yaml
# 在 config/default.yaml 中可調整：

model:
  name: "distilbert-base-uncased"
  num_labels: 2

training:
  learning_rate: 5.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 2

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_lin", "v_lin"]
  lora_dropout: 0.1
```

---

## 📊 實驗管理

### 實驗目錄結構

每次訓練會自動創建實驗專屬目錄：
```
results/
└── {實驗名稱}/             # 實驗目錄
    ├── config.yaml         # 實驗配置（含基礎模型設定）
    ├── metrics.json        # 評估指標
    └── artifacts/
        └── final_model/    # 訓練完成的模型（LoRA 權重）
```

- **系統日誌**：記錄設備、模型載入、資料處理等系統操作
- **訓練進度**：記錄每個步驟的損失值、學習率、評估指標等
- **實驗配置**：包含所有參數設定，確保實驗可重現
- **評估指標**：保存最終的訓練時間、準確率等結果

### Checkpoint 保留策略

為了有效管理訓練資源，系統會自動執行以下清理規則：

- 每個實驗目錄下只保留三個關鍵 checkpoints：
  1. 最佳評估準確率的 checkpoint（用於最終模型）
  2. 最後一個 checkpoint（用於恢復訓練）
  3. 訓練時間最短的 checkpoint（用於快速實驗）
- 清理時機：每次保存新的 checkpoint 後自動執行
- 清理方式：分析 `trainer_state.json` 中的評估指標，保留符合條件的檔案

### 實驗追蹤

提供多種方式查看實驗記錄：

1. **網頁界面**（推薦）：
   - 訪問 http://localhost:8501
   - 切換到「實驗記錄」頁籤
   - 支援篩選、排序、統計功能
   - 即時更新實驗狀態

2. **命令列工具**：
   ```bash
   # 查看實驗記錄（表格形式）
   make db-list
   
   # 查看最新實驗的訓練進度
   make logs-local
   ```

3. **REST API**：
   ```bash
   # 列出所有實驗（支援篩選和排序）
   curl "http://localhost:8000/experiments?min_accuracy=0.8&sort_by=eval_accuracy&desc=true"

   # 查詢單一實驗
   curl http://localhost:8000/experiments/{experiment_id}

   # 獲取實驗統計
   curl http://localhost:8000/experiments/stats

   # 查詢審計日誌（需要管理員權限）
   curl -H "Authorization: Bearer {admin_token}" \
        "http://localhost:8000/audit/logs?user_id=test_user&limit=10"
   ```

   支援的篩選條件：
   - **實驗記錄**：
     - `name`：實驗名稱（模糊匹配）
     - `min_accuracy`：最低準確率
     - `max_runtime`：最長訓練時間
     - `start_date`/`end_date`：時間範圍
     - `sort_by`：排序欄位（created_at/name/train_runtime/eval_accuracy）
     - `desc`：是否降序排序
     - `limit`：返回數量限制
   - **審計日誌**：
     - `user_id`：特定用戶的日誌
     - `role`：特定角色的日誌
     - `start_time`/`end_time`：時間範圍（Unix timestamp）
     - `limit`：返回數量限制（1-1000）

### 效能監控與分析
本專案提供了強大的效能監控與分析工具，幫助用戶追蹤訓練過程中的系統資源使用情況和模型性能。
`performance.py` 提供了 `PerformanceMonitor` 類，用於收集 CPU、記憶體使用率以及訓練過程中的 token 處理速度。
`analyze_metrics.py` 提供了分析工具，能夠從多個實驗中提取數據並生成比較報告。

---

## 🔧 資料管理工具

以下指令使用預設的 SST-2 範例資料集，僅供開發測試用途。
實際訓練時，這些功能已整合在訓練流程中自動執行。

```bash
make data-analyze    # 分析標籤分布
make data-validate   # 驗證資料品質
make data-versions   # 管理資料版本
```

**資料驗證報告範例**：
```json
{
  "total_samples": 500,
  "label_counts": {"0": 245, "1": 255},
  "imbalance_ratio": 1.04,
  "is_balanced": true
}
```

---

## 💡 注意事項

### 環境設置
- 首次使用請執行 `make setup-conda` 設置環境
- 使用 `make help` 查看完整的命令說明
- 支援 Docker 和 Kubernetes 部署
- 環境變數配置：複製 `.env.example` 到 `.env` 並根據需要調整

### Docker 與 Kubernetes
- 使用多階段構建優化映像大小和構建速度
- 支援快速構建（開發用）和完整構建（生產用）
- Kubernetes 部署提供完整的服務管理功能
- 使用 `make k8s-port-forward` 簡化服務訪問

### 訓練與配置
- 實驗配置會自動保存，方便追蹤和重現
- 資料管理功能在訓練時自動執行，確保資料品質
- 訓練結果統一保存在 `results/` 目錄
- 支援直接傳遞訓練配置，無需臨時文件

### 非同步服務
- 支援 Docker Compose 和 Kubernetes 兩種部署方式
- 提供完整的服務監控和管理功能
- 建議使用網頁界面操作，API 接口主要用於程式整合
- 服務端口可通過 `.env` 文件自定義調整


### 推論服務

本專案提供輕量級的情感分類推論服務，支援多種預訓練模型：

- distilbert-base-uncased（英文，輕量級）
- roberta-base（英文，高性能）
- albert-base-v2（英文，輕量級）
- bert-base-chinese（中文）

1. **啟動服務**：
```bash
# 使用最新實驗的模型
make serve

# 或指定特定實驗
make serve exp=default_experiment_20250911_233842
```

2. **測試預測**：
```bash
# 檢查服務狀態
make predict-health

# 測試自訂文本
make predict-text text='This movie was great!'

# 使用預設範例
make predict-positive  # 測試正面評論
make predict-negative  # 測試負面評論
```

### 回應格式

```json
{
    "label": 1,                    # 1 = positive, 0 = negative
    "probability": 0.9983,         # prediction confidence
    "latency_ms": 25.4,           # processing time (ms)
    "base_model": "distilbert-base-uncased",  # base model name
    "language": "English"          # model language
}
```

### 注意事項

1. **模型選擇**：
   - 英文文本請使用英文模型（distilbert/roberta/albert）
   - 中文文本請使用中文模型（bert-base-chinese）
   - 注意文本語言要和模型語言匹配

2. **服務配置**：
   - 服務運行在 8002 端口
   - 自動檢測並使用 MPS（Apple Silicon）或 CPU
   - 支援模型熱切換（不需重啟服務）
