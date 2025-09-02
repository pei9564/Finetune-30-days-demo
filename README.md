# Finetune-30-days — LoRA 訓練與實驗管理

此專案提供一個完整的 **LoRA 微調系統**，支援 M3 晶片 (MPS)、NVIDIA GPU (CUDA) 與 CPU。

**主要特點**：
- 🚀 支援同步（本地）與非同步（分散式）訓練
- 📊 完整的資料版本管理與驗證機制
- 🎯 實驗結果自動保存與追蹤
- 🌐 網頁界面支援任務提交與進度監控
- 🔄 基於 Celery + Redis 的非同步任務系統
- 📝 結構化的配置管理（Pydantic + YAML）

---

## 📂 專案結構

```
├── app/
│   ├── api.py                    # FastAPI 應用
│   ├── config.py                 # 配置定義與管理
│   ├── data_management/         # 資料管理模組
│   │   ├── data_validator.py    # 資料驗證與清理
│   │   ├── dataset_analyzer.py  # 標籤分布分析
│   │   └── version_manager.py   # 資料版本控制
│   ├── logger_config.py         # 日誌系統
│   ├── tasks/                   # Celery 任務
│   │   ├── __init__.py         # Celery 應用配置
│   │   └── training.py         # 訓練任務定義
│   └── train_lora_v2.py        # LoRA 訓練主程式
├── config/
│   └── default.yaml              # 預設配置文件
├── results/                       # 實驗結果目錄
│   └── {實驗名稱}_{時間戳}/      # 獨立實驗目錄
│       ├── logs.txt            # 系統日誌與訓練進度
│       ├── config.yaml         # 實驗配置
│       ├── metrics.json        # 評估指標
│       └── artifacts/          # 模型與其他產出
│           └── final_model/    # 訓練完成的模型
├── requirements.txt              # 依賴管理
├── Makefile                      # 簡化指令
└── README.md
```

---

## 🚀 快速開始

### 基本使用

1. **本地直接訓練**：
```bash
make setup-conda   # 建立 Conda 環境（自動偵測 GPU/MPS/CPU）
make run-local     # 使用預設配置開始訓練
make logs-local    # 查看最新實驗的訓練進度
```

2. **非同步訓練服務**：
```bash
# 1. 啟動服務（需要開啟四個終端）
make start-services  # 啟動 Redis（任務佇列與結果存儲）
make start-worker   # 啟動 Celery worker
make start-api      # 啟動 FastAPI 服務
make start-ui       # 啟動網頁界面

# 2. 使用網頁界面（推薦）
# 訪問 http://localhost:8501
# - 提交任務：選擇「提交任務」頁籤，設置參數
# - 追蹤進度：選擇「追蹤進度」頁籤，輸入 task_id

# 3. 或使用 API（進階）
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "test_async", "epochs": 1}'

curl http://localhost:8000/task/{task_id}  # 查詢任務狀態
```

> 💡 **服務說明**：
> - **Redis (localhost:6379)**：
>   - DB 0：任務佇列（broker）
>   - DB 1：任務結果（backend）
>   - 預設為記憶體儲存，重啟後清空
> - **FastAPI (localhost:8000)**：REST API 服務
> - **Streamlit UI (localhost:8501)**：網頁操作界面
>   - 支援所有 default.yaml 中的參數配置
>   - 自動生成臨時配置文件（不納入版控）
>   - 即時顯示訓練進度（每 2 秒更新）

### 自定義訓練

1. **修改預設配置**：
   直接編輯 `config/default.yaml`

2. **使用命令列參數**：
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

## 📊 實驗記錄

每次訓練會自動創建實驗專屬目錄：`results/{實驗名稱}_{時間戳}/`

```
results/
└── experiment_name_20240101_120000/
    ├── logs.txt           # 系統日誌與訓練進度
    ├── config.yaml        # 本次實驗的完整配置
    ├── metrics.json       # 訓練結果與評估指標
    └── artifacts/         # 模型與其他產出
        └── final_model/   # 訓練完成的模型
```

- **系統日誌**：記錄設備、模型載入、資料處理等系統操作
- **訓練進度**：記錄每個步驟的損失值、學習率、評估指標等
- **實驗配置**：包含所有參數設定，確保實驗可重現
- **評估指標**：保存最終的訓練時間、準確率等結果

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
- 非同步訓練需要安裝 Docker（用於運行 Redis）

### 訓練與配置
- 實驗配置會自動保存，方便追蹤和重現
- 資料管理功能在訓練時自動執行，確保資料品質
- 訓練結果與同步模式相同，都保存在 `results/` 目錄

### 非同步服務
- Redis 資料僅用於任務佇列和狀態追蹤，不保存訓練結果
- 網頁界面生成的臨時配置文件（`config/temp_*.yaml`）不會納入版控
- 建議使用網頁界面操作，API 接口主要用於程式整合
- 所有服務都需要在 Conda 環境中運行（`lora-m3` 或 `lora-cuda`）