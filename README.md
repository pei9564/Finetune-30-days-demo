# Finetune-30-days — LoRA 訓練與資料管理

此專案提供一個最小化的 **LoRA 微調範例**，支援 M3 晶片 (MPS)、NVIDIA GPU (CUDA) 與 CPU。
設計目標：快速建立環境、驗證流程、保存結果，並具備 **資料版本管理與驗證機制**。

---

## 📂 專案結構

```
├── app/
│   ├── config.py                  # 配置定義與管理
│   ├── data_management/          # 資料管理模組
│   │   ├── data_validator.py     # 資料驗證與清理
│   │   ├── dataset_analyzer.py   # 標籤分布分析
│   │   └── version_manager.py    # 資料版本控制
│   ├── logger_config.py          # 日誌系統
│   └── train_lora_v2.py         # LoRA 訓練主程式
├── config/
│   └── default.yaml              # 預設配置文件
├── results/
│   ├── configs/                  # 實驗配置記錄
│   └── final_model/             # 訓練完成的模型
├── logs/                         # 訓練與驗證日誌
├── requirements.txt              # 依賴管理
├── Makefile                      # 簡化指令
└── README.md
```

---

## 🚀 快速開始

### 基本使用

```bash
make setup-conda   # 建立 Conda 環境（自動偵測 GPU/MPS/CPU）
make run-local     # 使用預設配置開始訓練
make logs-local    # 查看訓練進度
```

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

- **配置記錄**：`results/configs/{實驗名稱}_{準確率}_{時間戳}.yaml`
- **訓練日誌**：`logs/training_progress.log`
- **模型保存**：`results/final_model/`

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

- 首次使用請執行 `make setup-conda` 設置環境
- 使用 `make help` 查看完整的命令說明
- 實驗配置會自動保存，方便追蹤和重現
- 資料管理功能在訓練時自動執行，確保資料品質