# Finetune-30-days — LoRA 訓練環境

此目錄提供一個最小化的 **LoRA 微調範例**，支援 M3 晶片 (MPS)、NVIDIA GPU (CUDA) 以及 CPU。
設計目標：快速建立環境、驗證流程、保存結果。

---

## 📂 專案結構

```
├── app/
│   ├── train_lora.py      # LoRA 訓練腳本
│   └── logger_config.py   # 日誌系統
├── results/               # 訓練輸出
├── logs/                  # 訓練日誌
├── requirements.txt       # 依賴管理
├── Makefile               # 簡化指令
└── README.md
```

---

## 🚀 快速開始

### 一鍵執行（推薦）

```bash
make setup-conda   # 自動建立 Conda 環境（依硬體判斷 GPU/MPS/CPU）
make run-local     # 開始訓練
```

### 分步驟

```bash
brew install --cask miniforge   # 安裝 Conda
make setup-conda                # 建立環境
make run-local                  # 啟動訓練
```

---

## ⚙️ 可調參數

在 `app/train_lora.py` 可修改：

```python
num_train_epochs = 1
learning_rate = 5e-4
per_device_train_batch_size = 2
logging_steps = 10

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)
```

---

## 📊 訓練監控

* 終端：即時進度條與 loss/acc
* 檔案：詳細日誌輸出至 `logs/local_training.log`
* 結果：模型與指標保存在 `results/`

---

## ✅ 使用流程

1. `make setup-conda` — 建立環境
2. `make run-local` — 啟動訓練
3. `make logs-local` — 查看最後 20 行日誌
4. 查看 `results/` — 分析模型輸出

---

## 🎯 定位

這是一個 **專案起點**，目的在於：

* 確認 LoRA 訓練流程可在不同硬體環境正常運行
* 建立後續微調平台開發的基礎
