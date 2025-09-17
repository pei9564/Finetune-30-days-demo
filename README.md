# Finetune-30-days — LoRA 訓練與資料管理

此專案提供一個最小化的 **LoRA 微調範例**，支援 M3 晶片 (MPS)、NVIDIA GPU (CUDA) 與 CPU。
設計目標：快速建立環境、驗證流程、保存結果，並具備 **資料版本管理與驗證機制**。

---

## 📂 專案結構

```
├── app/
│   ├── data_management/             # 資料管理模組
│   │   ├── data_validator.py        # 資料驗證與清理 (空值 / 重複 / HTML)
│   │   ├── dataset_analyzer.py      # 標籤分布與不平衡分析
│   │   └── version_manager.py       # 資料版本控制與 metadata
│   ├── logger_config.py             # 日誌系統
│   └── train_lora.py                # LoRA 訓練主程式
├── results/                         # 訓練輸出 (模型 / 評估報告)
├── logs/                            # 訓練與驗證日誌
├── requirements.txt                 # 依賴管理
├── Makefile                         # 簡化指令 (訓練 / 資料檢查)
└── README.md
```

---

## 🚀 快速開始

### 一鍵執行

```bash
make setup-conda   # 建立 Conda 環境（自動偵測 GPU/MPS/CPU）
make run-local     # 啟動訓練
```

### 分步驟

```bash
brew install --cask miniforge
make setup-conda
make run-local
```

---

## 📊 資料管理

專案內建 **版本管理 / 分布檢查 / 品質驗證**，並在 LoRA 訓練流程中自動執行。
以下指令僅作為 **單獨測試範例**：

```bash
make data-versions   # 建立資料版本
make data-analyze    # 標籤分布分析
make data-validate   # 檢查空值 / 重複 / 長度 / HTML
```

**輸出範例**

```json
{
  "total_samples": 500,
  "label_counts": {"0": 245, "1": 255},
  "imbalance_ratio": 1.04,
  "is_balanced": true
}
```

實際使用時，這些步驟會在 `make run-local` 執行訓練時 **自動套用到資料前處理**。
因此每一次訓練都能保證：

* 有資料版本記錄
* 有標籤分布報告
* 已完成基礎資料清理

---

## ⚙️ 可調參數

在 `app/train_lora.py` 可修改：

```python
num_train_epochs = 1
learning_rate = 5e-4
per_device_train_batch_size = 2
```

LoRA 相關設定：

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)
```

---

## ✅ 使用流程

1. `make setup-conda` — 建立環境
2. `make data-validate` — 驗證資料品質
3. `make run-local` — 啟動訓練
4. 查看 `results/` — 分析模型輸出

---

這是一個 **Day4 的專案基礎版**，目的在於：

* 確認 LoRA 訓練可在不同硬體正常運行
* 具備資料版本化與驗證，確保實驗可追溯
* 為後續任務排程、參數管理、模型評估做好基礎
