import logging
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from logger_config import setup_logger
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from data_management import (
    DataValidator,
    DataVersionManager,
    analyze_distribution,
    get_data_summary,
)

logger = setup_logger()


# ===== 訓練進度記錄 Callback =====
class TrainingProgressCallback(TrainerCallback):
    """訓練進度記錄 callback，將訓練指標保存到文件"""

    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

        # 創建 logger
        self.logger = logging.getLogger("training_progress")
        self.logger.setLevel(logging.INFO)

        # 清除現有 handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 只輸出到文件，避免與終端顯示衝突
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - PROGRESS - %(message)s")
        )

        self.logger.addHandler(file_handler)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """記錄訓練進度"""
        if logs:
            metrics = []

            # 收集訓練指標
            if "loss" in logs:
                metrics.append(f"loss={logs['loss']:.4f}")
            if "learning_rate" in logs:
                metrics.append(f"lr={logs['learning_rate']:.6f}")
            if "epoch" in logs:
                metrics.append(f"epoch={logs['epoch']:.2f}")
            if "eval_loss" in logs:
                metrics.append(f"eval_loss={logs['eval_loss']:.4f}")
            if "eval_accuracy" in logs:
                metrics.append(f"accuracy={logs['eval_accuracy']:.4f}")

            # 合併指標到一行
            if metrics:
                message = f"Step {state.global_step}: {' | '.join(metrics)}"
                self.logger.info(message)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """記錄評估結果"""
        if metrics:
            eval_metrics = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    eval_metrics.append(f"{key}={value:.4f}")
                else:
                    eval_metrics.append(f"{key}={value}")

            if eval_metrics:
                message = f"Evaluation: {' | '.join(eval_metrics)}"
                self.logger.info(message)


# ===== 1. 自動偵測裝置 =====
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("🚀 使用 CUDA GPU 加速")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("🚀 使用 MPS 加速（Apple Silicon）")
else:
    device = torch.device("cpu")
    logger.info("⚠️ 未檢測到 GPU，使用 CPU 模式")


# ===== 2. 載入模型與 tokenizer =====
logger.info("📥 載入模型與 tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
    device
)
logger.info(f"✅ 模型載入完成: {model_name}")

# ===== 3. 載入小型資料集 =====
logger.info("📊 載入資料集...")
dataset = load_dataset("glue", "sst2")
train_small = dataset["train"].select(range(500))
eval_small = dataset["validation"].select(range(100))
logger.info(f"   - 訓練資料: {len(train_small)} 筆")
logger.info(f"   - 驗證資料: {len(eval_small)} 筆")

# ===== 資料管理與版本控制 =====

logger.info("📋 進行資料分析與管理...")

# 1. 資料摘要
summary = get_data_summary(train_small)
logger.info(f"📊 資料摘要:")
logger.info(f"   - 特徵數: {summary['num_features']}")
logger.info(f"   - 特徵名稱: {summary['feature_names']}")

# 2. 分布分析
distribution_analysis = analyze_distribution(train_small)
logger.info(f"📊 類別分布:")
logger.info(f"   - 類別數: {distribution_analysis['num_classes']}")
logger.info(f"   - 各類別數量: {distribution_analysis['label_counts']}")
logger.info(f"   - 不平衡比例: {distribution_analysis['imbalance_ratio']:.2f}:1")
logger.info(f"   - 是否平衡: {'✅' if distribution_analysis['is_balanced'] else '❌'}")

# 3. 資料驗證
validator = DataValidator(logger)
validator.set_validation_rules(
    {
        "min_text_length": 5,
        "max_text_length": 500,
        "allow_empty": False,
        "remove_html": True,
    }
)

validation_report = validator.validate_dataset(train_small, ["sentence"])
total_issues = sum(
    len(issue_list) for issue_list in validation_report["issues"].values()
)

if total_issues > 0:
    logger.warning(f"⚠️ 發現 {total_issues} 個資料問題")
    # 清理資料
    train_small = validator.clean_dataset(train_small, ["sentence"], validation_report)
    logger.info(f"🧹 資料清理完成，剩餘 {len(train_small)} 筆訓練資料")
else:
    logger.info("✅ 資料驗證通過，無問題發現")

# 4. 版本管理
try:
    version_manager = DataVersionManager(logger=logger)

    # 檢查是否已有當前版本
    current_version = version_manager.get_current_version()
    if current_version:
        logger.info(f"📦 當前資料版本: {current_version}")
    else:
        # 創建第一個版本
        version_name = f"sst2_train_{len(train_small)}samples"
        version_manager.create_version(
            train_small,
            version_name,
            description=f"SST-2 訓練集，經過清理，{len(train_small)} 筆資料",
            cleaning_strategy="移除空值、HTML標籤清理、重複資料移除",
            source_info={
                "dataset": "glue/sst2",
                "split": "train",
                "original_samples": 500,
                "cleaned_samples": len(train_small),
            },
        )
        logger.info(f"📦 創建資料版本: {version_name}")

except Exception as e:
    logger.warning(f"⚠️ 版本管理失敗: {e}")

logger.info("=" * 50)


def tokenize(batch):
    return tokenizer(
        batch["sentence"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = train_small.map(tokenize, batched=True)
eval_dataset = eval_small.map(tokenize, batched=True)
logger.info("✅ 訓練和驗證資料集處理完成")

# ===== 4. LoRA 配置 =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config).to(device)
logger.info("✅ LoRA 配置完成")

# 顯示可訓練參數數量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(
    f"📊 可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)"
)

# ===== 5. 評估方法 =====
logger.info("📈 設置評估方法...")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


# ===== 6. 訓練參數 =====
logger.info("⚙️ 設置訓練參數...")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    report_to=None,
)

logger.info(f"📝 訓練參數:")
logger.info(f"   - 學習率: {training_args.learning_rate}")
logger.info(f"   - 批次大小: {training_args.per_device_train_batch_size}")
logger.info(f"   - 訓練輪數: {training_args.num_train_epochs}")
logger.info(f"   - 記錄頻率: 每 {training_args.logging_steps} 步")

# ===== 7. 建立 Trainer 並訓練 =====
logger.info("🚀 開始訓練...")
logger.info("=" * 50)

# 創建自定義 callback 來記錄訓練進度（只記錄到文件，不在終端顯示）
progress_callback = TrainingProgressCallback("logs/local_training.log")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[progress_callback],  # 添加自定義 callback
)

# 訓練
train_result = trainer.train()

logger.info("🎉 訓練完成！")
logger.info("=" * 50)

# 評估
logger.info("📊 評估模型...")
eval_result = trainer.evaluate()
logger.info(f"✅ 驗證準確率: {eval_result['eval_accuracy']:.4f}")

# 保存模型
logger.info("💾 保存模型...")
trainer.save_model("./results/final_model")
logger.info("✅ 模型已保存到 ./results/final_model")

# 訓練總結
logger.info("🎯 訓練總結:")
logger.info(f"   - 總訓練步數: {train_result.global_step}")
logger.info(f"   - 總訓練時間: {train_result.metrics['train_runtime']:.2f} 秒")
logger.info(f"   - 驗證準確率: {eval_result['eval_accuracy']:.4f}")
logger.info(f"   - 模型保存位置: ./results/final_model")
