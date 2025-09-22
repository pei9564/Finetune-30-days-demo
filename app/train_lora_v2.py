"""
LoRA 訓練腳本 v2
使用統一配置系統
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from app.config import load_config
from app.data_management import (
    DataValidator,
    DataVersionManager,
    analyze_distribution,
    get_data_summary,
)
from app.logger_config import setup_progress_logger, setup_system_logger

# 全局 logger，會在 setup_experiment_dir 中初始化
logger: logging.Logger


class TrainingProgressCallback(TrainerCallback):
    """訓練進度記錄 callback"""

    def __init__(self, log_file):
        super().__init__()
        self.logger = setup_progress_logger(log_file)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """記錄訓練進度"""
        if logs:
            metrics = []
            for key in ["loss", "learning_rate", "epoch", "eval_loss", "eval_accuracy"]:
                if key in logs:
                    value = logs[key]
                    metrics.append(f"{key}={value:.4f}")

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


def setup_device(config):
    """設置訓練設備"""
    if config.training.device:
        return torch.device(config.training.device)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("🚀 使用 CUDA GPU 加速")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 使用 MPS 加速（Apple Silicon）")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ 未檢測到 GPU，使用 CPU 模式")

    return device


def load_model_and_tokenizer(config, device):
    """載入模型與 tokenizer"""
    logger.info("📥 載入模型與 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name, num_labels=config.model.num_labels
    ).to(device)
    logger.info(f"✅ 模型載入完成: {config.model.name}")
    return model, tokenizer


def load_and_process_data(config, tokenizer):
    """載入與處理資料"""
    logger.info("📊 載入資料集...")
    dataset = load_dataset(config.data.dataset_name, config.data.dataset_config)
    train_small = dataset["train"].select(range(config.data.train_samples))
    eval_small = dataset["validation"].select(range(config.data.eval_samples))
    logger.info(f"   - 訓練資料: {len(train_small)} 筆")
    logger.info(f"   - 驗證資料: {len(eval_small)} 筆")

    # 資料分析
    logger.info("📋 進行資料分析與管理...")
    summary = get_data_summary(train_small)
    logger.info("📊 資料摘要:")
    logger.info(f"   - 特徵數: {summary['num_features']}")
    logger.info(f"   - 特徵名稱: {summary['feature_names']}")

    distribution_analysis = analyze_distribution(train_small)
    logger.info("📊 類別分布:")
    logger.info(f"   - 類別數: {distribution_analysis['num_classes']}")
    logger.info(f"   - 各類別數量: {distribution_analysis['label_counts']}")
    logger.info(f"   - 不平衡比例: {distribution_analysis['imbalance_ratio']:.2f}:1")
    logger.info(
        f"   - 是否平衡: {'✅' if distribution_analysis['is_balanced'] else '❌'}"
    )

    # 資料驗證
    validator = DataValidator(logger)
    validator.set_validation_rules(config.data.validation_rules)
    validation_report = validator.validate_dataset(train_small, ["sentence"])
    total_issues = sum(
        len(issue_list) for issue_list in validation_report["issues"].values()
    )

    if total_issues > 0:
        logger.warning(f"⚠️ 發現 {total_issues} 個資料問題")
        train_small = validator.clean_dataset(
            train_small, ["sentence"], validation_report
        )
        logger.info(f"🧹 資料清理完成，剩餘 {len(train_small)} 筆訓練資料")
    else:
        logger.info("✅ 資料驗證通過，無問題發現")

    # 版本管理
    try:
        version_manager = DataVersionManager(logger=logger)
        current_version = version_manager.get_current_version()
        if current_version:
            logger.info(f"📦 當前資料版本: {current_version}")
        else:
            version_name = f"sst2_train_{len(train_small)}samples"
            version_manager.create_version(
                train_small,
                version_name,
                description=f"SST-2 訓練集，經過清理，{len(train_small)} 筆資料",
                cleaning_strategy="移除空值、HTML標籤清理、重複資料移除",
                source_info={
                    "dataset": f"{config.data.dataset_name}/{config.data.dataset_config}",
                    "split": "train",
                    "original_samples": config.data.train_samples,
                    "cleaned_samples": len(train_small),
                },
            )
            logger.info(f"📦 創建資料版本: {version_name}")
    except Exception as e:
        logger.warning(f"⚠️ 版本管理失敗: {e}")

    logger.info("=" * 50)

    # 資料處理
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length,
        )

    train_dataset = train_small.map(tokenize, batched=True)
    eval_dataset = eval_small.map(tokenize, batched=True)
    logger.info("✅ 訓練和驗證資料集處理完成")

    return train_dataset, eval_dataset


def setup_lora(config, model, device):
    """設置 LoRA"""
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_config).to(device)
    logger.info("✅ LoRA 配置完成")

    # 顯示可訓練參數數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"📊 可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.2f}%)"
    )

    return model


def setup_training(config, model, train_dataset, eval_dataset, exp_dir):
    """設置訓練"""
    # 評估方法
    logger.info("📈 設置評估方法...")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 訓練參數
    logger.info("⚙️ 設置訓練參數...")
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        eval_strategy=config.training.eval_strategy,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        report_to=None,
    )

    logger.info("📝 訓練參數:")
    logger.info(f"   - 學習率: {training_args.learning_rate}")
    logger.info(f"   - 批次大小: {training_args.per_device_train_batch_size}")
    logger.info(f"   - 訓練輪數: {training_args.num_train_epochs}")
    logger.info(f"   - 記錄頻率: 每 {training_args.logging_steps} 步")

    # 創建自定義 callback，使用實驗目錄中的日誌文件
    progress_callback = TrainingProgressCallback(exp_dir / "logs.txt")

    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[progress_callback],
    )

    return trainer


def train_and_evaluate(config, trainer):
    """訓練與評估"""
    logger.info("🚀 開始訓練...")
    logger.info("=" * 50)

    # 訓練
    train_result = trainer.train()
    logger.info("🎉 訓練完成！")
    logger.info("=" * 50)

    # 評估
    logger.info("📊 評估模型...")
    eval_result = trainer.evaluate()
    logger.info(f"✅ 驗證準確率: {eval_result['eval_accuracy']:.4f}")

    # 保存模型
    output_dir = Path(config.training.output_dir) / "final_model"
    logger.info(f"💾 保存模型到 {output_dir}...")
    trainer.save_model(str(output_dir))
    logger.info("✅ 模型保存完成")

    # 訓練總結
    logger.info("🎯 訓練總結:")
    logger.info(f"   - 總訓練步數: {train_result.global_step}")
    logger.info(f"   - 總訓練時間: {train_result.metrics['train_runtime']:.2f} 秒")
    logger.info(f"   - 驗證準確率: {eval_result['eval_accuracy']:.4f}")
    logger.info(f"   - 模型保存位置: {output_dir}")

    return train_result, eval_result


def save_experiment_results(exp_dir, config, train_result, eval_result):
    """保存實驗結果"""
    # 準備實驗結果
    metrics = {
        "train": {
            "global_step": train_result.global_step,
            "runtime": train_result.metrics["train_runtime"],
        },
        "eval": {"accuracy": eval_result["eval_accuracy"]},
        "timestamp": datetime.now().isoformat(),
    }

    # 保存指標
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 保存配置
    config_dict = config.dict()
    config_dict["results"] = metrics
    config_file = exp_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)

    # 保存到資料庫
    from .db import Database, ExperimentRecord

    db = Database()
    db.save_experiment(
        ExperimentRecord(
            id=exp_dir.name,  # 使用實驗目錄名稱作為 ID
            name=config.experiment_name,
            created_at=datetime.fromisoformat(metrics["timestamp"]),
            config_path=str(config_file),
            train_runtime=metrics["train"]["runtime"],
            eval_accuracy=metrics["eval"]["accuracy"],
            log_path=str(exp_dir / "logs.txt"),
        )
    )

    logger.info(f"✅ 實驗結果已保存到 {exp_dir} 和資料庫")


def setup_experiment_dir(config):
    """設置實驗目錄"""
    global logger

    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 建立實驗目錄
    exp_dir = Path("results") / f"{config.experiment_name}_{timestamp}"
    exp_dir.mkdir(exist_ok=True)

    # 建立子目錄
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # 設置日誌文件
    log_file = exp_dir / "logs.txt"

    # 更新配置中的路徑
    config.training.output_dir = str(artifacts_dir)

    # 設置全局 logger
    logger = setup_system_logger(name=f"experiment_{timestamp}", log_file=str(log_file))

    return exp_dir


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="配置文件路徑"
    )
    parser.add_argument("--experiment_name", type=str, help="實驗名稱")
    parser.add_argument("--learning_rate", type=float, help="學習率")
    parser.add_argument("--epochs", type=int, help="訓練輪數")
    parser.add_argument("--train_samples", type=int, help="訓練樣本數")
    parser.add_argument("--device", type=str, help="訓練設備")
    return parser.parse_args()


def main(config=None):
    """主函數

    Args:
        config: 配置對象，如果為 None 則從命令列參數載入

    Returns:
        tuple: (train_result, eval_result) 訓練結果和評估結果
    """
    if config is None:
        # 解析參數
        args = parse_args()

        # 載入配置
        config = load_config(args.config)

        # 更新配置
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.epochs:
            config.training.num_train_epochs = args.epochs
        if args.train_samples:
            config.data.train_samples = args.train_samples
        if args.device:
            config.training.device = args.device

    # 設置實驗目錄和日誌
    exp_dir = setup_experiment_dir(config)
    logger.info(f"📂 實驗目錄：{exp_dir}")

    # 設置設備
    device = setup_device(config)

    # 載入模型
    model, tokenizer = load_model_and_tokenizer(config, device)

    # 載入資料
    train_dataset, eval_dataset = load_and_process_data(config, tokenizer)

    # 設置 LoRA
    model = setup_lora(config, model, device)

    # 設置訓練
    trainer = setup_training(config, model, train_dataset, eval_dataset, exp_dir)

    # 訓練與評估
    train_result, eval_result = train_and_evaluate(config, trainer)

    # 保存實驗結果
    save_experiment_results(exp_dir, config, train_result, eval_result)

    return train_result, eval_result


if __name__ == "__main__":
    main()
