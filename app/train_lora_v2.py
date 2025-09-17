"""
LoRA 訓練腳本 v2
使用統一配置系統
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import psutil
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
from app.db import Database, ExperimentRecord
from app.logger_config import setup_progress_logger, setup_system_logger
from app.monitoring import PerformanceMonitor
from app.tools.checkpoint_manager import CheckpointManager

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
    try:
        dataset = load_dataset(config.data.dataset_name, config.data.dataset_config)
    except Exception as e:
        raise ValueError(f"無法載入數據集 {config.data.dataset_name}: {str(e)}")

    # 檢查數據集是否存在必要的分割
    required_splits = ["train", "validation"]
    for split in required_splits:
        if split not in dataset:
            raise ValueError(f"數據集缺少必要的分割: {split}")

    # 選擇指定數量的樣本
    try:
        train_small = dataset["train"].select(range(config.data.train_samples))
        eval_small = dataset["validation"].select(range(config.data.eval_samples))
    except Exception as e:
        raise ValueError(f"選擇數據樣本時發生錯誤: {str(e)}")

    # 檢查數據集大小
    if len(train_small) == 0:
        raise ValueError("訓練數據集不能為空")
    if len(eval_small) == 0:
        raise ValueError("驗證數據集不能為空")

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
        # 計算 token 長度
        token_lengths = [len(tokenizer.encode(text)) for text in batch["sentence"]]
        max_token_length = max(token_lengths)

        # 如果有超長序列，記錄警告
        if max_token_length > config.data.max_length:
            num_truncated = sum(
                1 for length in token_lengths if length > config.data.max_length
            )
            logger.warning(
                f"發現 {num_truncated} 個超長序列 "
                f"(最長: {max_token_length} tokens, "
                f"限制: {config.data.max_length} tokens)"
            )

        # 執行 tokenize
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length,
            # 不返回 overflowing_tokens，因為它會改變序列長度
            return_length=True,  # 返回序列長度信息
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
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        report_to=None,
        # Checkpoint 相關配置
        save_strategy="epoch",  # 每個 epoch 保存一次
        save_total_limit=None,  # 不限制保存數量，由 CheckpointManager 管理
        eval_strategy="epoch",  # 每個 epoch 評估一次
        load_best_model_at_end=True,  # 訓練結束後載入最佳模型
        metric_for_best_model="eval_accuracy",  # 使用驗證準確率選擇最佳模型
        greater_is_better=True,  # 指標越大越好
    )

    logger.info("📝 訓練參數:")
    logger.info(f"   - 學習率: {training_args.learning_rate}")
    logger.info(f"   - 批次大小: {training_args.per_device_train_batch_size}")
    logger.info(f"   - 訓練輪數: {training_args.num_train_epochs}")
    logger.info(f"   - 記錄頻率: 每 {training_args.logging_steps} 步")

    logger.info("📝 Checkpoint 設置:")
    logger.info("   - 保存策略: 每個 epoch")
    logger.info("   - 評估策略: 每個 epoch")
    logger.info(f"   - 載入最佳模型: {training_args.load_best_model_at_end}")
    logger.info(f"   - 評估指標: {training_args.metric_for_best_model}")
    logger.info("   - 保留三個關鍵 checkpoints:")
    logger.info("     1. 最佳評估準確率")
    logger.info("     2. 最後一個（用於恢復訓練）")
    logger.info("     3. 訓練時間最短（用於快速實驗）")

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

    # 檢查初始記憶體狀態
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        logger.info(f"初始 GPU 記憶體使用: {initial_gpu_memory:.2f}GB")

    # 訓練
    try:
        train_result = trainer.train()
        logger.info("🎉 訓練完成！")

        # 記錄最大記憶體使用量
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            current_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"最大 GPU 記憶體使用: {peak_gpu_memory:.2f}GB")
            logger.info(f"當前 GPU 記憶體使用: {current_gpu_memory:.2f}GB")

            # 檢查是否接近記憶體限制
            total_gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
            if peak_gpu_memory > total_gpu_memory * 0.9:  # 使用超過 90% 的記憶體
                logger.warning(
                    f"⚠️ GPU 記憶體使用率過高: {(peak_gpu_memory / total_gpu_memory) * 100:.1f}%"
                )
    except RuntimeError as e:
        if "out of memory" in str(e):
            if torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                total_gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )
                raise RuntimeError(
                    f"GPU 記憶體不足: 已使用 {current_gpu_memory:.1f}GB / 總計 {total_gpu_memory:.1f}GB"
                ) from e
            else:
                current_memory = psutil.Process().memory_info().rss / 1024**3
                total_memory = psutil.virtual_memory().total / 1024**3
                raise RuntimeError(
                    f"CPU 記憶體不足: 已使用 {current_memory:.1f}GB / 總計 {total_memory:.1f}GB"
                ) from e
        raise

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


def save_experiment_results(exp_dir, config, train_result, eval_result, trainer):
    """保存實驗結果並管理 checkpoints"""
    # 清理當前實驗的 checkpoints
    artifacts_dir = exp_dir / "artifacts"
    checkpoint_manager = CheckpointManager(results_dir=str(artifacts_dir))
    checkpoint_manager.cleanup_experiment(artifacts_dir)

    # 創建效能監控器
    monitor = PerformanceMonitor(exp_dir)

    # 更新序列長度統計
    for batch in trainer.get_train_dataloader():
        monitor.update_sequence_length(
            len(batch["input_ids"]), batch["input_ids"].shape[1]
        )

    # 保存完整指標
    metrics = monitor.save_metrics(trainer, train_result, eval_result)

    # 保存配置
    config_dict = config.model_dump()  # 使用 model_dump 替代 dict
    config_dict["results"] = metrics
    config_file = exp_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)

    # 保存到資料庫
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
            # 新增效能指標
            tokens_per_sec=metrics["train"]["tokens_per_sec"],
            cpu_percent=metrics["system"]["cpu_percent"],
            memory_gb=metrics["system"]["memory_gb"],
            # 新增訓練參數
            model_name=config.model.name,
            dataset_name=config.data.dataset_name,
            train_samples=config.data.train_samples,
            batch_size=config.training.per_device_train_batch_size,
            learning_rate=config.training.learning_rate,
            num_epochs=config.training.num_train_epochs,
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
    save_experiment_results(exp_dir, config, train_result, eval_result, trainer)

    return train_result, eval_result


if __name__ == "__main__":
    main()
