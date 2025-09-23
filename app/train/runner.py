"""
訓練主流程相關功能
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from app.core.config import Config
from app.train.evaluator import TrainingProgressCallback, compute_metrics

logger = logging.getLogger(__name__)


def setup_device(config: Config) -> torch.device:
    """設置訓練設備

    Args:
        config: 訓練配置

    Returns:
        torch.device: 訓練設備
    """
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


def load_model_and_tokenizer(
    config: Config, device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """載入模型與 tokenizer

    Args:
        config: 訓練配置
        device: 訓練設備

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info("📥 載入模型與 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name, num_labels=config.model.num_labels
    ).to(device)
    logger.info(f"✅ 模型載入完成: {config.model.name}")
    return model, tokenizer


def setup_lora(
    config: Config, model: PreTrainedModel, device: torch.device
) -> PreTrainedModel:
    """設置 LoRA

    Args:
        config: 訓練配置
        model: 基礎模型
        device: 訓練設備

    Returns:
        PreTrainedModel: 加入 LoRA 後的模型
    """
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


def setup_training(
    config: Config,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    exp_dir: Path,
) -> Trainer:
    """設置訓練

    Args:
        config: 訓練配置
        model: 模型
        train_dataset: 訓練資料集
        eval_dataset: 驗證資料集
        exp_dir: 實驗目錄

    Returns:
        Trainer: 訓練器實例
    """
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


def train_and_evaluate(config: Config, trainer: Trainer) -> Tuple[Dict, Optional[Dict]]:
    """訓練與評估

    Args:
        config: 訓練配置
        trainer: 訓練器實例

    Returns:
        tuple: (train_result, eval_result) 訓練結果和評估結果

    Raises:
        RuntimeError: 當記憶體不足時
    """
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
