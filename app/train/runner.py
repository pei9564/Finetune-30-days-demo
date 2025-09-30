"""
訓練主流程相關功能
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import mlflow
import psutil
import torch
from datasets import Dataset, load_dataset
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
from app.core.mlflow_config import init_mlflow
from app.models.model_registry import ModelCard, registry
from app.tools.artifact_utils import save_artifact
from app.train.evaluator import TrainingProgressCallback, compute_metrics

logger = logging.getLogger(__name__)

DEFAULT_TEST_SIZE = 0.1
DEFAULT_SEED = 42
MEMORY_WARNING_THRESHOLD = 0.9


def load_model_and_tokenizer(
    config: Config, device: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """載入模型和分詞器

    Args:
        config: 訓練配置
        device: 設備名稱

    Returns:
        tuple: (model, tokenizer) 模型和分詞器
    """
    logger.info(f"🤖 載入模型 {config.model.name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # 移動模型到指定設備
    model = model.to(device)
    logger.info(f"✅ 模型已載入到 {device}")

    return model, tokenizer


def setup_lora(config: Config, model: PreTrainedModel, device: str) -> PreTrainedModel:
    """設置 LoRA 配置

    Args:
        config: 訓練配置
        model: 基礎模型
        device: 設備名稱

    Returns:
        PreTrainedModel: 添加 LoRA 後的模型
    """
    logger.info("🔧 設置 LoRA 配置...")
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )

    # 添加 LoRA 適配器
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # 打印參數統計
    model.print_trainable_parameters()
    logger.info("✅ LoRA 配置完成")

    return model


def setup_device(config: Config) -> str:
    """設置訓練設備

    Args:
        config: 訓練配置

    Returns:
        str: 設備名稱
    """
    if config.training.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.training.device

    logger.info(f"💻 使用設備: {device}")
    return device


def load_and_process_data(
    config: Config, tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """載入並處理數據

    Args:
        config: 訓練配置
        tokenizer: 分詞器

    Returns:
        tuple: (train_dataset, eval_dataset) 訓練集和驗證集
    """
    logger.info("📦 載入數據集...")

    dataset = load_dataset(
        config.data.dataset_name,
        config.data.subset,
        split="train",
        trust_remote_code=True,
    )

    # 隨機分割訓練集和驗證集
    dataset = dataset.train_test_split(test_size=DEFAULT_TEST_SIZE, seed=DEFAULT_SEED)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 如果指定了訓練樣本數，則只使用部分數據
    if config.training.train_samples:
        train_dataset = train_dataset.select(range(config.training.train_samples))

    logger.info(f"✅ 訓練集大小: {len(train_dataset)}")
    logger.info(f"✅ 驗證集大小: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def setup_training(
    config: Config,
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    exp_dir: str,
) -> Trainer:
    """設置訓練

    Args:
        config: 訓練配置
        model: 模型
        train_dataset: 訓練資料集
        eval_dataset: 驗證資料集
        exp_dir: 實驗目錄（未使用，保留以維持接口一致性）

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

    # 合併訓練參數日誌
    logger.info("📝 訓練參數:")
    logger.info(f"   - 學習率: {training_args.learning_rate}")
    logger.info(f"   - 批次大小: {training_args.per_device_train_batch_size}")
    logger.info(f"   - 訓練輪數: {training_args.num_train_epochs}")
    logger.info(f"   - 記錄頻率: 每 {training_args.logging_steps} 步")
    logger.info("   - 保存策略: 每個 epoch")
    logger.info("   - 評估策略: 每個 epoch")
    logger.info(f"   - 載入最佳模型: {training_args.load_best_model_at_end}")
    logger.info(f"   - 評估指標: {training_args.metric_for_best_model}")

    # 創建自定義 callback，使用 artifacts 目錄中的日誌文件
    log_file = os.path.join(str(config.training.output_dir), "logs.txt")
    # 確保日誌目錄存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    progress_callback = TrainingProgressCallback(log_file)

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


def train_and_evaluate(
    config: Config, trainer: Trainer
) -> Tuple[Dict, Optional[Dict], str]:
    """訓練與評估

    Args:
        config: 訓練配置
        trainer: 訓練器實例

    Returns:
        tuple: (train_result, eval_result, run_id) 訓練結果、評估結果和 MLflow run ID

    Raises:
        RuntimeError: 當記憶體不足時
    """
    # Initialize MLflow and start run
    mlflow_config = init_mlflow()

    with mlflow.start_run(
        experiment_id=mlflow_config["experiment_id"],
        run_name=f"{config.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    ) as run:
        logger.info("🚀 開始訓練...")
        logger.info("=" * 50)
        logger.info(f"MLflow 實驗 ID: {mlflow_config['experiment_id']}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_params(
            {
                "model_name": config.model.name,
                "batch_size": config.training.per_device_train_batch_size,
                "learning_rate": config.training.learning_rate,
                "epochs": config.training.num_train_epochs,
                "device": str(trainer.args.device),
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "lora_dropout": config.lora.lora_dropout,
            }
        )

        # 檢查初始記憶體狀態
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"初始 GPU 記憶體使用: {initial_gpu_memory:.2f}GB")

        try:
            # 訓練
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time

            logger.info("🎉 訓練完成！")

            # Log training metrics
            mlflow.log_metrics(
                {
                    "training_time": training_time,
                    "total_steps": train_result.global_step,
                    "train_loss": train_result.metrics.get("train_loss", 0.0),
                    "train_runtime": train_result.metrics["train_runtime"],
                    "train_samples_per_second": train_result.metrics[
                        "train_samples_per_second"
                    ],
                }
            )

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
                if (
                    peak_gpu_memory > total_gpu_memory * MEMORY_WARNING_THRESHOLD
                ):  # 使用超過 90% 的記憶體
                    logger.warning(
                        f"⚠️ GPU 記憶體使用率過高: {(peak_gpu_memory / total_gpu_memory) * 100:.1f}%"
                    )

            logger.info("=" * 50)

            # 評估
            logger.info("📊 評估模型...")
            eval_result = trainer.evaluate()
            logger.info(f"✅ 驗證準確率: {eval_result['eval_accuracy']:.4f}")

            # Log evaluation metrics
            mlflow.log_metrics(
                {
                    "eval_accuracy": eval_result["eval_accuracy"],
                    "eval_loss": eval_result["eval_loss"],
                }
            )

            # 保存模型到本地
            output_dir = os.path.join(config.training.output_dir, "final_model")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"💾 保存模型到 {output_dir}...")

            # 保存到本地
            trainer.save_model(output_dir)
            logger.info("✅ 模型保存完成")

            save_artifact(output_dir, run.info.run_id)

            # 記錄模型到 MLflow
            model_name = config.model.name.split("/")[-1]  # Get base name without org
            logger.info("📦 記錄模型到 MLflow...")

            # 使用 mlflow.pytorch.save_model 先保存到臨時目錄
            temp_model_dir = os.path.join(output_dir, "mlflow_model")
            os.makedirs(temp_model_dir, exist_ok=True)
            mlflow.pytorch.save_model(trainer.model, temp_model_dir)

            # 使用 log_artifacts 記錄到 MLflow
            mlflow.log_artifacts(temp_model_dir, "final_model")

            # 註冊模型到 MLflow Registry
            model_uri = f"runs:/{run.info.run_id}/final_model"
            logger.info("📝 註冊模型到 MLflow Registry...")
            registered_model = mlflow.register_model(
                model_uri=model_uri, name=model_name
            )
            logger.info(
                f"✅ 模型已註冊到 MLflow Registry: {model_name} v{registered_model.version}"
            )

            # 設置模型到 Staging 階段
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=registered_model.version, stage="Staging"
            )
            logger.info(
                f"✅ 已將模型設置為 Staging 階段: {model_name} v{registered_model.version}"
            )

            # 創建模型卡片
            model_id = f"task_{run.info.run_name}"  # 使用 run name 作為唯一標識

            # 創建新的模型卡片
            model_card = ModelCard(
                id=model_id,
                name=model_name,  # 使用 model_name 而不是 run_name
                base_model=config.model.name.split("/")[-1],
                language="en",
                task="text-classification",
                description="LoRA fine-tuned model on glue dataset",
                version=registered_model.version,  # 直接設置版本
                stage="Staging",  # 直接設置階段
                run_id=run.info.run_id,
                tags=["glue", "lora", config.model.name.split("/")[-1]],
            )

            # 保存模型卡片
            registry.save_model_card(model_card)
            logger.info(f"✅ 模型卡片已創建: {model_id}")

            # Log training logs if exists
            log_file = os.path.join(str(trainer.args.output_dir), "logs.txt")
            if os.path.exists(log_file):
                mlflow.log_artifact(log_file, "logs")
            else:
                logger.warning("⚠️ 找不到訓練日誌文件")

            # Log config file if exists
            config_path = os.path.join(str(trainer.args.output_dir), "config.yaml")
            if os.path.exists(config_path):
                mlflow.log_artifact(config_path, "config")
            else:
                logger.warning("⚠️ 找不到配置文件")

            # 訓練總結
            logger.info("🎯 訓練總結:")
            logger.info(f"   - 總訓練步數: {train_result.global_step}")
            logger.info(
                f"   - 總訓練時間: {train_result.metrics['train_runtime']:.2f} 秒"
            )
            logger.info(f"   - 驗證準確率: {eval_result['eval_accuracy']:.4f}")
            logger.info(f"   - 模型保存位置: {output_dir}")

            return train_result, eval_result, run.info.run_id

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
