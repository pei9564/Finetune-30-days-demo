"""
訓練任務定義
"""

from typing import Dict, Optional

from ..config import load_config
from ..train_lora_v2 import main as train_main
from . import celery_app


@celery_app.task
def train_lora(
    config_path: str = "config/default.yaml",
    experiment_name: Optional[str] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    train_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """執行 LoRA 訓練任務

    Args:
        config_path: 配置文件路徑
        experiment_name: 實驗名稱
        learning_rate: 學習率
        epochs: 訓練輪數
        train_samples: 訓練樣本數
        device: 訓練設備

    Returns:
        Dict: 包含訓練結果的字典
    """
    # 載入配置
    config = load_config(config_path)

    # 更新配置
    if experiment_name:
        config.experiment_name = experiment_name
    if learning_rate:
        config.training.learning_rate = learning_rate
    if epochs:
        config.training.num_train_epochs = epochs
    if train_samples:
        config.data.train_samples = train_samples
    if device:
        config.training.device = device

    # 執行訓練
    train_result, eval_result = train_main(config)

    # 返回結果
    return {
        "status": "success",
        "train": {
            "global_step": train_result.global_step,
            "runtime": train_result.metrics["train_runtime"],
        },
        "eval": {"accuracy": eval_result["eval_accuracy"]},
    }
