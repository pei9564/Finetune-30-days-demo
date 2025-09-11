"""
訓練任務定義
"""

from typing import Dict

from train_lora_v2 import main as train_main

from . import celery_app


@celery_app.task
def train_lora(config: Dict) -> Dict:
    """執行 LoRA 訓練任務

    Args:
        config: 完整的訓練配置字典

    Returns:
        Dict: 包含訓練結果的字典
    """
    # 將字典轉換為配置對象
    from config import Config

    config = Config(**config)

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
