"""
訓練任務定義
"""

from typing import Dict

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure

from app.core.config import Config
from app.core.logger import setup_system_logger
from app.exceptions import (
    OutOfMemoryError,
    TrainingError,
    TrainingTimeoutError,
)
from app.tasks import celery_app
from app.train import (
    load_and_process_data,
    load_model_and_tokenizer,
    setup_device,
    setup_lora,
    setup_training,
    train_and_evaluate,
)

# 配置任務錯誤日誌
task_logger = setup_system_logger(
    name="celery.task.error", log_file="logs/task_errors.log", console_output=True
)


@task_failure.connect
def handle_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """處理任務失敗的信號處理器

    記錄任務失敗的詳細信息，包括：
    - 任務 ID
    - 錯誤類型
    - 錯誤訊息
    - 重試次數（如果有）
    """
    retry_count = kwargs.get("request", {}).get("retries", 0)
    task_name = sender.name if sender else "unknown"

    task_logger.error(
        f"Task {task_name}[{task_id}] failed (retry {retry_count}): "
        f"{exception.__class__.__name__} - {str(exception)}"
    )


@celery_app.task(
    autoretry_for=(OutOfMemoryError, SoftTimeLimitExceeded),
    retry_backoff=True,
    retry_backoff_max=600,  # 最大延遲 10 分鐘
    retry_jitter=True,  # 添加隨機變化以避免同時重試
    max_retries=3,
    soft_time_limit=3600,  # 1 小時超時
)
def train_lora(config: Dict) -> Dict:
    """執行 LoRA 訓練任務

    Args:
        config: 完整的訓練配置字典

    Returns:
        Dict: 包含訓練結果的字典

    Raises:
        OutOfMemoryError: 當訓練過程中發生記憶體不足
        TrainingTimeoutError: 當訓練超過時間限制
        TrainingError: 其他訓練相關錯誤
    """
    try:
        config = Config(**config)

        # 設置訓練環境
        device = setup_device(config)
        model, tokenizer = load_model_and_tokenizer(config, device)
        train_dataset, eval_dataset = load_and_process_data(config, tokenizer)
        model = setup_lora(config, model, device)
        trainer = setup_training(
            config, model, train_dataset, eval_dataset, "results/test"
        )
        train_result, eval_result = train_and_evaluate(config, trainer)

        # 返回結果
        return {
            "status": "success",
            "train": {
                "global_step": train_result.global_step,
                "runtime": train_result.metrics["train_runtime"],
            },
            "eval": {"accuracy": eval_result["eval_accuracy"]},
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise OutOfMemoryError(f"訓練過程記憶體不足: {str(e)}")
        raise TrainingError(f"訓練過程發生錯誤: {str(e)}")
    except SoftTimeLimitExceeded:
        raise TrainingTimeoutError("訓練超過時間限制")
    except Exception as e:
        raise TrainingError(f"未預期的錯誤: {str(e)}")
