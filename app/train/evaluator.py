"""
評估相關功能
"""

import logging
import os
from typing import Dict, Optional

import evaluate
import mlflow
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState

from app.core.logger import setup_progress_logger

logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """訓練進度記錄 callback"""

    def __init__(self, log_file: str):
        super().__init__()
        # 確保日誌目錄存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = setup_progress_logger(log_file)

    def on_log(
        self,
        args: TrainerState,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """記錄訓練進度"""
        if logs:
            metrics = []
            mlflow_metrics = {}

            for key in ["loss", "learning_rate", "epoch", "eval_loss", "eval_accuracy"]:
                if key in logs:
                    value = logs[key]
                    metrics.append(f"{key}={value:.4f}")

                    # Log metrics to MLflow
                    if key != "learning_rate":  # Skip learning rate as it's a parameter
                        mlflow_metrics[key] = value

            if metrics:
                message = f"Step {state.global_step}: {' | '.join(metrics)}"
                self.logger.info(message)

                # Add step number to metrics and log to MLflow
                if mlflow_metrics:
                    mlflow.log_metrics(mlflow_metrics, step=state.global_step)

    def on_evaluate(
        self,
        args: TrainerState,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """記錄評估結果"""
        if metrics:
            eval_metrics = []
            mlflow_metrics = {}

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    eval_metrics.append(f"{key}={value:.4f}")
                    mlflow_metrics[f"eval_{key}"] = float(value)
                else:
                    eval_metrics.append(f"{key}={value}")

            if eval_metrics:
                message = f"Evaluation: {' | '.join(eval_metrics)}"
                self.logger.info(message)

                # Log evaluation metrics to MLflow
                if mlflow_metrics:
                    mlflow.log_metrics(mlflow_metrics, step=state.global_step)


def compute_metrics(eval_pred: tuple) -> Dict:
    """計算評估指標

    Args:
        eval_pred: (logits, labels) 的 tuple

    Returns:
        dict: 包含評估指標的字典
    """
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)
