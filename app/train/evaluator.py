"""
評估相關功能
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import evaluate
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState

from app.core.logger import setup_progress_logger

logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """訓練進度記錄 callback"""

    def __init__(self, log_file: Union[str, Path]):
        super().__init__()
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
            for key in ["loss", "learning_rate", "epoch", "eval_loss", "eval_accuracy"]:
                if key in logs:
                    value = logs[key]
                    metrics.append(f"{key}={value:.4f}")

            if metrics:
                message = f"Step {state.global_step}: {' | '.join(metrics)}"
                self.logger.info(message)

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
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    eval_metrics.append(f"{key}={value:.4f}")
                else:
                    eval_metrics.append(f"{key}={value}")

            if eval_metrics:
                message = f"Evaluation: {' | '.join(eval_metrics)}"
                self.logger.info(message)


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
