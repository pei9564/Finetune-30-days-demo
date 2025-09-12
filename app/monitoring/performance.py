"""
效能監控模組
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
from transformers import Trainer


class PerformanceMonitor:
    def __init__(self, experiment_dir: str):
        """初始化效能監控器

        Args:
            experiment_dir: 實驗目錄路徑
        """
        self.experiment_dir = experiment_dir
        self.start_time = time.time()
        self.metrics_file = os.path.join(experiment_dir, "metrics.json")
        self.process = psutil.Process()

        # 初始化計數器
        self.total_tokens = 0
        self.sequence_lengths = []

    def update_sequence_length(self, batch_size: int, seq_length: int):
        """更新序列長度統計"""
        self.total_tokens += batch_size * seq_length
        self.sequence_lengths.append(seq_length)

    def get_system_metrics(self) -> Dict[str, float]:
        """獲取系統指標"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_gb": self.process.memory_info().rss / (1024 * 1024 * 1024),
        }

    def calculate_tokens_per_sec(self, runtime: float) -> float:
        """計算每秒處理的 token 數"""
        if runtime > 0 and self.total_tokens > 0:
            return self.total_tokens / runtime
        return 0.0

    def save_metrics(
        self, trainer: Trainer, train_result: Any, eval_result: Optional[Dict] = None
    ):
        """保存訓練指標"""
        # 從訓練結果中獲取運行時間
        metrics = getattr(train_result, "metrics", {})
        train_runtime = metrics.get("train_runtime", 0.0)
        if not train_runtime:
            train_runtime = time.time() - self.start_time  # 備用計時

        # 從訓練結果中提取指標
        train_metrics = {
            "global_step": getattr(train_result, "global_step", 0),
            "runtime": train_runtime,
            "tokens_per_sec": self.calculate_tokens_per_sec(train_runtime),
        }

        metrics = {
            "train": train_metrics,
            "eval": {
                "accuracy": eval_result.get("eval_accuracy", 0.0)
                if eval_result
                else 0.0,
                "loss": eval_result.get("eval_loss", 0.0) if eval_result else 0.0,
            },
            "system": self.get_system_metrics(),
            "timestamp": datetime.now().isoformat(),
        }

        # 保存到文件
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics
