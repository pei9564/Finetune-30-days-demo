"""
訓練日誌與進度追蹤工具
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


def calculate_tokens_per_sec(total_tokens: int, runtime: float) -> float:
    """計算每秒處理的 token 數

    Args:
        total_tokens: 總 token 數
        runtime: 運行時間（秒）

    Returns:
        float: 每秒處理的 token 數
    """
    if runtime > 0 and total_tokens > 0:
        return total_tokens / runtime
    return 0.0


def save_training_metrics(
    metrics_file: str,
    train_result: Any,
    eval_result: Optional[Dict],
    system_metrics: Dict,
    tokens_info: Dict,
    start_time: float,
) -> Dict:
    """保存訓練指標

    Args:
        metrics_file: 指標文件路徑
        train_result: 訓練結果
        eval_result: 評估結果
        system_metrics: 系統指標
        tokens_info: token 相關信息
        start_time: 開始時間

    Returns:
        Dict: 完整的指標數據
    """
    # 從訓練結果中獲取運行時間
    metrics = getattr(train_result, "metrics", {})
    train_runtime = metrics.get("train_runtime", 0.0)
    if not train_runtime:
        train_runtime = time.time() - start_time  # 備用計時

    # 從訓練結果中提取指標
    train_metrics = {
        "global_step": getattr(train_result, "global_step", 0),
        "runtime": train_runtime,
        "tokens_per_sec": calculate_tokens_per_sec(
            tokens_info.get("total_tokens", 0), train_runtime
        ),
    }

    metrics = {
        "train": train_metrics,
        "eval": {
            "accuracy": eval_result.get("eval_accuracy", 0.0) if eval_result else 0.0,
            "loss": eval_result.get("eval_loss", 0.0) if eval_result else 0.0,
        },
        "system": system_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存到文件
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
