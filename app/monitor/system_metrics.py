"""
系統資源監控工具
"""

from typing import Dict, List, Optional

import psutil


class SystemMetricsMonitor:
    """系統資源監控器"""

    def __init__(self):
        """初始化監控器"""
        self.process = psutil.Process()
        self.total_tokens = 0
        self.sequence_lengths: List[int] = []

    def update_sequence_length(self, batch_size: int, seq_length: int) -> None:
        """更新序列長度統計

        Args:
            batch_size: 批次大小
            seq_length: 序列長度
        """
        self.total_tokens += batch_size * seq_length
        self.sequence_lengths.append(seq_length)

    def get_system_metrics(self) -> Dict[str, float]:
        """獲取系統指標

        Returns:
            Dict[str, float]: 系統指標數據
        """
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_gb": self.process.memory_info().rss / (1024 * 1024 * 1024),
        }

    def get_tokens_info(self) -> Dict[str, int]:
        """獲取 token 相關信息

        Returns:
            Dict[str, int]: token 統計信息
        """
        return {
            "total_tokens": self.total_tokens,
            "avg_sequence_length": (
                sum(self.sequence_lengths) / len(self.sequence_lengths)
                if self.sequence_lengths
                else 0
            ),
            "max_sequence_length": max(self.sequence_lengths)
            if self.sequence_lengths
            else 0,
        }
