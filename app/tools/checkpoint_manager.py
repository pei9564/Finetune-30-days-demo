"""
Checkpoint 管理工具

主要功能：
1. 分析並保留最佳評估準確率的 checkpoint
2. 保留最後一個 checkpoint（用於恢復訓練）
3. 保留訓練時間最短的 checkpoint（用於快速實驗）
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetrics:
    """Checkpoint 指標數據"""

    path: str
    accuracy: float = 0.0
    runtime: float = float("inf")
    is_last: bool = False


class CheckpointManager:
    """Checkpoint 管理器

    用於分析和管理訓練過程中產生的 checkpoints，根據指定的策略保留重要的 checkpoints。
    """

    def __init__(
        self,
        results_dir: str = "results",
        checkpoint_prefix: str = "checkpoint-",
    ):
        """初始化 Checkpoint 管理器

        Args:
            results_dir: 結果目錄路徑
            checkpoint_prefix: checkpoint 目錄前綴
        """
        self.results_dir = results_dir
        self.checkpoint_prefix = checkpoint_prefix

    def get_experiment_dirs(self) -> List[str]:
        """獲取所有實驗目錄

        Returns:
            List[str]: 實驗目錄列表
        """
        if not os.path.exists(self.results_dir):
            return []
        return [
            os.path.join(self.results_dir, d)
            for d in os.listdir(self.results_dir)
            if os.path.isdir(os.path.join(self.results_dir, d))
            and not d.startswith(".")
        ]

    def get_checkpoints(self, experiment_dir: str) -> List[str]:
        """獲取實驗的所有 checkpoint 目錄

        Args:
            experiment_dir: 實驗目錄路徑

        Returns:
            List[str]: checkpoint 目錄列表
        """
        if not os.path.exists(experiment_dir):
            return []

        # 檢查是否有必要的檔案
        checkpoints = []
        for item in os.listdir(experiment_dir):
            item_path = os.path.join(experiment_dir, item)
            if (
                os.path.isdir(item_path)
                and item.startswith(self.checkpoint_prefix)
                and os.path.exists(os.path.join(item_path, "adapter_config.json"))
                and os.path.exists(os.path.join(item_path, "adapter_model.safetensors"))
            ):
                checkpoints.append(item_path)

        # 按照創建時間排序
        return sorted(checkpoints, key=lambda x: os.path.getctime(x), reverse=True)

    def read_checkpoint_metrics(self, checkpoint: str) -> Optional[CheckpointMetrics]:
        """讀取 checkpoint 的評估指標

        Args:
            checkpoint: checkpoint 目錄路徑

        Returns:
            Optional[CheckpointMetrics]: checkpoint 指標數據，如果讀取失敗則返回 None
        """
        try:
            state_file = os.path.join(checkpoint, "trainer_state.json")
            if not os.path.exists(state_file):
                return None

            with open(state_file) as f:
                state = json.load(f)
                return CheckpointMetrics(
                    path=checkpoint,
                    accuracy=state.get("best_metric", 0.0),
                    runtime=state.get("total_flos", float("inf")),
                )
        except Exception as e:
            logger.warning(f"⚠️ 讀取 checkpoint 指標失敗 {checkpoint}: {e}")
            return None

    def analyze_checkpoints(
        self, experiment_dir: str
    ) -> Tuple[Set[str], Dict[str, CheckpointMetrics]]:
        """分析實驗的 checkpoints 並選擇要保留的檔案

        Args:
            experiment_dir: 實驗目錄路徑

        Returns:
            Tuple[Set[str], Dict[str, CheckpointMetrics]]:
                - 要保留的 checkpoint 路徑集合
                - 保留的 checkpoint 指標信息
        """
        checkpoints = self.get_checkpoints(experiment_dir)
        if not checkpoints:
            return set(), {}

        # 讀取所有 checkpoint 的指標
        metrics_list = []
        for checkpoint in checkpoints:
            metrics = self.read_checkpoint_metrics(checkpoint)
            if metrics:
                metrics_list.append(metrics)

        if not metrics_list:
            return set(), {}

        # 標記最後一個 checkpoint
        metrics_list[-1].is_last = True

        # 選擇要保留的 checkpoints
        to_keep = set()
        kept_metrics = {}

        # 1. 最佳評估準確率
        best_checkpoint = max(metrics_list, key=lambda x: x.accuracy)
        to_keep.add(best_checkpoint.path)
        kept_metrics["best"] = best_checkpoint

        # 2. 最後一個 checkpoint
        last_checkpoint = next(x for x in metrics_list if x.is_last)
        to_keep.add(last_checkpoint.path)
        kept_metrics["last"] = last_checkpoint

        # 3. 最快的 checkpoint
        fastest_checkpoint = min(metrics_list, key=lambda x: x.runtime)
        to_keep.add(fastest_checkpoint.path)
        kept_metrics["fastest"] = fastest_checkpoint

        return to_keep, kept_metrics

    def cleanup_experiment(self, experiment_dir: str) -> None:
        """清理單個實驗的 checkpoints，保留指定的重要 checkpoints

        Args:
            experiment_dir: 實驗目錄路徑
        """
        try:
            to_keep, kept_metrics = self.analyze_checkpoints(experiment_dir)
            if not to_keep:
                return

            # 刪除不需要保留的 checkpoints
            for checkpoint in self.get_checkpoints(experiment_dir):
                if checkpoint not in to_keep:
                    try:
                        shutil.rmtree(checkpoint)
                        logger.info(
                            f"🗑️ 已刪除 checkpoint: {os.path.basename(checkpoint)}"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ 刪除 checkpoint 失敗 {checkpoint}: {e}")

            # 記錄保留的 checkpoints
            logger.info("✅ Checkpoint 清理完成，保留:")
            logger.info(
                f"   - 最佳準確率 ({kept_metrics['best'].accuracy:.4f}): {os.path.basename(kept_metrics['best'].path)}"
            )
            logger.info(
                f"   - 最後檢查點: {os.path.basename(kept_metrics['last'].path)}"
            )
            logger.info(
                f"   - 最快訓練: {os.path.basename(kept_metrics['fastest'].path)}"
            )

        except Exception as e:
            logger.error(f"清理實驗 {experiment_dir} 失敗: {e}")

    def cleanup_all(self) -> None:
        """清理所有實驗的 checkpoints"""
        for exp_dir in self.get_experiment_dirs():
            try:
                self.cleanup_experiment(exp_dir)
                logger.info(f"已清理實驗 {os.path.basename(exp_dir)} 的 checkpoints")
            except Exception as e:
                logger.error(f"清理實驗 {os.path.basename(exp_dir)} 失敗: {e}")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="清理舊的 checkpoints")
    parser.add_argument("--results-dir", default="results", help="結果目錄路徑")
    args = parser.parse_args()

    # 配置日誌
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # 執行清理
    manager = CheckpointManager(results_dir=args.results_dir)
    manager.cleanup_all()


if __name__ == "__main__":
    main()
