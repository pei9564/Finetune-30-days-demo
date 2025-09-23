"""
LoRA 訓練腳本 v2
使用統一配置系統
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import yaml

from app.core.config import load_config
from app.core.logger import setup_system_logger
from app.db import Database, ExperimentRecord
from app.monitor import SystemMetricsMonitor, save_training_metrics
from app.tools.checkpoint_manager import CheckpointManager
from app.train import (
    load_and_process_data,
    load_model_and_tokenizer,
    setup_device,
    setup_lora,
    setup_training,
    train_and_evaluate,
)

# 全局 logger，會在 setup_experiment_dir 中初始化
logger: logging.Logger


def setup_experiment_dir(config):
    """設置實驗目錄，添加錯誤處理"""
    global logger

    try:
        # 生成時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 確保 results 目錄存在
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True, parents=True)

        # 建立實驗目錄
        exp_dir = Path("results") / f"{config.experiment_name}_{timestamp}"
        exp_dir.mkdir(exist_ok=True, parents=True)

        # 建立子目錄
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True, parents=True)

        # 設置日誌文件
        log_file = exp_dir / "logs.txt"

        # 更新配置中的路徑
        config.training.output_dir = str(artifacts_dir)

        # 設置全局 logger
        logger = setup_system_logger(
            name=f"experiment_{timestamp}", log_file=str(log_file)
        )

        return exp_dir

    except (OSError, PermissionError) as e:
        # 如果創建目錄失敗，嘗試創建臨時目錄
        import logging

        temp_logger = logging.getLogger(__name__)
        temp_logger.error(f"創建實驗目錄失敗: {e}")

        try:
            # 創建臨時目錄作為備用
            temp_dir = Path("results") / f"temp_{timestamp}"
            temp_dir.mkdir(exist_ok=True, parents=True)

            # 設置基本的 logger
            logger = setup_system_logger(
                name=f"temp_experiment_{timestamp}", log_file=str(temp_dir / "logs.txt")
            )
            logger.warning(f"使用臨時目錄: {temp_dir}")

            return temp_dir
        except Exception as fallback_error:
            temp_logger.error(f"創建臨時目錄也失敗: {fallback_error}")
            raise RuntimeError(f"無法創建實驗目錄: {e}")


def save_experiment_results(exp_dir, config, train_result, eval_result, trainer):
    """保存實驗結果並管理 checkpoints"""
    # 清理當前實驗的 checkpoints
    artifacts_dir = exp_dir / "artifacts"
    checkpoint_manager = CheckpointManager(results_dir=str(artifacts_dir))
    checkpoint_manager.cleanup_experiment(artifacts_dir)

    # 創建系統監控器
    monitor = SystemMetricsMonitor()

    # 更新序列長度統計
    for batch in trainer.get_train_dataloader():
        monitor.update_sequence_length(
            len(batch["input_ids"]), batch["input_ids"].shape[1]
        )

    # 保存完整指標
    metrics = save_training_metrics(
        metrics_file=os.path.join(exp_dir, "metrics.json"),
        train_result=train_result,
        eval_result=eval_result,
        system_metrics=monitor.get_system_metrics(),
        tokens_info=monitor.get_tokens_info(),
        start_time=time.time(),
    )

    # 保存配置
    config_dict = config.model_dump()  # 使用 model_dump 替代 dict
    config_dict["results"] = metrics
    config_file = exp_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)

    # 保存到資料庫
    db = Database()
    db.save_experiment(
        ExperimentRecord(
            id=exp_dir.name,  # 使用實驗目錄名稱作為 ID
            name=config.experiment_name,
            created_at=datetime.fromisoformat(metrics["timestamp"]),
            config_path=str(config_file),
            train_runtime=metrics["train"]["runtime"],
            eval_accuracy=metrics["eval"]["accuracy"],
            log_path=str(exp_dir / "logs.txt"),
            # 新增效能指標
            tokens_per_sec=metrics["train"]["tokens_per_sec"],
            cpu_percent=metrics["system"]["cpu_percent"],
            memory_gb=metrics["system"]["memory_gb"],
            # 新增訓練參數
            model_name=config.model.name,
            dataset_name=config.data.dataset_name,
            train_samples=config.data.train_samples,
            batch_size=config.training.per_device_train_batch_size,
            learning_rate=config.training.learning_rate,
            num_epochs=config.training.num_train_epochs,
        )
    )

    logger.info(f"✅ 實驗結果已保存到 {exp_dir} 和資料庫")


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="LoRA 訓練腳本")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml", help="配置文件路徑"
    )
    parser.add_argument("--experiment_name", type=str, help="實驗名稱")
    parser.add_argument("--learning_rate", type=float, help="學習率")
    parser.add_argument("--epochs", type=int, help="訓練輪數")
    parser.add_argument("--train_samples", type=int, help="訓練樣本數")
    parser.add_argument("--device", type=str, help="訓練設備")
    return parser.parse_args()


def main(config=None):
    """主函數

    Args:
        config: 配置對象，如果為 None 則從命令列參數載入

    Returns:
        tuple: (train_result, eval_result) 訓練結果和評估結果
    """
    if config is None:
        # 解析參數
        args = parse_args()

        # 載入配置
        config = load_config(args.config)

        # 更新配置
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.epochs:
            config.training.num_train_epochs = args.epochs
        if args.train_samples:
            config.data.train_samples = args.train_samples
        if args.device:
            config.training.device = args.device

    # 設置實驗目錄和日誌
    exp_dir = setup_experiment_dir(config)
    logger.info(f"📂 實驗目錄：{exp_dir}")

    # 設置設備
    device = setup_device(config)

    # 載入模型
    model, tokenizer = load_model_and_tokenizer(config, device)

    # 載入資料
    train_dataset, eval_dataset = load_and_process_data(config, tokenizer)

    # 設置 LoRA
    model = setup_lora(config, model, device)

    # 設置訓練
    trainer = setup_training(config, model, train_dataset, eval_dataset, exp_dir)

    # 訓練與評估
    train_result, eval_result = train_and_evaluate(config, trainer)

    # 保存實驗結果
    save_experiment_results(exp_dir, config, train_result, eval_result, trainer)

    return train_result, eval_result


if __name__ == "__main__":
    main()
