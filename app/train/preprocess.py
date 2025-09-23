"""
資料處理相關功能
"""

import logging
from typing import Dict, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from app.core.config import Config
from app.data import (
    DataValidator,
    DataVersionManager,
    analyze_distribution,
    get_data_summary,
)

logger = logging.getLogger(__name__)


def load_and_process_data(
    config: Config, tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """載入與處理資料

    Args:
        config: 訓練配置
        tokenizer: tokenizer 實例

    Returns:
        tuple: (訓練資料集, 驗證資料集)

    Raises:
        ValueError: 當資料載入或處理失敗時
    """
    logger.info("📊 載入資料集...")
    try:
        dataset = load_dataset(config.data.dataset_name, config.data.dataset_config)
    except Exception as e:
        raise ValueError(f"無法載入數據集 {config.data.dataset_name}: {str(e)}")

    # 檢查數據集是否存在必要的分割
    required_splits = ["train", "validation"]
    for split in required_splits:
        if split not in dataset:
            raise ValueError(f"數據集缺少必要的分割: {split}")

    # 選擇指定數量的樣本
    try:
        train_small = dataset["train"].select(range(config.data.train_samples))
        eval_small = dataset["validation"].select(range(config.data.eval_samples))
    except Exception as e:
        raise ValueError(f"選擇數據樣本時發生錯誤: {str(e)}")

    # 檢查數據集大小
    if len(train_small) == 0:
        raise ValueError("訓練數據集不能為空")
    if len(eval_small) == 0:
        raise ValueError("驗證數據集不能為空")

    logger.info(f"   - 訓練資料: {len(train_small)} 筆")
    logger.info(f"   - 驗證資料: {len(eval_small)} 筆")

    # 資料分析
    logger.info("📋 進行資料分析與管理...")
    summary = get_data_summary(train_small)
    logger.info("📊 資料摘要:")
    logger.info(f"   - 特徵數: {summary['num_features']}")
    logger.info(f"   - 特徵名稱: {summary['feature_names']}")

    distribution_analysis = analyze_distribution(train_small)
    logger.info("📊 類別分布:")
    logger.info(f"   - 類別數: {distribution_analysis['num_classes']}")
    logger.info(f"   - 各類別數量: {distribution_analysis['label_counts']}")
    logger.info(f"   - 不平衡比例: {distribution_analysis['imbalance_ratio']:.2f}:1")
    logger.info(
        f"   - 是否平衡: {'✅' if distribution_analysis['is_balanced'] else '❌'}"
    )

    # 資料驗證
    validator = DataValidator(logger)
    validator.set_validation_rules(config.data.validation_rules)
    validation_report = validator.validate_dataset(train_small, ["sentence"])
    total_issues = sum(
        len(issue_list) for issue_list in validation_report["issues"].values()
    )

    if total_issues > 0:
        logger.warning(f"⚠️ 發現 {total_issues} 個資料問題")
        train_small = validator.clean_dataset(
            train_small, ["sentence"], validation_report
        )
        logger.info(f"🧹 資料清理完成，剩餘 {len(train_small)} 筆訓練資料")
    else:
        logger.info("✅ 資料驗證通過，無問題發現")

    # 版本管理
    try:
        version_manager = DataVersionManager(logger=logger)
        current_version = version_manager.get_current_version()
        if current_version:
            logger.info(f"📦 當前資料版本: {current_version}")
        else:
            version_name = f"sst2_train_{len(train_small)}samples"
            version_manager.create_version(
                train_small,
                version_name,
                description=f"SST-2 訓練集，經過清理，{len(train_small)} 筆資料",
                cleaning_strategy="移除空值、HTML標籤清理、重複資料移除",
                source_info={
                    "dataset": f"{config.data.dataset_name}/{config.data.dataset_config}",
                    "split": "train",
                    "original_samples": config.data.train_samples,
                    "cleaned_samples": len(train_small),
                },
            )
            logger.info(f"📦 創建資料版本: {version_name}")
    except Exception as e:
        logger.warning(f"⚠️ 版本管理失敗: {e}")

    logger.info("=" * 50)

    # 資料處理
    def tokenize(batch: Dict) -> Dict:
        # 計算 token 長度
        token_lengths = [len(tokenizer.encode(text)) for text in batch["sentence"]]
        max_token_length = max(token_lengths)

        # 如果有超長序列，記錄警告
        if max_token_length > config.data.max_length:
            num_truncated = sum(
                1 for length in token_lengths if length > config.data.max_length
            )
            logger.warning(
                f"發現 {num_truncated} 個超長序列 "
                f"(最長: {max_token_length} tokens, "
                f"限制: {config.data.max_length} tokens)"
            )

        # 執行 tokenize
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.data.max_length,
            # 不返回 overflowing_tokens，因為它會改變序列長度
            return_length=True,  # 返回序列長度信息
        )

    train_dataset = train_small.map(tokenize, batched=True)
    eval_dataset = eval_small.map(tokenize, batched=True)
    logger.info("✅ 訓練和驗證資料集處理完成")

    return train_dataset, eval_dataset
