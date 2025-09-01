"""
資料集分析工具
提供資料分布分析、摘要統計等功能
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset


def analyze_distribution(
    dataset: Dataset, label_column: str = "label"
) -> Dict[str, Any]:
    """
    分析資料集的類別分布

    Args:
        dataset: Hugging Face Dataset 物件
        label_column: 標籤欄位名稱

    Returns:
        包含統計資訊的字典
    """
    # 獲取標籤
    labels = dataset[label_column]

    # 統計各類別數量
    label_counts = Counter(labels)
    total_samples = len(labels)

    # 計算比例
    label_percentages = {
        label: (count / total_samples) * 100 for label, count in label_counts.items()
    }

    # 計算統計指標
    counts = list(label_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    analysis = {
        "total_samples": total_samples,
        "num_classes": len(label_counts),
        "label_counts": dict(label_counts),
        "label_percentages": label_percentages,
        "imbalance_ratio": imbalance_ratio,
        "is_balanced": imbalance_ratio <= 3.0,  # 認為 3:1 以內算平衡
        "analysis_timestamp": datetime.now().isoformat(),
    }

    return analysis


def compare_datasets(
    datasets: Dict[str, Dataset],
    label_column: str = "label",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    比較多個資料集的分布差異

    Args:
        datasets: 資料集字典 {"name": dataset}
        label_column: 標籤欄位名稱
        save_dir: 報告保存目錄

    Returns:
        比較結果字典
    """
    comparison = {"datasets": {}, "comparison_timestamp": datetime.now().isoformat()}

    # 分析每個資料集
    for name, dataset in datasets.items():
        analysis = analyze_distribution(dataset, label_column)
        comparison["datasets"][name] = analysis

    # 保存比較報告
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, "datasets_comparison.json")
        save_analysis_report(comparison, report_path)

    return comparison


def save_analysis_report(
    analysis_or_comparison: Dict[str, Any], save_path: str
) -> None:
    """
    保存分析報告為 JSON 文件

    Args:
        analysis_or_comparison: 分析結果或比較結果
        save_path: 保存路徑
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(analysis_or_comparison, f, ensure_ascii=False, indent=2)

    print(f"📄 分析報告已保存至: {save_path}")


def get_data_summary(dataset: Dataset) -> Dict[str, Any]:
    """
    獲取資料集基本摘要資訊

    Args:
        dataset: Hugging Face Dataset 物件

    Returns:
        摘要資訊字典
    """
    summary = {
        "num_samples": len(dataset),
        "num_features": len(dataset.features),
        "feature_names": list(dataset.features.keys()),
        "feature_types": {
            name: str(feature) for name, feature in dataset.features.items()
        },
    }

    # 檢查文本長度統計（如果有文本欄位）
    text_columns = []
    for col_name, feature in dataset.features.items():
        if "string" in str(feature).lower() or "text" in col_name.lower():
            text_columns.append(col_name)

    if text_columns:
        text_stats = {}
        for col in text_columns:
            texts = dataset[col]
            lengths = [len(text) if text else 0 for text in texts]
            text_stats[col] = {
                "avg_length": np.mean(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "std_length": np.std(lengths),
            }
        summary["text_statistics"] = text_stats

    return summary


# 示例使用函數
def analyze_glue_sst2_example():
    """示例：分析 GLUE SST-2 資料集"""
    from datasets import load_dataset

    print("📊 開始分析 SST-2 資料集...")

    # 載入資料集
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"].select(range(1000))  # 取 1000 筆範例

    # 基本摘要
    summary = get_data_summary(train_dataset)
    print("📋 資料集摘要:")
    print(f"   - 樣本數: {summary['num_samples']}")
    print(f"   - 特徵數: {summary['num_features']}")
    print(f"   - 特徵名稱: {summary['feature_names']}")

    # 分析分布
    analysis = analyze_distribution(train_dataset)
    print("\n📊 類別分布分析:")
    print(f"   - 總樣本數: {analysis['total_samples']}")
    print(f"   - 類別數: {analysis['num_classes']}")
    print(f"   - 各類別數量: {analysis['label_counts']}")
    print(f"   - 不平衡比例: {analysis['imbalance_ratio']:.2f}:1")
    print(f"   - 是否平衡: {'✅' if analysis['is_balanced'] else '❌'}")

    # 保存報告
    os.makedirs("data/metadata", exist_ok=True)
    save_analysis_report(analysis, "data/metadata/sst2_analysis.json")

    return analysis


if __name__ == "__main__":
    # 執行示例分析
    analyze_glue_sst2_example()
