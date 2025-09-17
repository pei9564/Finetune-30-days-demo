"""
實驗效能分析工具
"""

import pandas as pd

from app.db import Database, ExperimentFilter


def analyze_experiments(
    filter_params: ExperimentFilter = None,
    sort_by: str = "created_at",
    desc: bool = True,
) -> pd.DataFrame:
    """分析實驗記錄"""
    db = Database()
    records = db.list_experiments(filter_params, sort_by, desc)

    if not records:
        return pd.DataFrame()

    # 轉換為 DataFrame
    df = pd.DataFrame(
        [
            {
                "實驗名稱": r.name,
                "模型": r.model_name,
                "資料集": r.dataset_name,
                "訓練樣本數": r.train_samples,
                "批次大小": r.batch_size,
                "學習率": r.learning_rate,
                "訓練輪數": r.num_epochs,
                "訓練時間": r.train_runtime,
                "處理速度": r.tokens_per_sec,
                "準確率": r.eval_accuracy,
                "CPU使用率": r.cpu_percent,
                "記憶體": r.memory_gb,
                "時間戳": r.created_at,
            }
            for r in records
        ]
    )

    return df


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """格式化指標顯示"""
    if df.empty:
        return df

    formatted = df.copy()
    formatted["訓練時間"] = formatted["訓練時間"].map("{:.2f}秒".format)
    formatted["處理速度"] = formatted["處理速度"].map("{:.2f} tokens/s".format)
    formatted["準確率"] = formatted["準確率"].map("{:.2%}".format)
    formatted["CPU使用率"] = formatted["CPU使用率"].map("{:.1f}%".format)
    formatted["記憶體"] = formatted["記憶體"].map("{:.2f}GB".format)
    formatted["學習率"] = formatted["學習率"].map("{:.1e}".format)

    return formatted


def print_statistics(df: pd.DataFrame):
    """顯示統計資訊"""
    print("\n📊 實驗效能統計")
    print("=" * 80)

    if df.empty:
        print("⚠️ 沒有找到任何實驗記錄")
        return

    db = Database()
    stats = db.get_statistics()

    print(f"總實驗數：{stats['total_experiments']}")
    print(f"平均訓練時間：{stats['avg_runtime']:.2f} 秒")
    print(f"平均處理速度：{stats['avg_tokens_per_sec']:.2f} tokens/sec")
    print(f"平均準確率：{stats['avg_accuracy']:.2%}")
    print(f"最佳準確率：{stats['best_accuracy']:.2%}")
    print(f"最短訓練時間：{stats['min_runtime']:.2f} 秒")
    print(f"平均 CPU 使用率：{stats['avg_cpu_percent']:.1f}%")
    print(f"平均記憶體使用：{stats['avg_memory_gb']:.2f} GB")


def print_comparison(df: pd.DataFrame):
    """顯示實驗比較"""
    if df.empty:
        return

    print("\n📋 實驗比較")
    print("=" * 80)

    # 設置顯示選項
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # 格式化並移除時間戳列
    df_display = format_metrics(df).drop(columns=["時間戳"])

    # 顯示表格
    print(df_display.to_string(index=False))


def print_group_stats(df: pd.DataFrame, group_by: str):
    """顯示分組統計"""
    if df.empty:
        return

    group_name = {"model_name": "模型", "dataset_name": "資料集"}[group_by]

    print(f"\n📊 按 {group_name} 分組統計")
    print("=" * 80)

    # 計算分組統計
    grouped = (
        df.groupby(group_name)
        .agg(
            {
                "訓練時間": "mean",
                "處理速度": "mean",
                "準確率": "mean",
                "實驗名稱": "count",
            }
        )
        .rename(columns={"實驗名稱": "實驗數"})
    )

    # 格式化
    formatted = pd.DataFrame(
        {
            "訓練時間": grouped["訓練時間"].map("{:.2f}秒".format),
            "處理速度": grouped["處理速度"].map("{:.2f} tokens/s".format),
            "準確率": grouped["準確率"].map("{:.2%}".format),
            "實驗數": grouped["實驗數"],
        }
    )

    print(formatted.to_string())


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="分析實驗效能")
    parser.add_argument(
        "--group-by", choices=["model_name", "dataset_name"], help="按指定欄位分組"
    )
    parser.add_argument("--model", help="篩選特定模型")
    parser.add_argument("--dataset", help="篩選特定資料集")
    parser.add_argument("--min-accuracy", type=float, help="最低準確率")
    parser.add_argument(
        "--sort-by",
        default="created_at",
        choices=["created_at", "eval_accuracy", "train_runtime", "tokens_per_sec"],
        help="排序依據",
    )
    parser.add_argument("--desc", action="store_true", help="降序排序")

    args = parser.parse_args()

    # 準備篩選條件
    filter_params = ExperimentFilter(
        model_name=args.model, dataset_name=args.dataset, min_accuracy=args.min_accuracy
    )

    # 獲取數據
    df = analyze_experiments(filter_params, args.sort_by, args.desc)

    # 是否需要分組
    if args.group_by:
        print_group_stats(df, args.group_by)
    else:
        # 顯示統計
        print_statistics(df)

        # 顯示比較
        print_comparison(df)

        # 提供分析建議
        print("\n💡 分析建議：")
        print("  1. 按模型分組：make analyze-by-model")
        print("  2. 按資料集分組：make analyze-by-dataset")
        print(
            "  3. 篩選特定模型：python -m app.tools.analyze_metrics --model <model_name>"
        )
        print("  4. 篩選準確率：python -m app.tools.analyze_metrics --min-accuracy 0.8")


if __name__ == "__main__":
    main()
