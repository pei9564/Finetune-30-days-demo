"""
資料驗證與清理工具
提供資料品質檢查、清理、驗證等功能
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset


class DataValidator:
    """資料驗證與清理工具"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules = {
            "min_text_length": 1,
            "max_text_length": 1000,
            "allow_empty": False,
            "remove_html": True,
            "remove_special_chars": False,
            "check_encoding": True,
        }
        self.issues_found = []

    def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """設定驗證規則"""
        self.validation_rules.update(rules)
        self.logger.info(f"📋 更新驗證規則: {rules}")

    def validate_dataset(
        self, dataset: Dataset, text_columns: List[str]
    ) -> Dict[str, Any]:
        """
        驗證整個資料集

        Args:
            dataset: 要驗證的資料集
            text_columns: 文本欄位列表

        Returns:
            驗證報告
        """
        self.logger.info(f"🔍 開始驗證資料集，共 {len(dataset)} 筆資料...")
        self.issues_found = []

        validation_report = {
            "total_samples": len(dataset),
            "text_columns": text_columns,
            "validation_rules": self.validation_rules.copy(),
            "issues": {
                "empty_values": [],
                "length_violations": [],
                "encoding_issues": [],
                "html_content": [],
                "duplicate_entries": [],
            },
            "statistics": {},
            "recommendations": [],
        }

        # 檢查每一筆資料
        for idx in range(len(dataset)):
            sample = dataset[idx]
            self._validate_sample(sample, idx, text_columns, validation_report)

        # 檢查重複資料
        self._check_duplicates(dataset, text_columns, validation_report)

        # 生成統計資訊
        self._generate_statistics(dataset, text_columns, validation_report)

        # 生成建議
        self._generate_recommendations(validation_report)

        self._log_validation_summary(validation_report)

        return validation_report

    def _validate_sample(
        self,
        sample: Dict[str, Any],
        idx: int,
        text_columns: List[str],
        report: Dict[str, Any],
    ) -> None:
        """驗證單一樣本"""

        for col in text_columns:
            if col not in sample:
                continue

            text = sample[col]

            # 檢查空值
            if text is None or (isinstance(text, str) and text.strip() == ""):
                report["issues"]["empty_values"].append(
                    {"index": idx, "column": col, "issue": "Empty or null value"}
                )
                continue

            if not isinstance(text, str):
                text = str(text)

            # 檢查長度
            text_length = len(text)
            if (
                text_length < self.validation_rules["min_text_length"]
                or text_length > self.validation_rules["max_text_length"]
            ):
                report["issues"]["length_violations"].append(
                    {
                        "index": idx,
                        "column": col,
                        "length": text_length,
                        "min_required": self.validation_rules["min_text_length"],
                        "max_allowed": self.validation_rules["max_text_length"],
                    }
                )

            # 檢查 HTML 標籤
            if self.validation_rules["remove_html"] and self._contains_html(text):
                report["issues"]["html_content"].append(
                    {
                        "index": idx,
                        "column": col,
                        "sample_text": text[:100] + "..." if len(text) > 100 else text,
                    }
                )

            # 檢查編碼問題
            if self.validation_rules["check_encoding"] and self._has_encoding_issues(
                text
            ):
                report["issues"]["encoding_issues"].append(
                    {
                        "index": idx,
                        "column": col,
                        "sample_text": text[:100] + "..." if len(text) > 100 else text,
                    }
                )

    def _check_duplicates(
        self, dataset: Dataset, text_columns: List[str], report: Dict[str, Any]
    ) -> None:
        """檢查重複資料"""
        seen_texts = set()
        duplicates = []

        for idx in range(len(dataset)):
            sample = dataset[idx]

            # 組合所有文本欄位作為唯一識別
            combined_text = ""
            for col in text_columns:
                if col in sample and sample[col]:
                    combined_text += str(sample[col]).strip() + " "

            combined_text = combined_text.strip()

            if combined_text in seen_texts:
                duplicates.append(
                    {
                        "index": idx,
                        "text": combined_text[:100] + "..."
                        if len(combined_text) > 100
                        else combined_text,
                    }
                )
            else:
                seen_texts.add(combined_text)

        report["issues"]["duplicate_entries"] = duplicates

    def _contains_html(self, text: str) -> bool:
        """檢查是否包含 HTML 標籤"""
        html_pattern = re.compile(r"<[^>]+>")
        return bool(html_pattern.search(text))

    def _has_encoding_issues(self, text: str) -> bool:
        """檢查編碼問題"""
        # 檢查常見的編碼問題字符
        # \ufffd 是 Unicode 替換字符（replacement character），通常表示編碼錯誤
        # \x00 是 null 字符，可能導致問題
        problematic_chars = ["\ufffd", "\x00"]
        return any(char in text for char in problematic_chars)

    def _generate_statistics(
        self, dataset: Dataset, text_columns: List[str], report: Dict[str, Any]
    ) -> None:
        """生成統計資訊"""
        stats = {}

        for col in text_columns:
            if col not in dataset.features:
                continue

            texts = [sample[col] for sample in dataset if sample.get(col)]
            text_lengths = [len(str(text)) for text in texts if text]

            if text_lengths:
                stats[col] = {
                    "count": len(text_lengths),
                    "avg_length": np.mean(text_lengths),
                    "min_length": min(text_lengths),
                    "max_length": max(text_lengths),
                    "std_length": np.std(text_lengths),
                }

        report["statistics"] = stats

    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """生成清理建議"""
        recommendations = []
        issues = report["issues"]

        if issues["empty_values"]:
            recommendations.append(
                {
                    "priority": "high",
                    "issue": f"發現 {len(issues['empty_values'])} 筆空值",
                    "action": "建議移除或填充空值",
                }
            )

        if issues["length_violations"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "issue": f"發現 {len(issues['length_violations'])} 筆長度異常",
                    "action": "建議調整文本長度限制或截斷/擴充文本",
                }
            )

        if issues["html_content"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "issue": f"發現 {len(issues['html_content'])} 筆包含 HTML",
                    "action": "建議清理 HTML 標籤",
                }
            )

        if issues["encoding_issues"]:
            recommendations.append(
                {
                    "priority": "high",
                    "issue": f"發現 {len(issues['encoding_issues'])} 筆編碼問題",
                    "action": "建議修復編碼或移除問題字符",
                }
            )

        if issues["duplicate_entries"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "issue": f"發現 {len(issues['duplicate_entries'])} 筆重複資料",
                    "action": "建議移除重複項目",
                }
            )

        report["recommendations"] = recommendations

    def _log_validation_summary(self, report: Dict[str, Any]) -> None:
        """記錄驗證摘要"""
        issues = report["issues"]
        total_issues = sum(len(issue_list) for issue_list in issues.values())

        self.logger.info("🔍 驗證完成!")
        self.logger.info(f"   - 總樣本數: {report['total_samples']}")
        self.logger.info(f"   - 發現問題: {total_issues} 筆")

        if total_issues > 0:
            self.logger.warning("⚠️ 發現的問題:")
            for issue_type, issue_list in issues.items():
                if issue_list:
                    self.logger.warning(f"   - {issue_type}: {len(issue_list)} 筆")
        else:
            self.logger.info("✅ 沒有發現資料問題!")

    def clean_dataset(
        self,
        dataset: Dataset,
        text_columns: List[str],
        validation_report: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """
        根據驗證結果清理資料集

        Args:
            dataset: 原始資料集
            text_columns: 文本欄位列表
            validation_report: 驗證報告（可選）

        Returns:
            清理後的資料集
        """
        self.logger.info("🧹 開始清理資料集...")

        if validation_report is None:
            validation_report = self.validate_dataset(dataset, text_columns)

        # 記錄要移除的索引
        indices_to_remove = set()

        # 處理空值
        for issue in validation_report["issues"]["empty_values"]:
            if not self.validation_rules["allow_empty"]:
                indices_to_remove.add(issue["index"])

        # 處理編碼問題
        for issue in validation_report["issues"]["encoding_issues"]:
            indices_to_remove.add(issue["index"])

        # 處理重複資料
        for issue in validation_report["issues"]["duplicate_entries"]:
            indices_to_remove.add(issue["index"])

        # 清理文本內容
        cleaned_data = []
        for idx in range(len(dataset)):
            if idx in indices_to_remove:
                continue

            sample = dict(dataset[idx])

            # 清理每個文本欄位
            for col in text_columns:
                if col in sample and sample[col]:
                    cleaned_text = self._clean_text(str(sample[col]))
                    sample[col] = cleaned_text

            cleaned_data.append(sample)

        # 創建新的資料集
        if cleaned_data:
            cleaned_dataset = Dataset.from_list(cleaned_data)
            self.logger.info(
                f"✅ 清理完成: {len(dataset)} → {len(cleaned_dataset)} 筆資料"
            )
            return cleaned_dataset
        else:
            self.logger.warning("⚠️ 清理後沒有剩餘資料!")
            return dataset

    def _clean_text(self, text: str) -> str:
        """清理單一文本"""
        # 移除 HTML 標籤
        if self.validation_rules["remove_html"]:
            text = re.sub(r"<[^>]+>", "", text)

        # 移除特殊字符（如果設定）
        if self.validation_rules["remove_special_chars"]:
            text = re.sub(r"[^\w\s]", "", text)

        # 清理空白字符
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def save_validation_report(self, report: Dict[str, Any], save_path: str) -> None:
        """保存驗證報告"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"📄 驗證報告已保存至: {save_path}")


# 示例使用函數
def validate_sst2_example():
    """示例：驗證 SST-2 資料集"""
    from datasets import load_dataset

    # 設定 logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 創建驗證器
    validator = DataValidator(logger)

    # 設定驗證規則
    validator.set_validation_rules(
        {
            "min_text_length": 5,
            "max_text_length": 500,
            "allow_empty": False,
            "remove_html": True,
        }
    )

    # 載入資料集
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"].select(range(1000))  # 取 1000 筆範例

    # 驗證資料集
    report = validator.validate_dataset(train_dataset, ["sentence"])

    # 保存驗證報告
    os.makedirs("data/metadata", exist_ok=True)
    validator.save_validation_report(
        report, "data/metadata/sst2_validation_report.json"
    )

    # 清理資料集
    cleaned_dataset = validator.clean_dataset(train_dataset, ["sentence"], report)

    return report, cleaned_dataset


if __name__ == "__main__":
    # 執行示例驗證
    validate_sst2_example()
