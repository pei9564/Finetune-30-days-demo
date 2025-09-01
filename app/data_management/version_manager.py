"""
資料版本管理工具
提供資料集版本創建、載入、管理等功能
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset

from .dataset_analyzer import (
    analyze_distribution,
    get_data_summary,
    save_analysis_report,
)


class DataVersionManager:
    """資料版本管理系統"""

    def __init__(self, base_dir: str = "data", logger: Optional[logging.Logger] = None):
        self.base_dir = base_dir
        self.datasets_dir = os.path.join(base_dir, "datasets")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.logger = logger or logging.getLogger(__name__)

        # 確保目錄存在
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.metadata_file = os.path.join(self.metadata_dir, "versions.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """載入版本元數據"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "versions": {},
                "current_version": None,
                "created_at": datetime.now().isoformat(),
            }

    def _save_metadata(self):
        """保存版本元數據"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def create_version(
        self,
        dataset: Dataset,
        version_name: str,
        description: str = "",
        cleaning_strategy: str = "",
        source_info: Dict[str, Any] = None,
    ) -> str:
        """
        創建新的資料版本

        Args:
            dataset: 資料集
            version_name: 版本名稱
            description: 版本描述
            cleaning_strategy: 清理策略說明
            source_info: 來源資訊

        Returns:
            版本 ID
        """
        if version_name in self.metadata["versions"]:
            raise ValueError(f"版本 '{version_name}' 已存在")

        self.logger.info(f"📦 創建新版本: {version_name}")

        # 創建版本目錄
        version_dir = os.path.join(self.datasets_dir, version_name)
        os.makedirs(version_dir, exist_ok=True)

        # 保存資料集
        dataset_path = os.path.join(version_dir, "dataset.json")
        dataset.to_json(dataset_path)

        # 分析資料集
        summary = get_data_summary(dataset)

        # 如果有標籤欄位，分析分布
        distribution_analysis = None
        if "label" in dataset.features:
            distribution_analysis = analyze_distribution(dataset)

        # 創建版本記錄
        version_info = {
            "version_name": version_name,
            "description": description,
            "cleaning_strategy": cleaning_strategy,
            "source_info": source_info or {},
            "created_at": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "num_samples": len(dataset),
            "summary": summary,
            "distribution_analysis": distribution_analysis,
            "file_size_mb": os.path.getsize(dataset_path) / (1024 * 1024),
        }

        # 保存詳細分析
        analysis_path = os.path.join(version_dir, "analysis.json")
        save_analysis_report(version_info, analysis_path)
        version_info["analysis_path"] = analysis_path

        # 更新元數據
        self.metadata["versions"][version_name] = version_info
        self.metadata["current_version"] = version_name
        self._save_metadata()

        self.logger.info(f"✅ 版本創建完成: {version_name}")
        return version_name

    def load_version(self, version_name: str) -> Dataset:
        """載入指定版本的資料集"""
        if version_name not in self.metadata["versions"]:
            raise ValueError(f"版本 '{version_name}' 不存在")

        version_info = self.metadata["versions"][version_name]
        dataset_path = version_info["dataset_path"]

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"資料集文件不存在: {dataset_path}")

        dataset = Dataset.from_json(dataset_path)
        self.logger.info(f"📥 載入版本: {version_name} ({len(dataset)} 筆資料)")
        return dataset

    def get_current_version(self) -> Optional[str]:
        """獲取當前版本名稱"""
        return self.metadata["current_version"]

    def list_versions(self) -> Dict[str, Any]:
        """列出所有版本的基本資訊"""
        versions_info = {}
        for version_name, version_info in self.metadata["versions"].items():
            versions_info[version_name] = {
                "description": version_info["description"],
                "num_samples": version_info["num_samples"],
                "created_at": version_info["created_at"],
                "file_size_mb": version_info["file_size_mb"],
                "is_current": version_name == self.metadata["current_version"],
            }
        return versions_info


def create_sst2_versions_example():
    """示例：創建 SST-2 資料集版本"""
    # 設定 logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 創建版本管理器
    manager = DataVersionManager(logger=logger)

    # 載入原始資料
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"].select(range(500))  # 取 500 筆範例

    # 檢查當前版本
    current_version = manager.get_current_version()
    if current_version:
        logger.info(f"📦 當前版本: {current_version}")
    else:
        # 創建第一個版本
        manager.create_version(
            train_dataset,
            "v1_sst2_sample",
            description="SST-2 範例資料",
            source_info={"dataset": "glue/sst2", "split": "train", "samples": 500},
        )
        logger.info("✅ 已創建第一個版本")

    # 列出所有版本
    versions = manager.list_versions()
    logger.info("📋 所有版本:")
    for version_name, info in versions.items():
        current_mark = " (當前)" if info["is_current"] else ""
        logger.info(f"   - {version_name}: {info['num_samples']} 筆{current_mark}")
        logger.info(f"     {info['description']}")

    return manager


if __name__ == "__main__":
    # 執行示例
    create_sst2_versions_example()
