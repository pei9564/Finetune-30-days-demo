"""
實驗追蹤資料庫模組
使用 SQLite 儲存實驗結果
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class ExperimentRecord(BaseModel):
    """實驗記錄模型"""

    id: str
    name: str
    created_at: datetime
    config_path: str
    train_runtime: float
    eval_accuracy: float
    log_path: str

    # 新增欄位
    tokens_per_sec: float = 0.0
    cpu_percent: float = 0.0
    memory_gb: float = 0.0
    model_name: str = ""
    dataset_name: str = ""
    train_samples: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    num_epochs: int = 0


class ExperimentFilter(BaseModel):
    """實驗篩選條件"""

    name: Optional[str] = None
    min_accuracy: Optional[float] = None
    max_runtime: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    model_name: Optional[str] = None
    dataset_name: Optional[str] = None


class Database:
    """資料庫管理類"""

    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """確保資料庫和資料表存在"""
        # 確保目錄存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # 建立資料表
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    config_path TEXT NOT NULL,
                    train_runtime FLOAT NOT NULL,
                    eval_accuracy FLOAT NOT NULL,
                    log_path TEXT NOT NULL,
                    tokens_per_sec FLOAT DEFAULT 0,
                    cpu_percent FLOAT DEFAULT 0,
                    memory_gb FLOAT DEFAULT 0,
                    model_name TEXT DEFAULT '',
                    dataset_name TEXT DEFAULT '',
                    train_samples INTEGER DEFAULT 0,
                    batch_size INTEGER DEFAULT 0,
                    learning_rate FLOAT DEFAULT 0,
                    num_epochs INTEGER DEFAULT 0
                )
            """)
            # 建立索引
            indexes = [
                "idx_created_at ON experiments(created_at)",
                "idx_name ON experiments(name)",
                "idx_model_name ON experiments(model_name)",
                "idx_dataset_name ON experiments(dataset_name)",
                "idx_accuracy ON experiments(eval_accuracy)",
                "idx_runtime ON experiments(train_runtime)",
            ]
            for idx in indexes:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx}")

    def save_experiment(self, experiment: ExperimentRecord) -> None:
        """儲存實驗記錄"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, name, created_at, config_path,
                    train_runtime, eval_accuracy, log_path,
                    tokens_per_sec, cpu_percent, memory_gb,
                    model_name, dataset_name, train_samples,
                    batch_size, learning_rate, num_epochs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.created_at.isoformat(),
                    experiment.config_path,
                    experiment.train_runtime,
                    experiment.eval_accuracy,
                    experiment.log_path,
                    experiment.tokens_per_sec,
                    experiment.cpu_percent,
                    experiment.memory_gb,
                    experiment.model_name,
                    experiment.dataset_name,
                    experiment.train_samples,
                    experiment.batch_size,
                    experiment.learning_rate,
                    experiment.num_epochs,
                ),
            )

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """查詢單一實驗記錄"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()
            if row:
                return ExperimentRecord(**dict(row))
            return None

    def list_experiments(
        self,
        filter_params: Optional[ExperimentFilter] = None,
        sort_by: str = "created_at",
        desc: bool = True,
        limit: int = 100,
    ) -> List[ExperimentRecord]:
        """列出實驗記錄"""
        query = ["SELECT * FROM experiments"]
        params = []

        # 處理篩選條件
        if filter_params:
            conditions = []
            if filter_params.name:
                conditions.append("name LIKE ?")
                params.append(f"%{filter_params.name}%")
            if filter_params.min_accuracy is not None:
                conditions.append("eval_accuracy >= ?")
                params.append(filter_params.min_accuracy)
            if filter_params.max_runtime is not None:
                conditions.append("train_runtime <= ?")
                params.append(filter_params.max_runtime)
            if filter_params.start_date:
                conditions.append("created_at >= ?")
                params.append(filter_params.start_date.isoformat())
            if filter_params.end_date:
                conditions.append("created_at <= ?")
                params.append(filter_params.end_date.isoformat())
            if filter_params.model_name:
                conditions.append("model_name = ?")
                params.append(filter_params.model_name)
            if filter_params.dataset_name:
                conditions.append("dataset_name = ?")
                params.append(filter_params.dataset_name)

            if conditions:
                query.append("WHERE " + " AND ".join(conditions))

        # 處理排序
        valid_sort_fields = {
            "created_at",
            "name",
            "train_runtime",
            "eval_accuracy",
            "tokens_per_sec",
            "train_samples",
        }
        if sort_by not in valid_sort_fields:
            sort_by = "created_at"
        query.append(f"ORDER BY {sort_by} {'DESC' if desc else 'ASC'}")

        # 處理分頁
        query.append("LIMIT ?")
        params.append(limit)

        # 執行查詢
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(" ".join(query), params)
            return [ExperimentRecord(**dict(row)) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """獲取實驗統計資訊"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_experiments,
                    AVG(train_runtime) as avg_runtime,
                    AVG(eval_accuracy) as avg_accuracy,
                    MAX(eval_accuracy) as best_accuracy,
                    MIN(train_runtime) as min_runtime,
                    AVG(tokens_per_sec) as avg_tokens_per_sec,
                    AVG(cpu_percent) as avg_cpu_percent,
                    AVG(memory_gb) as avg_memory_gb
                FROM experiments
            """)
            return dict(cursor.fetchone())
