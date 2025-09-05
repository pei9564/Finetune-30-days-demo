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


class ExperimentFilter(BaseModel):
    """實驗篩選條件"""

    name: Optional[str] = None
    min_accuracy: Optional[float] = None
    max_runtime: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


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
                    log_path TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON experiments(created_at)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON experiments(name)")

    def save_experiment(self, experiment: ExperimentRecord) -> None:
        """儲存實驗記錄"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    id, name, created_at, config_path,
                    train_runtime, eval_accuracy, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.created_at.isoformat(),
                    experiment.config_path,
                    experiment.train_runtime,
                    experiment.eval_accuracy,
                    experiment.log_path,
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
                return ExperimentRecord(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    config_path=row["config_path"],
                    train_runtime=row["train_runtime"],
                    eval_accuracy=row["eval_accuracy"],
                    log_path=row["log_path"],
                )
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

            if conditions:
                query.append("WHERE " + " AND ".join(conditions))

        # 處理排序
        valid_sort_fields = {"created_at", "name", "train_runtime", "eval_accuracy"}
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
            return [
                ExperimentRecord(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    config_path=row["config_path"],
                    train_runtime=row["train_runtime"],
                    eval_accuracy=row["eval_accuracy"],
                    log_path=row["log_path"],
                )
                for row in cursor.fetchall()
            ]

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
                    MIN(train_runtime) as min_runtime
                FROM experiments
            """)
            return dict(cursor.fetchone())
