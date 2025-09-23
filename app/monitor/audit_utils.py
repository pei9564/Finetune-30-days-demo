"""
審計日誌工具
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


@contextmanager
def get_db():
    """資料庫連線管理器"""
    conn = sqlite3.connect("results/experiments.db")
    try:
        yield conn
    finally:
        conn.close()


def init_audit_table():
    """初始化審計日誌表"""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                status_code INTEGER,
                timestamp INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def save_audit_log(
    user_id: str,
    role: str,
    method: str,
    path: str,
    status_code: int,
):
    """保存審計日誌

    Args:
        user_id: 使用者 ID
        role: 使用者角色
        method: HTTP 方法
        path: API 路徑
        status_code: HTTP 狀態碼
    """
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (
                    user_id, role, action, method, path,
                    status_code, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    role,
                    f"{method} {path}",
                    method,
                    path,
                    status_code,
                    int(time.time()),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"保存審計日誌失敗: {e}")


def get_audit_logs(
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 100,
) -> List[Dict]:
    """查詢審計日誌

    Args:
        user_id: 過濾特定用戶
        role: 過濾特定角色
        start_time: 開始時間戳（Unix timestamp）
        end_time: 結束時間戳（Unix timestamp）
        limit: 返回記錄數量限制

    Returns:
        List[Dict]: 審計日誌記錄列表
    """
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if role:
        query += " AND role = ?"
        params.append(role)
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        cursor = conn.execute(query, params)
        return [
            {
                "id": row[0],
                "user_id": row[1],
                "role": row[2],
                "action": row[3],
                "method": row[4],
                "path": row[5],
                "status_code": row[6],
                "timestamp": row[7],
            }
            for row in cursor.fetchall()
        ]


class AuditLogMiddleware(BaseHTTPMiddleware):
    """審計日誌中間件"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """處理請求並記錄審計日誌

        Args:
            request: FastAPI 請求對象
            call_next: 下一個處理器

        Returns:
            Response: FastAPI 響應對象
        """
        # 獲取用戶信息
        user_info: Optional[Dict] = getattr(request.state, "user", None)
        user_id = user_info["user_id"] if user_info else "anonymous"
        role = user_info["role"] if user_info else "anonymous"

        # 處理請求
        response = await call_next(request)

        # 記錄審計日誌
        save_audit_log(
            user_id=user_id,
            role=role,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        return response
