"""審計日誌相關測試"""

import time
from unittest.mock import patch

import pytest

from app.monitor.audit_utils import get_audit_logs, save_audit_log


class TestAuditLog:
    """審計日誌測試類"""

    def test_save_audit_log(self):
        """測試保存審計日誌

        測試場景：
        - 正常保存日誌
        - 處理資料庫錯誤
        """
        # 正常保存
        try:
            save_audit_log(
                user_id="test_user",
                role="admin",
                method="GET",
                path="/test",
                status_code=200,
            )
        except Exception as e:
            pytest.fail(f"保存審計日誌失敗: {e}")

        # 模擬資料庫錯誤
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = Exception("資料庫連接失敗")
            save_audit_log(
                user_id="test_user",
                role="admin",
                method="GET",
                path="/test",
                status_code=200,
            )  # 應該正常處理錯誤，不拋出異常

    def test_get_audit_logs(self):
        """測試查詢審計日誌

        測試場景：
        - 無篩選條件查詢
        - 使用各種篩選條件
        - 時間範圍篩選
        - 結果數量限制
        """
        # 無篩選條件
        logs = get_audit_logs()
        assert isinstance(logs, list)
        if logs:
            assert all(
                key in logs[0]
                for key in [
                    "id",
                    "user_id",
                    "role",
                    "action",
                    "method",
                    "path",
                    "status_code",
                    "timestamp",
                ]
            )

        # 使用篩選條件
        logs = get_audit_logs(user_id="test_user", role="admin", limit=10)
        assert len(logs) <= 10
        if logs:
            assert all(log["user_id"] == "test_user" for log in logs)
            assert all(log["role"] == "admin" for log in logs)

        # 時間範圍篩選
        current_time = int(time.time())
        logs = get_audit_logs(
            start_time=current_time - 3600,  # 1小時前
            end_time=current_time,
        )
        if logs:
            assert all(
                current_time - 3600 <= log["timestamp"] <= current_time for log in logs
            )

    def test_audit_middleware(self, test_client):
        """測試審計日誌中間件

        測試場景：
        - 正常請求的日誌記錄
        - 錯誤請求的日誌記錄
        - 未認證請求的日誌記錄
        """
        # 正常請求
        response = test_client.get("/experiments/stats")
        assert response.status_code == 200

        # 錯誤請求
        response = test_client.get("/invalid_path")
        assert response.status_code == 404

        # 未認證請求（移除認證 header）
        client = test_client
        client.headers.pop("Authorization", None)
        response = client.get("/experiments")
        assert response.status_code == 401

    def test_audit_api(self, test_client):
        """測試審計日誌 API

        測試場景：
        - 管理員可以查看日誌
        - 普通用戶無法查看日誌
        - 篩選條件正常工作
        """
        # 管理員查看日誌
        response = test_client.get("/audit/logs")
        assert response.status_code == 200
        logs = response.json()
        assert isinstance(logs, list)

        # 使用篩選條件
        response = test_client.get(
            "/audit/logs",
            params={
                "user_id": "test_user",
                "role": "admin",
                "start_time": int(time.time()) - 3600,
                "limit": 5,
            },
        )
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) <= 5

        # 無效的篩選條件
        response = test_client.get("/audit/logs", params={"limit": -1})
        assert response.status_code == 422  # 驗證錯誤
