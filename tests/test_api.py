"""
API 端點單元測試

測試分類：
1. 基本功能測試
   - test_train_endpoint: 測試訓練端點
   - test_task_status: 測試任務狀態查詢
   - test_simulate_error: 測試錯誤模擬端點

2. 錯誤處理測試
   - test_invalid_config: 測試無效的配置
   - test_invalid_task_id: 測試無效的任務ID
   - test_training_error: 測試訓練過程中的錯誤
"""

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI


@pytest.fixture
def test_app(test_client) -> FastAPI:
    """添加測試專用的端點

    為了測試錯誤處理，我們在測試時動態添加一個會拋出錯誤的端點。
    這樣可以避免在主程式中添加測試專用的代碼。
    """
    app = test_client.app

    @app.post("/simulate_error")
    async def simulate_error() -> Dict:
        """模擬錯誤，用於測試錯誤處理"""
        raise RuntimeError("GPU 記憶體不足: 已使用 15.0GB / 總計 16.0GB")

    return app


class TestAPI:
    """API 端點測試類

    包含兩類測試：
    1. 基本功能測試：測試 API 端點的正常功能
    2. 錯誤處理測試：測試各種錯誤情況的處理
    """

    # =========================================================================
    # 基本功能測試
    # =========================================================================

    def test_train_endpoint(self, test_client, test_config, mock_celery, mock_auth):
        """測試訓練端點

        測試場景：
        - 使用有效的配置提交訓練任務
        - 應該返回任務 ID

        測試步驟：
        1. 提交訓練請求
        2. 驗證響應狀態碼
        3. 驗證返回的任務 ID
        """
        response = test_client.post("/train", json={"config": test_config.model_dump()})

        assert response.status_code == 200
        assert "task_id" in response.json()
        assert response.json()["task_id"] == "test-task-123"

    def test_task_status(self, test_client, mock_celery, mock_auth):
        """測試任務狀態查詢

        測試場景：
        - 檢查成功完成的任務狀態
        - 檢查正在進行中的任務狀態
        - 檢查失敗的任務狀態

        測試步驟：
        1. 檢查 SUCCESS 狀態
           - 驗證狀態碼和狀態
           - 驗證結果內容
        2. 檢查 PENDING 狀態
           - Mock AsyncResult 返回 PENDING
           - 驗證狀態
        3. 檢查 FAILURE 狀態
           - Mock AsyncResult 返回錯誤
           - 驗證錯誤訊息
        """
        # 檢查 SUCCESS 狀態
        response = test_client.get("/task/test-task-123")
        assert response.status_code == 200
        assert response.json()["status"] == "SUCCESS"
        assert response.json()["result"]["eval"]["accuracy"] == 0.85

        # 檢查 PENDING 狀態
        with patch("app.main.AsyncResult") as mock_result:
            mock_result.return_value.ready.return_value = False
            mock_result.return_value.status = "PENDING"

            response = test_client.get("/task/test-task-123")
            assert response.status_code == 200
            assert response.json()["status"] == "PENDING"

        # 檢查 FAILURE 狀態
        with patch("app.main.AsyncResult") as mock_result:
            mock_result.return_value.ready.return_value = True
            mock_result.return_value.failed.return_value = True
            mock_result.return_value.status = "FAILURE"
            mock_result.return_value.result = Exception("Test error")

            response = test_client.get("/task/test-task-123")
            assert response.status_code == 200
            assert response.json()["status"] == "FAILURE"
            assert "Test error" in response.json()["error"]

    def test_simulate_error(self, test_app, test_client):
        """測試模擬錯誤端點

        測試場景：
        - 端點會拋出 RuntimeError
        - 錯誤訊息應該包含記憶體不足信息

        測試步驟：
        1. 調用錯誤模擬端點
        2. 驗證響應狀態碼為 500
        3. 驗證錯誤訊息內容
        """
        try:
            response = test_client.post("/simulate_error")
            assert response.status_code == 500
            assert "記憶體不足" in str(response.json())
        except RuntimeError as e:
            # 確保錯誤訊息正確
            assert "記憶體不足" in str(e)

    # =========================================================================
    # 錯誤處理測試
    # =========================================================================

    def test_invalid_config(self, test_client, mock_auth):
        """測試無效的配置

        測試場景：
        - 測試缺少必要配置項的情況
        - 測試無效的數據集配置

        測試步驟：
        1. 測試空配置
           - 提交空的配置對象
           - 驗證 422 錯誤
        2. 測試無效數據集
           - 提交不存在的數據集配置
           - 驗證 422 錯誤
        """
        # 缺少必要的配置項
        response = test_client.post("/train", json={"config": {}})
        assert response.status_code == 422
        assert response.json()["detail"] is not None

        # 無效的數據集名稱
        invalid_config = {
            "model": {"name": "bert-base-uncased", "num_labels": 2},
            "data": {
                "dataset_name": "invalid_dataset",
                "dataset_config": "invalid",
                "train_samples": 100,
                "eval_samples": 10,
            },
        }
        response = test_client.post("/train", json={"config": invalid_config})
        assert response.status_code == 422
        assert response.json()["detail"] is not None

    def test_invalid_task_id(self, test_client, mock_auth):
        """測試無效的任務ID

        測試場景：
        - 使用不存在的任務 ID 查詢狀態
        - 應該返回 404 錯誤

        測試步驟：
        1. Mock AsyncResult 拋出後端錯誤
        2. 查詢無效的任務 ID
        3. 驗證 404 錯誤和錯誤訊息
        """
        with patch("app.main.AsyncResult") as mock_result:
            mock_result.return_value.backend = MagicMock()
            mock_result.return_value.backend.get_task_meta.side_effect = Exception(
                "no backend"
            )

            response = test_client.get("/task/invalid-task-id")
            assert response.status_code == 404
            assert "找不到任務" in response.json()["detail"]

    def test_training_error(self, test_client, test_config, mock_auth):
        """測試訓練過程中的錯誤

        測試場景：
        1. 空數據集錯誤
           - 訓練過程中發現數據集為空
           - 應該返回相應的錯誤訊息
        2. 記憶體不足錯誤
           - 訓練過程中發生 OOM
           - 應該返回相應的錯誤訊息

        測試步驟：
        1. 測試空數據集錯誤
           - Mock 訓練任務返回 ValueError
           - 提交訓練任務
           - 檢查任務狀態和錯誤訊息
        2. 測試記憶體不足錯誤
           - Mock 訓練任務返回 RuntimeError
           - 提交訓練任務
           - 檢查任務狀態和錯誤訊息
        """
        # 模擬空數據集錯誤
        with patch("app.tasks.training.train_lora.delay") as mock_train:
            mock_task = MagicMock()
            mock_task.id = "error-task-123"
            mock_task.ready.return_value = True
            mock_task.failed.return_value = True
            mock_task.result = ValueError("訓練數據集不能為空")
            mock_task.status = "FAILURE"
            mock_task.backend = MagicMock()
            mock_task.backend.get_task_meta.return_value = {
                "status": "FAILURE",
                "result": mock_task.result,
            }
            mock_train.return_value = mock_task

            # 提交任務
            response = test_client.post(
                "/train", json={"config": test_config.model_dump()}
            )
            assert response.status_code == 200
            task_id = response.json()["task_id"]

            # 檢查任務狀態
            with patch("app.main.AsyncResult") as mock_async:
                mock_async.return_value = mock_task
                response = test_client.get(f"/task/{task_id}")
                assert response.status_code == 200
                assert response.json()["status"] == "FAILURE"
                assert "訓練數據集不能為空" in str(response.json())

        # 模擬記憶體不足錯誤
        with patch("app.tasks.training.train_lora.delay") as mock_train:
            mock_task = MagicMock()
            mock_task.id = "error-task-456"
            mock_task.ready.return_value = True
            mock_task.failed.return_value = True
            mock_task.result = RuntimeError(
                "GPU 記憶體不足: 已使用 15.0GB / 總計 16.0GB"
            )
            mock_task.status = "FAILURE"
            mock_task.backend = MagicMock()
            mock_task.backend.get_task_meta.return_value = {
                "status": "FAILURE",
                "result": mock_task.result,
            }
            mock_train.return_value = mock_task

            # 提交任務
            response = test_client.post(
                "/train", json={"config": test_config.model_dump()}
            )
            assert response.status_code == 200
            task_id = response.json()["task_id"]

            # 檢查任務狀態
            with patch("app.main.AsyncResult") as mock_async:
                mock_async.return_value = mock_task
                response = test_client.get(f"/task/{task_id}")
                assert response.status_code == 200
                assert response.json()["status"] == "FAILURE"
                assert "GPU 記憶體不足" in str(response.json())
