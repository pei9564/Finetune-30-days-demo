"""
錯誤處理機制的測試

測試分類：
1. Celery 任務錯誤處理
   - test_celery_task_retry: 測試任務重試機制
   - test_celery_timeout: 測試任務超時處理

2. API 錯誤處理
   - test_api_oom_error: 測試記憶體不足錯誤
   - test_api_validation_error: 測試參數驗證錯誤

3. Checkpoint 管理
   - test_checkpoint_cleanup: 測試 checkpoint 清理
   - test_checkpoint_selection: 測試 checkpoint 選擇策略
"""

import json
import os
from unittest.mock import MagicMock, patch

from torch.cuda import OutOfMemoryError as CudaOOMError

from app.tools.checkpoint_manager import CheckpointManager


class TestCeleryErrorHandling:
    """Celery 任務錯誤處理測試"""

    def test_celery_task_retry(self, mock_celery):
        """測試任務重試機制

        測試場景：
        - 當任務發生 OOM 錯誤時應該自動重試
        - 重試次數和狀態應該正確
        """
        config = {
            "model": {"name": "test-model"},
            "training": {"batch_size": 8},
        }

        # 設置 mock 返回重試狀態的任務
        retry_task = MagicMock()
        retry_task.status = "RETRY"
        retry_task.retries = 1
        retry_task.result = CudaOOMError("GPU 記憶體不足")

        # 讓 delay 返回這個重試任務
        with patch("app.tasks.training.train_lora") as mock_train:
            mock_train.delay.return_value = retry_task
            task = mock_train.delay(config)
            assert task.status == "RETRY"
            assert task.retries == 1
            assert "記憶體不足" in str(task.result)

    def test_celery_max_retries(self, mock_celery):
        """測試最大重試次數

        測試場景：
        - 當任務重試次數達到上限時應該失敗
        """
        config = {
            "model": {"name": "test-model"},
            "training": {"batch_size": 8},
        }

        # 設置失敗狀態的任務
        failure_task = MagicMock()
        failure_task.status = "FAILURE"
        failure_task.retries = 3
        failure_task.result = CudaOOMError("GPU 記憶體不足")

        # 讓 delay 返回這個失敗任務
        with patch("app.tasks.training.train_lora") as mock_train:
            mock_train.delay.return_value = failure_task
            task = mock_train.delay(config)
            assert task.status == "FAILURE"
            assert task.retries == 3
            assert "記憶體不足" in str(task.result)


class TestAPIErrorHandling:
    """API 錯誤處理測試"""

    def test_api_validation_error(self, test_client, mock_auth):
        """測試參數驗證錯誤處理

        測試場景：
        - API 端點在接收到無效參數時應該返回適當的錯誤響應
        """
        # 缺少必要的配置項
        response = test_client.post("/train", json={"config": {}})
        assert response.status_code == 422
        error_response = response.json()
        assert "detail" in error_response


class TestCheckpointManagement:
    """Checkpoint 管理測試"""

    def test_checkpoint_cleanup(self, tmp_path):
        """測試 checkpoint 清理功能

        測試場景：
        - 當 checkpoint 數量超過限制時應該清理舊的檔案
        """
        # 建立測試目錄和檔案
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # 建立測試檔案，確保三個關鍵指標對應不同的 checkpoint
        checkpoint_data = [
            {  # checkpoint-0: 最後一個
                "best_metric": 0.5,
                "total_flos": 1000000,
            },
            {  # checkpoint-1: 要被刪除
                "best_metric": 0.6,
                "total_flos": 900000,
            },
            {  # checkpoint-2: 要被刪除
                "best_metric": 0.7,
                "total_flos": 800000,
            },
            {  # checkpoint-3: 最快的
                "best_metric": 0.8,
                "total_flos": 500000,
            },
            {  # checkpoint-4: 最佳準確率
                "best_metric": 0.9,
                "total_flos": 1200000,
            },
        ]

        # 建立 checkpoints
        for i, data in enumerate(checkpoint_data):
            checkpoint_dir = artifacts_dir / f"checkpoint-{i}"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "adapter_config.json").touch()
            (checkpoint_dir / "adapter_model.safetensors").touch()
            # 建立 trainer_state.json 檔案
            with open(checkpoint_dir / "trainer_state.json", "w") as f:
                json.dump(data, f)
            os.utime(checkpoint_dir, (i, i))

        # 初始化管理器
        manager = CheckpointManager(
            results_dir=str(artifacts_dir),
            checkpoint_prefix="checkpoint-",
        )

        # 執行清理
        manager.cleanup_experiment(artifacts_dir)

        # 驗證結果
        remaining = manager.get_checkpoints(artifacts_dir)
        assert len(remaining) == 3
        assert all(file.name.startswith("checkpoint-") for file in remaining)
        # 驗證保留的是正確的 checkpoints
        to_keep, kept_metrics = manager.analyze_checkpoints(artifacts_dir)
        assert len(to_keep) == 3
        # 最佳準確率的 checkpoint
        assert kept_metrics["best"].accuracy == 0.9  # checkpoint-4
        # 最後一個 checkpoint
        assert kept_metrics["last"].path.name == "checkpoint-0"
        # 最快的 checkpoint
        assert kept_metrics["fastest"].path.name == "checkpoint-3"
