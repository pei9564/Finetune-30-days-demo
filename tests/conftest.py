"""
測試配置和共用 fixtures
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

# Mock Celery before any app imports
mock_celery_app = MagicMock()
mock_celery_app.task = lambda *args, **kwargs: lambda func: func
sys.modules["app.tasks.celery_app"] = mock_celery_app

from app.core.config import Config  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """設置測試環境"""
    # 確保資料庫目錄存在
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Mock Celery settings
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "cache+memory://")

    # Mock 審計日誌相關操作
    def mock_noop(*args, **kwargs):
        return None

    def mock_get_audit_logs(*args, **kwargs):
        return [
            {
                "id": 1,
                "user_id": "test_user",
                "role": "admin",
                "action": "GET /test",
                "method": "GET",
                "path": "/test",
                "status_code": 200,
                "timestamp": int(time.time()),
            }
        ]

    monkeypatch.setattr("app.monitor.audit_utils.init_audit_table", mock_noop)
    monkeypatch.setattr("app.monitor.audit_utils.save_audit_log", mock_noop)
    monkeypatch.setattr("app.monitor.audit_utils.get_audit_logs", mock_get_audit_logs)

    # Mock 模型儲存相關操作
    monkeypatch.setattr("torch.save", mock_noop)
    monkeypatch.setattr("safetensors.torch.save_file", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_model", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_state", mock_noop)

    # 全局 Mock AsyncResult 以避免 DisabledBackend 问题
    def mock_async_result_global(task_id):
        mock_task = MagicMock()
        mock_task.id = task_id
        mock_task.status = "SUCCESS"
        mock_task.result = {
            "status": "success",
            "train": {"global_step": 100},
            "eval": {"accuracy": 0.85},
        }
        mock_task.ready.return_value = True
        mock_task.failed.return_value = False

        # 创建完整的 backend mock
        mock_backend = MagicMock()
        task_meta = {
            "status": "SUCCESS",
            "result": mock_task.result,
            "task_id": task_id,
        }
        mock_backend.get_task_meta.return_value = task_meta
        mock_backend._get_task_meta_for.return_value = task_meta
        mock_backend.as_tuple.return_value = (
            task_meta["status"],
            task_meta["result"],
            None,
        )
        mock_task.backend = mock_backend

        return mock_task

    # 在所有可能的地方 patch AsyncResult
    monkeypatch.setattr("celery.result.AsyncResult", mock_async_result_global)
    monkeypatch.setattr("app.api.routes.task.AsyncResult", mock_async_result_global)


@pytest.fixture
def mock_auth(monkeypatch):
    """Mock 認證相關的依賴

    替換 get_current_user, check_admin, check_task_owner 這些認證相關的依賴函數，
    使它們在測試中直接返回一個模擬的管理員用戶。
    """
    # Mock user data
    mock_user = {"user_id": "test_user", "role": "admin"}

    # Mock authentication functions
    mock_decode_token = MagicMock(return_value=mock_user)
    mock_check_admin = MagicMock(return_value=mock_user)
    mock_check_task_owner = MagicMock(return_value=mock_user)

    monkeypatch.setattr("app.auth.jwt_utils.decode_token", mock_decode_token)
    monkeypatch.setattr("app.auth.jwt_utils.check_admin", mock_check_admin)
    monkeypatch.setattr("app.auth.jwt_utils.check_task_owner", mock_check_task_owner)

    return mock_user


@pytest.fixture
def mock_token():
    """提供固定的測試 token"""
    return "test-token-123"


@pytest.fixture
def auth_headers(mock_token):
    """提供認證 headers"""
    return {"Authorization": f"Bearer {mock_token}"}


@pytest.fixture
def test_client(mock_auth, auth_headers):
    """提供 FastAPI 測試客戶端"""
    client = TestClient(app)
    client.headers.update(auth_headers)
    return client


@pytest.fixture
def mock_celery(monkeypatch):
    """Mock Celery 任務"""

    def create_task_result(task_id, status="SUCCESS", result=None, error=None):
        """创建一个任务结果对象"""
        task = MagicMock()
        task.id = task_id  # 確保這是字符串
        task.status = status
        task.result = error if error else result
        task.ready.return_value = status != "PENDING"
        task.failed.return_value = status == "FAILURE"

        # 设置后端
        task.backend = MagicMock()
        task_meta = {
            "status": status,
            "result": task.result,
            "task_id": task_id,
        }
        task.backend.get_task_meta.return_value = task_meta
        task.backend._get_task_meta_for.return_value = task_meta
        task.backend.as_tuple.return_value = (status, task.result, None)
        return task

    # 预定义的任务结果
    success_result = create_task_result(
        "test-task-123",
        status="SUCCESS",
        result={
            "status": "success",
            "train": {"global_step": 100},
            "eval": {"accuracy": 0.85},
        },
    )

    error_result = create_task_result(
        "error-task-123", status="FAILURE", error=ValueError("訓練數據集不能為空")
    )

    oom_result = create_task_result(
        "error-task-456",
        status="FAILURE",
        error=RuntimeError("GPU 記憶體不足: 已使用 15.0GB / 總計 16.0GB"),
    )

    pending_result = create_task_result("pending-task", status="PENDING")

    invalid_result = MagicMock()
    invalid_result.id = "invalid-task-id"
    invalid_result.backend = MagicMock()
    invalid_result.backend.get_task_meta.side_effect = Exception("Task not found")
    invalid_result.backend._get_task_meta_for.side_effect = Exception("Task not found")

    # 創建 mock train_lora 任務 - 確保 task.id 是字符串
    mock_train_lora = MagicMock()

    def mock_delay(*args, **kwargs):
        # 根據配置返回不同的任務結果
        if "config" in kwargs and isinstance(kwargs["config"], dict):
            exp_name = kwargs["config"].get("experiment_name", "").lower()
            if "error" in exp_name:
                return error_result
            elif "oom" in exp_name:
                return oom_result
        return success_result

    mock_train_lora.delay = MagicMock(side_effect=mock_delay)
    mock_train_lora.apply_async = MagicMock(side_effect=mock_delay)
    mock_train_lora.__name__ = "train_lora"

    # Mock AsyncResult 类
    def mock_async_result_class(task_id):
        if task_id == "invalid-task-id":
            return invalid_result
        elif task_id == "error-task-123":
            return error_result
        elif task_id == "error-task-456":
            return oom_result
        elif task_id == "pending-task":
            return pending_result
        else:
            return success_result

    # 設置所有的 mock - 使用 try/except 避免模組導入問題
    monkeypatch.setattr("app.api.routes.train.train_lora_task", mock_train_lora)
    monkeypatch.setattr("app.api.routes.task.AsyncResult", mock_async_result_class)
    monkeypatch.setattr("celery.result.AsyncResult", mock_async_result_class)

    # 嘗試 mock training 模組，如果失敗就跳過
    try:
        monkeypatch.setattr("app.tasks.training.train_lora", mock_train_lora)
    except (AttributeError, ImportError):
        pass

    return mock_train_lora


@pytest.fixture
def test_config():
    """提供測試用配置"""
    return Config(
        experiment_name="test_experiment",
        model={"name": "bert-base-uncased", "num_labels": 2},
        data={
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_samples": 20,
            "eval_samples": 5,
            "max_length": 128,
            "validation_rules": {
                "min_text_length": 5,
                "max_text_length": 500,
                "allow_empty": False,
                "remove_html": True,
            },
        },
        training={
            "output_dir": "results/test",
            "eval_strategy": "steps",
            "learning_rate": 1e-3,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 1,
            "logging_steps": 10,
        },
        lora={
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["query", "value"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "SEQ_CLS",
        },
        system={
            "experiment_name": "test_experiment",
            "save_config": True,
        },
    )


@pytest.fixture
def test_dataset():
    """提供測試用小型數據集"""
    data = {
        "sentence": [
            "This is a great movie!",
            "The film was terrible.",
            "I love this book so much!",
            "What a waste of time.",
            "Amazing performance by the actors!",
        ]
        * 4,  # 重複 4 次得到 20 筆數據
        "label": [1, 0, 1, 0, 1] * 4,  # 1=正面, 0=負面
    }
    return Dataset.from_pandas(pd.DataFrame(data))


@pytest.fixture
def empty_dataset():
    """提供空數據集"""
    return Dataset.from_pandas(pd.DataFrame({"sentence": [], "label": []}))


@pytest.fixture
def long_sequence_dataset():
    """提供超長序列數據集"""
    long_text = " ".join(["very"] * 1000) + " long text"  # 產生超長文本
    data = {"sentence": [long_text] * 5, "label": [1] * 5}
    return Dataset.from_pandas(pd.DataFrame(data))
