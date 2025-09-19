"""
測試配置和共用 fixtures
"""

import os
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

from app.config import Config
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """設置測試環境"""
    # 確保資料庫目錄存在
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

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

    monkeypatch.setattr("app.auth.audit_log.init_audit_table", mock_noop)
    monkeypatch.setattr("app.auth.audit_log.save_audit_log", mock_noop)
    monkeypatch.setattr("app.auth.audit_log.get_audit_logs", mock_get_audit_logs)

    # Mock 模型儲存相關操作
    monkeypatch.setattr("torch.save", mock_noop)
    monkeypatch.setattr("safetensors.torch.save_file", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_model", mock_noop)
    monkeypatch.setattr("transformers.Trainer.save_state", mock_noop)


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
    mock_task = MagicMock()
    mock_task.id = "test-task-123"
    mock_task.status = "SUCCESS"
    mock_task.retries = 0
    mock_task.result = {
        "status": "success",
        "train": {"global_step": 100},
        "eval": {"accuracy": 0.85},
    }
    mock_task.ready.return_value = True
    mock_task.failed.return_value = False

    # Mock backend
    mock_backend = MagicMock()
    mock_backend.get_task_meta.return_value = {
        "status": "SUCCESS",
        "result": mock_task.result,
    }
    mock_task.backend = mock_backend

    # Mock delay
    def mock_delay(*args, **kwargs):
        return mock_task

    # Mock AsyncResult
    mock_async_result = MagicMock()
    mock_async_result.return_value = mock_task

    # Patch Celery task
    monkeypatch.setattr("app.tasks.training.train_lora.delay", mock_delay)
    monkeypatch.setattr("app.main.AsyncResult", mock_async_result)

    return mock_task


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
