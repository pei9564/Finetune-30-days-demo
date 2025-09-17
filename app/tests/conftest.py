"""
測試配置和共用 fixtures
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets import Dataset
from fastapi.testclient import TestClient

from app.api import app
from app.config import Config


@pytest.fixture
def test_client():
    """提供 FastAPI 測試客戶端"""
    return TestClient(app)


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
    monkeypatch.setattr("app.api.AsyncResult", mock_async_result)

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
