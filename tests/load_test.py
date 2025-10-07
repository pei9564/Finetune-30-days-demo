"""Locust load test for the finetune platform training endpoint."""

from __future__ import annotations

import os
import secrets
from copy import deepcopy
from typing import Dict, Any

from locust import HttpUser, task, between


DEFAULT_HOST = os.getenv("FINETUNE_API_HOST", "http://localhost:8000")
DEFAULT_USERNAME = os.getenv("FINETUNE_USERNAME", "load-test")
DEFAULT_PASSWORD = os.getenv("FINETUNE_PASSWORD", "load-test")
REQUEST_LIMIT = int(os.getenv("FINETUNE_LOADTEST_LIMIT", "10"))


BASE_TRAINING_CONFIG: Dict[str, Any] = {
    "experiment_name": "locust_experiment",
    "model": {"name": "bert-base-uncased", "num_labels": 2},
    "data": {
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
    "training": {
        "output_dir": "results/load_test",
        "eval_strategy": "steps",
        "learning_rate": 1e-3,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 1,
        "logging_steps": 10,
    },
    "lora": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["query", "value"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "SEQ_CLS",
    },
    "system": {
        "experiment_name": "locust_experiment",
        "save_config": True,
    },
}


def _build_training_payload() -> Dict[str, Any]:
    """Return a training payload with a unique experiment name."""

    config = deepcopy(BASE_TRAINING_CONFIG)
    suffix = secrets.token_hex(3)
    config["experiment_name"] = f"locust_experiment_{suffix}"
    config["system"]["experiment_name"] = config["experiment_name"]
    return {"config": config}


class TrainUser(HttpUser):
    """Simulates a user submitting training jobs to the worker API."""

    host = DEFAULT_HOST
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Authenticate once and reuse the bearer token for all tasks."""

        response = self.client.post(
            "/auth/login",
            json={"username": DEFAULT_USERNAME, "password": DEFAULT_PASSWORD},
        )
        response.raise_for_status()
        token = response.json()["token"]
        self.client.headers.update({"Authorization": f"Bearer {token}"})

    @task
    def submit_train_job(self) -> None:
        payload = _build_training_payload()
        response = self.client.post("/train", json=payload)
        response.raise_for_status()
        runner = getattr(self.environment, "runner", None)
        if runner:
            train_stats = runner.stats.get("/train", "POST")
            if train_stats and train_stats.num_requests >= REQUEST_LIMIT:
                runner.quit()
