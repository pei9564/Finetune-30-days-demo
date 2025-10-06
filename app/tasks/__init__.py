"""Celery 應用配置"""

import os

from celery import Celery
from celery.signals import worker_shutdown

from app.core.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# 創建 Celery 應用
celery_app = Celery(
    "lora_training",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# 配置 Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Taipei",
    enable_utc=True,
)


@worker_shutdown.connect
def cleanup_metrics_files(**_kwargs) -> None:
    """Remove multiprocess metric files for a worker that is shutting down."""

    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return

    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(os.getpid())
    except Exception:
        pass

# 導入任務以確保註冊
from .training import train_lora  # noqa
