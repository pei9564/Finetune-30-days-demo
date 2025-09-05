"""
Celery 應用配置
"""

from celery import Celery
from settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

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

# 導入任務以確保註冊
from .training import train_lora  # noqa
