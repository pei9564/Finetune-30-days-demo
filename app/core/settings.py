import os

# Celery settings
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# API settings
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Redis URL for direct connections (e.g., if needed by db.py or other modules)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
