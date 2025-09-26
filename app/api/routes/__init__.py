"""
API 路由模組
"""

from . import audit, auth, experiments, models, task, train
from .audit import router as audit_router
from .auth import router as auth_router
from .experiments import router as experiments_router
from .models import router as models_router
from .task import router as task_router
from .train import router as train_router

__all__ = [
    "auth",
    "audit",
    "experiments",
    "models",
    "task",
    "train",
    "auth_router",
    "audit_router",
    "experiments_router",
    "models_router",
    "task_router",
    "train_router",
]
