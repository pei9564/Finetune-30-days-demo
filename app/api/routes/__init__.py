"""
API 路由模組
"""

from . import audit, auth, experiments, task, train
from .audit import router as audit_router
from .auth import router as auth_router
from .experiments import router as experiments_router
from .task import router as task_router
from .train import router as train_router

__all__ = [
    "auth",
    "audit",
    "experiments",
    "task",
    "train",
    "auth_router",
    "audit_router",
    "experiments_router",
    "task_router",
    "train_router",
]
