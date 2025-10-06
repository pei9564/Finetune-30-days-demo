"""
FastAPI 應用
"""

from fastapi import FastAPI

from app.api.routes import (
    audit_router,
    auth_router,
    experiments_router,
    mlflow_router,
    models_router,
    task_router,
    train_router,
)
from app.exceptions import setup_error_handlers
from app.monitor.audit_utils import AuditLogMiddleware, init_audit_table
from app.monitor.exporter import router as metrics_router

app = FastAPI(title="LoRA Training API")
setup_error_handlers(app)  # 設置全域錯誤處理

# 初始化審計日誌表
init_audit_table()

# 註冊審計日誌中間件
app.add_middleware(AuditLogMiddleware)

# 註冊所有路由
app.include_router(auth_router)
app.include_router(train_router)
app.include_router(task_router)
app.include_router(experiments_router)
app.include_router(audit_router)
app.include_router(models_router)
app.include_router(mlflow_router)  # MLflow 端點（公開，不需要認證）
app.include_router(metrics_router)
