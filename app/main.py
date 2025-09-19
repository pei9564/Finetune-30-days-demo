"""
FastAPI 應用
"""

from fastapi import FastAPI

from app.api.routes import (
    audit_router,
    auth_router,
    experiments_router,
    task_router,
    train_router,
)
from app.exceptions import setup_error_handlers
from app.monitor.audit import AuditLogMiddleware, init_audit_table

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
