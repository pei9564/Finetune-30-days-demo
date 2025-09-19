"""
FastAPI 應用
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.api.routes.audit import router as audit_router
from app.auth.audit_log import AuditLogMiddleware, init_audit_table
from app.auth.jwt_utils import (
    check_admin,
    check_task_owner,
    create_token,
    get_current_user,
)
from app.db import Database, ExperimentFilter, ExperimentRecord
from app.exceptions import setup_error_handlers
from app.tasks.training import train_lora as train_lora_task

app = FastAPI(title="LoRA Training API")
setup_error_handlers(app)  # 設置全域錯誤處理

# 初始化審計日誌表
init_audit_table()

# 註冊審計日誌中間件
app.add_middleware(AuditLogMiddleware)

# 註冊審計日誌路由
app.include_router(audit_router)


class LoginRequest(BaseModel):
    """登入請求模型"""

    username: str
    password: str


class LoginResponse(BaseModel):
    """登入響應模型"""

    token: str
    user_id: str
    role: str


class TrainingConfig(BaseModel):
    """完整訓練配置模型"""

    experiment_name: str
    model: Dict[str, Any]  # 模型配置
    data: Dict[str, Any]  # 數據配置
    training: Dict[str, Any]  # 訓練配置
    lora: Dict[str, Any]  # LoRA 配置
    system: Dict[str, Any]  # 系統配置


class TrainingRequest(BaseModel):
    """訓練請求模型"""

    config: TrainingConfig


@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """用戶登入

    Args:
        request: 包含用戶名和密碼的請求

    Returns:
        LoginResponse: 包含 token 和用戶信息的響應

    Raises:
        HTTPException: 當認證失敗時
    """
    # 這裡應該實現真實的用戶認證邏輯
    # 目前僅作為示例：admin/admin 為管理員，其他為普通用戶
    if request.username == "admin" and request.password == "admin":
        role = "admin"
    else:
        role = "user"

    # 生成 token
    user_id = request.username  # 在實際應用中應該使用真實的用戶 ID
    token = create_token(user_id, role)

    return LoginResponse(token=token, user_id=user_id, role=role)


@app.post("/train")
async def start_training(
    request: TrainingRequest, user: Dict = Depends(get_current_user)
) -> Dict[str, str]:
    """提交訓練任務

    Args:
        request: 包含完整訓練配置的請求

    Returns:
        Dict[str, str]: 包含任務 ID 的字典
    """
    # 提交任務
    task = train_lora_task.delay(config=request.config.model_dump())

    return {"task_id": task.id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str, user: Dict = Depends(check_task_owner)) -> Dict:
    """查詢任務狀態

    Args:
        task_id: 任務 ID

    Returns:
        Dict: 包含任務狀態和結果的字典

    Raises:
        HTTPException: 當任務不存在時
    """

    try:
        task = AsyncResult(task_id)
        task_meta = task.backend.get_task_meta(task_id)
        if task_meta["status"] == "PENDING" and not task.backend.get_task_meta(task_id):
            raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")

        if not task.ready():
            return {"status": task.status}

        if task.failed():
            error = task.result
            return {
                "status": task.status,
                "error": str(error),
            }

        return {
            "status": task.status,
            "result": task.result,
        }
    except Exception as e:
        if "no backend" in str(e).lower() or not hasattr(task.backend, "get_task_meta"):
            raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        raise


@app.get("/experiments", response_model=List[ExperimentRecord])
async def list_experiments(
    user: Dict = Depends(check_admin),
    name: Optional[str] = None,
    min_accuracy: Optional[float] = Query(None, ge=0, le=1),
    max_runtime: Optional[float] = Query(None, gt=0),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sort_by: str = Query(
        "created_at", pattern="^(created_at|name|train_runtime|eval_accuracy)$"
    ),
    desc: bool = True,
    limit: int = Query(100, ge=1, le=1000),
) -> List[ExperimentRecord]:
    """列出實驗記錄

    Args:
        name: 實驗名稱（模糊匹配）
        min_accuracy: 最低準確率
        max_runtime: 最長訓練時間
        start_date: 開始日期
        end_date: 結束日期
        sort_by: 排序欄位
        desc: 是否降序排序
        limit: 返回數量限制

    Returns:
        List[ExperimentRecord]: 實驗記錄列表
    """
    db = Database()
    filter_params = ExperimentFilter(
        name=name,
        min_accuracy=min_accuracy,
        max_runtime=max_runtime,
        start_date=start_date,
        end_date=end_date,
    )
    return db.list_experiments(filter_params, sort_by, desc, limit)


@app.get("/experiments/stats")
async def get_experiment_stats(user: Dict = Depends(check_admin)) -> Dict:
    """獲取實驗統計資訊

    Returns:
        Dict: 統計資訊
    """
    db = Database()
    return db.get_statistics()


@app.get("/experiments/{experiment_id}", response_model=ExperimentRecord)
async def get_experiment(
    experiment_id: str, user: Dict = Depends(check_task_owner)
) -> ExperimentRecord:
    """查詢單一實驗記錄

    Args:
        experiment_id: 實驗 ID

    Returns:
        ExperimentRecord: 實驗記錄
    """
    db = Database()
    experiment = db.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="實驗不存在")
    return experiment
