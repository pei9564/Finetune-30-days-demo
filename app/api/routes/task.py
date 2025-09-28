"""
任務相關路由
"""

import logging
from typing import Dict

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.auth.jwt_utils import get_current_user

router = APIRouter(tags=["Tasks"])
logger = logging.getLogger(__name__)


@router.get("/task/{task_id}")
async def get_task_status(task_id: str, user: Dict = Depends(get_current_user)) -> Dict:
    """查詢任務狀態

    Args:
        task_id: 任務 ID

    Returns:
        Dict: 包含任務狀態和結果的字典

    Raises:
        HTTPException: 當任務不存在時
    """

    try:
        # 檢查任務狀態
        task = AsyncResult(task_id)

        # 檢查任務後端是否可用
        if not hasattr(task, "backend") or task.backend is None:
            raise HTTPException(status_code=503, detail="任務後端不可用")

        # 檢查任務是否存在
        try:
            task.backend.get_task_meta(task_id)
        except Exception as e:
            if "not found" in str(e).lower() or "no backend" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
            raise

        # 檢查用戶權限
        if user["role"] != "admin":
            # 檢查任務所有者
            task_meta = task.backend.get_task_meta(task_id)

            # 嘗試從任務結果中獲取用戶 ID
            if task_meta.get("status") == "SUCCESS":
                result = task_meta.get("result", {})
                if isinstance(result, dict):
                    config = result.get("config", {})
                    task_user_id = config.get("user_id")
                    if task_user_id and task_user_id != user["user_id"]:
                        raise HTTPException(status_code=403, detail="無權訪問此任務")
            else:
                # 從任務參數中獲取配置
                task_kwargs = task_meta.get("kwargs", {})
                config = task_kwargs.get("config", {})
                task_user_id = config.get("user_id")
                if task_user_id and task_user_id != user["user_id"]:
                    raise HTTPException(status_code=403, detail="無權訪問此任務")

        # 返回任務狀態
        if not task.ready():
            return {"status": task.status}

        if task.failed():
            error = task.result
            return {
                "status": "FAILURE",
                "error": str(error),
            }

        return {
            "status": task.status,
            "result": task.result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查詢任務狀態失敗 {task_id}: {e}")
        if "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="任務查詢超時")
        else:
            raise HTTPException(status_code=500, detail="內部服務器錯誤")
