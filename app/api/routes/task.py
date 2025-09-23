"""
任務相關路由
"""

from typing import Dict

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.auth.jwt_utils import get_current_user

router = APIRouter(tags=["Tasks"])


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

    # 檢查用戶權限
    if user["role"] != "admin" and not task_id.startswith(user["user_id"]):
        raise HTTPException(status_code=403, detail="無權訪問此任務")

    try:
        task = AsyncResult(task_id)

        # 檢查任務後端是否可用
        if not hasattr(task, "backend") or task.backend is None:
            raise HTTPException(status_code=503, detail="任務後端不可用")

        try:
            task.backend.get_task_meta(task_id)
        except Exception as e:
            if "not found" in str(e).lower() or "no backend" in str(e).lower():
                raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
            raise

        if not task.ready():
            return {"status": task.status}

        if task.failed():
            error = task.result
            return {
                "status": "FAILURE",  # 确保返回 FAILURE 而不是 SUCCESS
                "error": str(error),
            }

        return {
            "status": task.status,
            "result": task.result,
        }
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"查詢任務狀態失敗 {task_id}: {e}")

        if not hasattr(task, "backend") or task.backend is None:
            raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="任務查詢超時")
        else:
            raise HTTPException(status_code=500, detail="內部服務器錯誤")
