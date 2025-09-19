"""
任務相關路由
"""

from typing import Dict

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException

from app.auth.jwt_utils import check_task_owner

router = APIRouter(tags=["Tasks"])


@router.get("/task/{task_id}")
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

        # 檢查任務後端是否可用
        if not hasattr(task, "backend") or task.backend is None:
            raise HTTPException(status_code=503, detail="任務後端不可用")

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
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"查詢任務狀態失敗 {task_id}: {e}")

        if (
            "no backend" in str(e).lower()
            or not hasattr(task, "backend")
            or task.backend is None
        ):
            raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="任務查詢超時")
        else:
            raise HTTPException(status_code=500, detail="內部服務器錯誤")
