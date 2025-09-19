"""
實驗相關路由
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.jwt_utils import check_admin, check_task_owner
from app.db import Database, ExperimentFilter, ExperimentRecord

router = APIRouter(tags=["Experiments"])


@router.get("/experiments", response_model=List[ExperimentRecord])
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
    try:
        db = Database()
        filter_params = ExperimentFilter(
            name=name,
            min_accuracy=min_accuracy,
            max_runtime=max_runtime,
            start_date=start_date,
            end_date=end_date,
        )
        return db.list_experiments(filter_params, sort_by, desc, limit)
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"列出實驗記錄失敗: {e}")
        raise HTTPException(status_code=500, detail="內部服務器錯誤")


@router.get("/experiments/stats")
async def get_experiment_stats(user: Dict = Depends(check_admin)) -> Dict:
    """獲取實驗統計資訊，添加錯誤處理

    Returns:
        Dict: 統計資訊

    Raises:
        HTTPException: 當統計查詢失敗時
    """
    try:
        db = Database()
        return db.get_statistics()
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"獲取實驗統計失敗: {e}")
        raise HTTPException(status_code=500, detail="內部服務器錯誤")


@router.get("/experiments/{experiment_id}", response_model=ExperimentRecord)
async def get_experiment(
    experiment_id: str, user: Dict = Depends(check_task_owner)
) -> ExperimentRecord:
    """查詢單一實驗記錄，添加錯誤處理

    Args:
        experiment_id: 實驗 ID

    Returns:
        ExperimentRecord: 實驗記錄

    Raises:
        HTTPException: 當實驗不存在或資料損壞時
    """
    try:
        db = Database()
        experiment = db.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="實驗不存在")
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"查詢實驗失敗 {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="內部服務器錯誤")
