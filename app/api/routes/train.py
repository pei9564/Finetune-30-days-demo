"""
訓練相關路由
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth.jwt_utils import get_current_user
from app.tasks.training import create_training_job

router = APIRouter(tags=["Training"])


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


@router.post("/train")
async def start_training(
    request: TrainingRequest,
    http_request: Request,
    user: Dict = Depends(get_current_user),
) -> Dict[str, str]:
    """提交訓練任務，添加錯誤處理

    Args:
        request: 包含完整訓練配置的請求

    Returns:
        Dict[str, str]: 包含任務 ID 的字典

    Raises:
        HTTPException: 當任務提交失敗時
    """
    try:
        # 驗證配置
        if not request.config.experiment_name:
            raise HTTPException(status_code=400, detail="實驗名稱不能為空")

        # 設置用戶 ID 並提交任務
        config_dict = request.config.model_dump()
        config_dict["user_id"] = user["user_id"]
        task = create_training_job(config=config_dict, request=http_request)

        if not task or not hasattr(task, "id"):
            raise HTTPException(status_code=500, detail="任務提交失敗")

        return {"task_id": task.id}

    except HTTPException:
        raise
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"提交訓練任務失敗: {e}")
        raise HTTPException(status_code=500, detail="內部服務器錯誤")
