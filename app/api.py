"""
FastAPI 應用
"""

from typing import Dict, Optional

from celery.result import AsyncResult
from fastapi import FastAPI
from pydantic import BaseModel

from .tasks.training import train_lora as train_lora_task

app = FastAPI(title="LoRA Training API")


class TrainingRequest(BaseModel):
    """訓練請求模型"""

    config_path: str = "config/default.yaml"
    experiment_name: Optional[str] = None
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    train_samples: Optional[int] = None
    device: Optional[str] = None


@app.post("/train")
async def start_training(request: TrainingRequest) -> Dict[str, str]:
    """提交訓練任務

    Args:
        request: 訓練請求

    Returns:
        Dict[str, str]: 包含任務 ID 的字典
    """
    # 提交任務
    task = train_lora_task.delay(
        config_path=request.config_path,
        experiment_name=request.experiment_name,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        train_samples=request.train_samples,
        device=request.device,
    )

    return {"task_id": task.id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> Dict:
    """查詢任務狀態

    Args:
        task_id: 任務 ID

    Returns:
        Dict: 包含任務狀態和結果的字典
    """
    task = AsyncResult(task_id)

    if not task.ready():
        return {"status": task.status}

    if task.failed():
        return {
            "status": task.status,
            "error": str(task.result),
        }

    return {
        "status": task.status,
        "result": task.result,
    }
