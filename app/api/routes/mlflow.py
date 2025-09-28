"""
MLflow experiment tracking API endpoints.
"""

import logging
from typing import Dict, Optional

import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from mlflow.exceptions import MlflowException

from app.core.mlflow_config import get_mlflow_ui_url, init_mlflow

logger = logging.getLogger(__name__)

# 創建一個可選的安全性檢查器
optional_auth = HTTPBearer(auto_error=False)

# MLflow 路由器（公開，不需要認證）
router = APIRouter(prefix="/mlflow", tags=["mlflow"])


@router.get("/{run_id}")
async def get_run_details(
    run_id: str, auth: Optional[HTTPAuthorizationCredentials] = Security(optional_auth)
) -> Dict:
    """
    Get MLflow run details including parameters, metrics, and artifacts.

    Args:
        run_id: MLflow run ID

    Returns:
        dict: Run details including parameters, metrics, and artifacts URI

    Raises:
        HTTPException: If run not found or MLflow error occurs
    """
    try:
        # Initialize MLflow and get experiment
        try:
            mlflow_config = init_mlflow()
            experiment_id = mlflow_config["experiment_id"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"無法初始化 MLflow: {str(e)}")

        # Get run details
        run = mlflow.get_run(run_id)

        # Generate MLflow UI URL
        mlflow_ui_url = get_mlflow_ui_url(run_id)

        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "mlflow_ui_url": mlflow_ui_url,
            "experiment_id": experiment_id,
        }

    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_runs(
    limit: Optional[int] = 10,
    status: Optional[str] = "FINISHED",
    auth: Optional[HTTPAuthorizationCredentials] = Security(optional_auth),
) -> Dict:
    """
    List recent MLflow runs.

    Args:
        limit: Maximum number of runs to return
        status: Filter runs by status (FINISHED, RUNNING, FAILED, SCHEDULED)

    Returns:
        dict: List of runs with basic information
    """
    try:
        # Initialize MLflow and get experiment
        try:
            mlflow_config = init_mlflow()
            experiment_id = mlflow_config["experiment_id"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"無法初始化 MLflow: {str(e)}")

        # 搜索實驗運行記錄
        try:
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"status = '{status}'" if status else None,
                max_results=limit,
                order_by=["start_time DESC"],
            )
        except MlflowException as e:
            logger.error(f"搜索 MLflow 運行記錄失敗: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"搜索 MLflow 運行記錄失敗: {str(e)}"
            )

        # 轉換 pandas DataFrame 為字典列表
        runs_list = []
        for _, run in runs.iterrows():
            try:
                # 安全地獲取基本信息
                run_id = run.get("run_id") if pd.notna(run.get("run_id")) else None
                if not run_id:
                    continue

                # 處理時間戳
                start_time = None
                end_time = None
                if pd.notna(run.get("start_time")):
                    start_time = str(run["start_time"])
                if pd.notna(run.get("end_time")):
                    end_time = str(run["end_time"])

                # 處理指標
                metrics = {
                    "eval_accuracy": float(run["metrics.eval_accuracy"])
                    if pd.notna(run.get("metrics.eval_accuracy"))
                    else None,
                    "training_time": float(run["metrics.training_time"])
                    if pd.notna(run.get("metrics.training_time"))
                    else None,
                }

                # 處理參數
                params = {
                    "model_name": str(run["params.model_name"])
                    if pd.notna(run.get("params.model_name"))
                    else None,
                    "batch_size": str(run["params.batch_size"])
                    if pd.notna(run.get("params.batch_size"))
                    else None,
                    "learning_rate": str(run["params.learning_rate"])
                    if pd.notna(run.get("params.learning_rate"))
                    else None,
                }

                # 生成 MLflow UI URL
                mlflow_ui_url = get_mlflow_ui_url(run_id)

                # 構建運行記錄字典
                run_dict = {
                    "run_id": run_id,
                    "status": str(run["status"])
                    if pd.notna(run.get("status"))
                    else None,
                    "start_time": start_time,
                    "end_time": end_time,
                    "metrics": metrics,
                    "params": params,
                    "mlflow_ui_url": mlflow_ui_url,
                }

                # 過濾掉 None 值
                run_dict = {k: v for k, v in run_dict.items() if v is not None}
                runs_list.append(run_dict)
            except Exception as e:
                logger.error(f"處理運行記錄時出錯: {str(e)}")
                continue

        # 構建返回結果
        result = {
            "total": len(runs_list),
            "runs": runs_list,
            "experiment_id": experiment_id,
        }

        # 添加調試信息
        logger.debug(f"返回 {len(runs_list)} 條運行記錄")
        return result

    except MlflowException as e:
        logger.error(f"MLflow 操作失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MLflow 操作失敗: {str(e)}")
    except Exception as e:
        logger.error(f"未預期的錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"未預期的錯誤: {str(e)}")
