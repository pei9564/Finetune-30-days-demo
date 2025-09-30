import logging
from enum import Enum
from typing import List, Optional

import mlflow
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth.jwt_utils import check_admin
from app.models.model_registry import ModelCard, registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


class ModelStage(str, Enum):
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class SearchQuery(BaseModel):
    base_model: Optional[str] = None
    language: Optional[str] = None
    task: Optional[str] = None
    tags: Optional[List[str]] = None


class RecommendRequest(BaseModel):
    embedding: List[float]
    task: Optional[str] = None
    top_k: Optional[int] = 5


class RegistryModel(BaseModel):
    id: str
    name: str
    base_model: str
    version: int
    stage: str
    run_id: str


class TransitionRequest(BaseModel):
    model_id: str  # 例如: "task_default_experiment_20250927_211638"
    target_stage: ModelStage


# Model Search and Recommendation APIs
@router.get("/search", response_model=List[ModelCard])
async def search_models(
    base_model: Optional[str] = None,
    language: Optional[str] = None,
    task: Optional[str] = None,
    tags: Optional[str] = None,
):
    """
    Search for models based on specified criteria
    """
    try:
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",")]

        results = registry.search_models(
            base_model=base_model, language=language, task=task, tags=tags_list
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=List[ModelCard])
async def recommend_models(request: RecommendRequest):
    """
    Get model recommendations based on embedding similarity
    """
    try:
        results = registry.recommend_models(
            query_embedding=request.embedding, top_k=request.top_k, task=request.task
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model Registry Management APIs
@router.get("/", response_model=List[RegistryModel])
async def list_registry_models():
    """List all registered models with their versions and stages"""
    try:
        # Get all model cards
        models = registry.models.values()

        # 先重新加載所有模型卡片
        registry._load_models()
        models = registry.models.values()

        # 返回所有模型卡片
        registered_models = [
            RegistryModel(
                id=model.id,
                name=model.name,
                base_model=model.base_model,
                version=model.version or 0,  # 使用默認值而不是過濾
                stage=model.stage or "None",  # 使用默認值而不是過濾
                run_id=model.run_id or "",
            )
            for model in models
        ]

        # 按照 id 排序，確保顯示順序一致
        registered_models.sort(key=lambda x: x.id)

        return registered_models

    except Exception as e:
        logger.error(f"Failed to list registry models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transition")
async def transition_model_stage(request: TransitionRequest, _=Depends(check_admin)):
    """Transition a model version to a specific stage"""
    try:
        client = mlflow.tracking.MlflowClient()

        # 先找到對應的模型卡片
        model_card = registry.models.get(request.model_id)
        if not model_card:
            raise HTTPException(
                status_code=404,
                detail=f"Model card not found with ID: {request.model_id}",
            )

        # 使用模型卡片中的信息查找 MLflow 版本
        versions = client.search_model_versions(
            f"name='{model_card.base_model}' AND run_id='{model_card.run_id}'"
        )
        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"MLflow version not found for model {request.model_id}",
            )

        # If transitioning to Production, archive current Production version
        if request.target_stage == ModelStage.PRODUCTION:
            # 先獲取所有版本，然後過濾出 Production 版本
            all_versions = client.search_model_versions(
                f"name='{model_card.base_model}'"
            )
            prod_versions = [v for v in all_versions if v.current_stage == "Production"]

            # Archive current production version if exists
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_card.base_model,
                    version=version.version,
                    stage=ModelStage.ARCHIVED,
                )
                logger.info(
                    f"Archived model {model_card.base_model} version {version.version}"
                )

                # Update model cards - 使用 run_id 來確保是正確的訓練實例
                matching_cards = [
                    m
                    for m in registry.models.values()
                    if m.base_model == model_card.base_model
                    and m.run_id == version.run_id
                ]

                # 更新所有匹配的卡片
                for model_card in matching_cards:
                    model_card.stage = ModelStage.ARCHIVED
                    model_card.version = int(version.version)  # 確保版本號也正確
                    registry.save_model_card(model_card)
                    logger.info(
                        f"Archived model card {model_card.id} (run_id: {version.run_id})"
                    )

        # Transition target version to requested stage
        # 使用第一個找到的版本
        version_details = versions[0]
        client.transition_model_version_stage(
            name=model_card.base_model,
            version=version_details.version,
            stage=request.target_stage,
        )
        logger.info(
            f"Transitioned model {model_card.base_model} version {version_details.version} to {request.target_stage}"
        )

        # Get the MLflow model version details
        version_details = versions[0]  # We already checked versions is not empty
        run_id = version_details.run_id

        # Update model card - 使用 run_id 來確保是正確的訓練實例
        matching_cards = [
            m
            for m in registry.models.values()
            if m.base_model == model_card.base_model and m.run_id == run_id
        ]

        # 更新匹配的卡片
        for model_card in matching_cards:
            model_card.stage = request.target_stage
            model_card.version = int(version_details.version)  # 確保版本號也正確
            registry.save_model_card(model_card)
            logger.info(
                f"Updated model card {model_card.id} (run_id: {run_id}) to stage {request.target_stage}"
            )

        if not matching_cards:
            logger.warning(
                f"No matching model cards found for {request.model_name} run_id {run_id}"
            )

        return {"message": f"Model transitioned to {request.target_stage} successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to transition model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
