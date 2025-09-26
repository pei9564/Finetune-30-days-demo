from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.model_registry import ModelCard, registry

router = APIRouter(prefix="/models", tags=["models"])


class SearchQuery(BaseModel):
    base_model: Optional[str] = None
    language: Optional[str] = None
    task: Optional[str] = None
    tags: Optional[List[str]] = None


class RecommendRequest(BaseModel):
    embedding: List[float]
    task: Optional[str] = None
    top_k: Optional[int] = 5


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
