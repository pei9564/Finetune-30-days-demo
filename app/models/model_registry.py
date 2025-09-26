import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    loss: Optional[float] = None


class ModelCard(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Name of the model")
    base_model: str = Field(
        ..., description="Base model architecture (e.g., BERT, RoBERTa)"
    )
    language: str = Field(..., description="Primary language the model is trained for")
    task: str = Field(..., description="Primary task the model is designed for")
    description: Optional[str] = Field(
        None, description="Detailed description of the model"
    )
    metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(
        None, description="Vector representation of the model"
    )

    class Config:
        json_encoders = {np.ndarray: lambda x: x.tolist()}


class ModelRegistry:
    def __init__(self, storage_dir: str = "data/model_registry"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.models: Dict[str, ModelCard] = {}
        self._load_models()
        logger.info(f"初始化模型註冊表，存儲目錄: {storage_dir}")

    def _load_models(self):
        """Load all model cards from storage"""
        if not os.path.exists(self.storage_dir):
            logger.warning(f"存儲目錄不存在: {self.storage_dir}")
            return

        logger.info(f"從 {self.storage_dir} 載入模型卡片")

        for file in os.listdir(self.storage_dir):
            if file.endswith(".json"):
                file_path = os.path.join(self.storage_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        model_data = json.load(f)
                    model_card = ModelCard(**model_data)
                    self.models[model_card.id] = model_card
                except Exception as e:
                    logger.error(f"載入模型卡片失敗 {file_path}: {e}")

    def save_model_card(self, model_card: ModelCard) -> bool:
        """Save a model card to storage"""
        try:
            file_path = os.path.join(self.storage_dir, f"{model_card.id}.json")
            logger.info(f"保存模型卡片到: {file_path}")
            logger.info(f"模型卡片內容: {model_card.model_dump()}")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(model_card.model_dump(), f, indent=2, ensure_ascii=False)
            self.models[model_card.id] = model_card

            logger.info("✅ 模型卡片保存成功")
            return True
        except Exception as e:
            logger.error(f"❌ 保存模型卡片失敗: {e}")
            return False

    def search_models(
        self,
        base_model: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelCard]:
        """Search models based on criteria"""
        # 重新加載模型卡片
        self._load_models()
        logger.info(f"搜索模型，當前已載入 {len(self.models)} 個模型卡片")

        results = []
        for model in self.models.values():
            if base_model and model.base_model.lower() != base_model.lower():
                continue
            if language and model.language.lower() != language.lower():
                continue
            if task and model.task.lower() != task.lower():
                continue
            if tags and not all(tag in model.tags for tag in tags):
                continue
            results.append(model)

        logger.info(f"搜索結果: {len(results)} 個模型")
        return results

    def recommend_models(
        self, query_embedding: List[float], top_k: int = 5, task: Optional[str] = None
    ) -> List[ModelCard]:
        """Recommend models based on embedding similarity"""
        # Filter models that have embeddings and match the task if specified
        candidate_models = [
            model
            for model in self.models.values()
            if model.embedding is not None
            and (task is None or model.task.lower() == task.lower())
        ]

        if not candidate_models:
            return []

        # Convert embeddings to numpy array
        query_embedding = np.array(query_embedding).reshape(1, -1)
        model_embeddings = np.array([model.embedding for model in candidate_models])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, model_embeddings)[0]

        # Get top-k models
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [candidate_models[i] for i in top_indices]


# Create global registry instance
registry = ModelRegistry()
