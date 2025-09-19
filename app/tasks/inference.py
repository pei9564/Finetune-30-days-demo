import logging
import os
import time
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(
    title="Inference API", description="Text Classification API with MPS Acceleration"
)

# 設定設備 (MPS 或 CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: int  # 1 = positive, 0 = negative
    probability: float  # prediction confidence
    latency_ms: float  # processing time (ms)
    base_model: str  # base model name
    language: str  # model language


def load_experiment_config(model_path):
    """載入實驗配置"""
    # 從 model_path (e.g., results/exp_name/artifacts/final_model)
    # 找到 config.yaml (在 results/exp_name/config.yaml)
    exp_dir = Path(model_path).parent.parent
    config_path = exp_dir / "config.yaml"

    if not config_path.exists():
        raise ValueError(f"找不到配置文件：{config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def init_model():
    """初始化模型"""
    # 1. 檢查環境變數
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("未設置 MODEL_PATH 環境變數")
    if not os.path.exists(model_path):
        raise ValueError(f"模型路徑不存在：{model_path}")

    # 2. 載入實驗配置
    config = load_experiment_config(model_path)
    base_model = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    logger.info(f"載入基礎模型：{base_model}")

    # 3. 載入基礎模型
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    ).to(DEVICE)

    # 4. 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 5. 載入 LoRA 權重
    logger.info(f"載入 LoRA 權重：{model_path}")
    model = PeftModel.from_pretrained(model, model_path, is_trainable=False)

    return model, tokenizer, base_model


# 全局初始化
try:
    logger.info(f"初始化模型，使用設備：{DEVICE}")
    MODEL, TOKENIZER, BASE_MODEL = init_model()
    logger.info("模型初始化完成")
except Exception as e:
    logger.error(f"模型初始化失敗：{str(e)}")
    raise


@app.get("/health")
async def health_check():
    """檢查服務健康狀態"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_path": os.getenv("MODEL_PATH"),
        "base_model": BASE_MODEL,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """執行預測"""
    try:
        # 1. 記錄請求
        logger.info(f"收到預測請求，文本長度：{len(request.text)}")

        # 2. 開始計時
        start_time = time.time()

        # 3. 準備輸入
        inputs = TOKENIZER(
            request.text, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        # 4. 執行預測
        with torch.no_grad():
            outputs = MODEL(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 5. 取得結果
        label = torch.argmax(probs).item()
        prob = probs[0][label].item()

        # 6. 計算延遲
        latency = (time.time() - start_time) * 1000

        # 7. 判斷語言
        is_chinese = "chinese" in BASE_MODEL.lower()
        lang = "Chinese" if is_chinese else "English"

        # 8. 準備回應
        response = PredictResponse(
            label=label,
            probability=prob,
            latency_ms=latency,
            base_model=BASE_MODEL,
            language=lang,
        )

        logger.info(f"預測完成，耗時：{latency:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"預測失敗：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
