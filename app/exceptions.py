"""
錯誤處理模組

定義所有自定義錯誤類型和錯誤處理邏輯
"""

from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class TrainingError(Exception):
    """訓練過程中的基礎錯誤類型"""

    pass


class OutOfMemoryError(TrainingError):
    """記憶體不足錯誤"""

    pass


class TrainingTimeoutError(TrainingError):
    """訓練超時錯誤

    與內建的 TimeoutError 區分，專門用於訓練過程的超時處理
    """

    pass


class DatasetError(TrainingError):
    """數據集相關錯誤"""

    pass


class ModelError(TrainingError):
    """模型相關錯誤"""

    pass


class ErrorResponse:
    """錯誤響應格式化器"""

    @staticmethod
    def create(error_type: str, message: str) -> Dict:
        """創建標準錯誤響應格式

        Args:
            error_type: 錯誤類型
            message: 錯誤訊息

        Returns:
            Dict: 格式化的錯誤響應
        """
        return {"error": error_type, "message": message}


def setup_error_handlers(app: FastAPI) -> None:
    """設置全域錯誤處理器

    Args:
        app: FastAPI 應用實例
    """

    @app.exception_handler(OutOfMemoryError)
    async def handle_oom_error(request: Request, exc: OutOfMemoryError):
        return JSONResponse(
            status_code=500, content=ErrorResponse.create("OUT_OF_MEMORY", str(exc))
        )

    @app.exception_handler(TrainingTimeoutError)
    async def handle_timeout_error(request: Request, exc: TrainingTimeoutError):
        return JSONResponse(
            status_code=408, content=ErrorResponse.create("TIMEOUT", str(exc))
        )

    @app.exception_handler(TrainingError)
    async def handle_training_error(request: Request, exc: TrainingError):
        return JSONResponse(
            status_code=500, content=ErrorResponse.create("TRAINING_ERROR", str(exc))
        )

    @app.exception_handler(Exception)
    async def handle_general_exception(request: Request, exc: Exception):
        error_type = exc.__class__.__name__
        return JSONResponse(
            status_code=500, content=ErrorResponse.create(error_type.upper(), str(exc))
        )
