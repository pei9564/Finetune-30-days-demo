import logging
import os


def setup_logger(name="lora_training", log_file="logs/local_training.log"):
    """設置基本的 logger，同時輸出到文件和終端"""
    # 創建必要的目錄
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 創建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除現有的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 創建格式器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 終端 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
