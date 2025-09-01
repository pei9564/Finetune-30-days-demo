import logging
from pathlib import Path


def setup_system_logger(name="lora_training", log_file="logs/local_training.log"):
    """設置系統級別的 logger，同時輸出到文件和終端

    Args:
        name (str): logger 的名稱
        log_file (str): 日誌文件路徑

    Returns:
        logging.Logger: 配置好的 logger
    """
    # 創建必要的目錄
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

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


def setup_progress_logger(log_file):
    """設置訓練進度的 logger，只輸出到文件

    Args:
        log_file (str): 日誌文件路徑

    Returns:
        logging.Logger: 配置好的 progress logger
    """
    # 創建 logger
    logger = logging.getLogger("training_progress")
    logger.setLevel(logging.INFO)

    # 清除現有的 handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 創建格式器（專門用於進度記錄）
    formatter = logging.Formatter("%(asctime)s - PROGRESS - %(message)s")

    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
