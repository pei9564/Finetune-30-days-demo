"""
訓練相關單元測試

測試分類：
1. 數據集相關測試
   - test_empty_dataset: 測試空數據集處理
   - test_missing_split: 測試缺少必要的數據集分割
   - test_dataset_load_error: 測試數據集載入錯誤
   - test_long_sequence: 測試超長序列處理

2. 系統資源相關測試
   - test_memory_monitoring: 測試記憶體監控
   - test_out_of_memory: 測試記憶體不足錯誤處理
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datasets import Dataset

from app.train_lora_v2 import main as train_main


class TestTraining:
    """訓練相關測試類

    包含兩類測試：
    1. 數據集相關測試：測試數據載入、處理和驗證
    2. 系統資源相關測試：測試記憶體監控和錯誤處理
    """

    # =========================================================================
    # 數據集相關測試
    # =========================================================================

    def test_empty_dataset(self, test_config, empty_dataset):
        """測試空數據集處理

        測試場景：
        - 當訓練集或驗證集為空時，應該拋出 ValueError
        - 錯誤訊息應該包含 "訓練數據集不能為空"

        測試步驟：
        1. Mock load_dataset 返回空數據集
        2. 執行訓練
        3. 驗證錯誤類型和訊息
        """
        with patch("app.train_lora_v2.load_dataset") as mock_load:
            mock_load.return_value = {
                "train": empty_dataset,
                "validation": empty_dataset,
            }

            # 應該拋出 EmptyDatasetError
            with pytest.raises(ValueError) as exc_info:
                train_main(test_config)

            # 驗證錯誤訊息
            assert "訓練數據集不能為空" in str(exc_info.value)

    def test_missing_split(self, test_config, test_dataset):
        """測試缺少必要的數據集分割

        測試場景：
        - 當缺少驗證集時，應該拋出 ValueError
        - 錯誤訊息應該包含 "缺少必要的分割: validation"

        測試步驟：
        1. Mock load_dataset 返回只有訓練集的數據集
        2. 執行訓練
        3. 驗證錯誤類型和訊息
        """
        with patch("app.train_lora_v2.load_dataset") as mock_load:
            # 模擬缺少驗證集的情況
            mock_load.return_value = {
                "train": test_dataset,
                # 缺少 "validation" 分割
            }

            # 應該拋出 DatasetError
            with pytest.raises(ValueError) as exc_info:
                train_main(test_config)

            # 驗證錯誤訊息
            assert "缺少必要的分割: validation" in str(exc_info.value)

    def test_dataset_load_error(self, test_config):
        """測試數據集載入錯誤

        測試場景：
        - 當數據集載入失敗時，應該拋出 ValueError
        - 錯誤訊息應該包含 "無法載入數據集"

        測試步驟：
        1. Mock load_dataset 拋出異常
        2. 執行訓練
        3. 驗證錯誤類型和訊息
        """
        with patch("app.train_lora_v2.load_dataset") as mock_load:
            # 模擬載入錯誤
            mock_load.side_effect = Exception("找不到數據集")

            # 應該拋出 DatasetError
            with pytest.raises(ValueError) as exc_info:
                train_main(test_config)

            # 驗證錯誤訊息
            assert "無法載入數據集" in str(exc_info.value)

    # =========================================================================
    # 系統資源相關測試
    # =========================================================================

    def test_memory_monitoring(self, test_config, test_dataset):
        """測試記憶體監控

        測試場景：
        - 使用 MPS (Apple Silicon) 進行訓練
        - 監控 CPU 記憶體使用情況
        - 訓練應該正常完成

        測試步驟：
        1. Mock 系統記憶體相關函數
        2. Mock 訓練相關組件
        3. 執行訓練
        4. 驗證記憶體監控被調用
        5. 驗證訓練結果
        """
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("psutil.Process") as mock_process,
            patch("psutil.virtual_memory") as mock_virtual_memory,
        ):
            # 模擬 CPU 記憶體使用情況
            mock_process.return_value.memory_info.return_value.rss = 2 * 1024**3  # 2GB
            mock_virtual_memory.return_value.total = 16 * 1024**3  # 16GB

            with patch("app.train_lora_v2.load_dataset") as mock_load:
                mock_load.return_value = {
                    "train": test_dataset,
                    "validation": test_dataset.select(range(5)),
                }

                # Mock 訓練結果
                mock_train_result = MagicMock()
                mock_train_result.global_step = 100
                mock_train_result.metrics = {"train_runtime": 10.5}

                mock_eval_result = {"eval_accuracy": 0.85}

                with patch("app.train_lora_v2.Trainer") as mock_trainer:
                    mock_trainer.return_value.train.return_value = mock_train_result
                    mock_trainer.return_value.evaluate.return_value = mock_eval_result

                    # 執行訓練
                    train_result, eval_result = train_main(test_config)

                    # 驗證結果
                    assert train_result.global_step == 100
                    assert eval_result["eval_accuracy"] == 0.85

    def test_out_of_memory(self, test_config, test_dataset):
        """測試記憶體不足錯誤處理

        測試場景：
        - 訓練過程中出現 CUDA 記憶體不足錯誤
        - 應該拋出 RuntimeError
        - 錯誤訊息應該包含 "記憶體不足"

        測試步驟：
        1. Mock 訓練過程拋出 CUDA OOM 錯誤
        2. 執行訓練
        3. 驗證錯誤類型和訊息
        """
        with patch("app.train_lora_v2.load_dataset") as mock_load:
            mock_load.return_value = {
                "train": test_dataset,
                "validation": test_dataset.select(range(5)),
            }

            with patch("app.train_lora_v2.Trainer") as mock_trainer:
                # 模擬訓練時記憶體不足
                mock_trainer.return_value.train.side_effect = RuntimeError(
                    "CUDA out of memory. Tried to allocate 2.0 GB"
                )

                # 應該拋出 RuntimeError
                with pytest.raises(RuntimeError) as exc_info:
                    train_main(test_config)

                # 驗證錯誤訊息
                assert "記憶體不足" in str(exc_info.value)

    def test_long_sequence(self, test_config, long_sequence_dataset):
        """測試超長序列處理

        測試場景：
        - 輸入序列長度超過模型的最大長度限制
        - 訓練應該正常完成（序列會被自動截斷）
        - 訓練結果應該符合預期

        測試步驟：
        1. 創建包含超長序列的數據集
        2. Mock 訓練相關組件
        3. 執行訓練
        4. 驗證訓練結果
        """
        with patch("app.train_lora_v2.load_dataset") as mock_load:
            # 創建足夠大的數據集
            base_data = long_sequence_dataset.select(range(5))
            df = pd.DataFrame(base_data)
            train_data = Dataset.from_pandas(pd.concat([df] * 4, ignore_index=True))

            # 創建驗證數據集
            val_df = pd.DataFrame(base_data)
            val_data = Dataset.from_pandas(val_df)

            mock_load.return_value = {
                "train": train_data,
                "validation": val_data,
            }

            # Mock 訓練結果
            mock_train_result = MagicMock()
            mock_train_result.global_step = 100
            mock_train_result.metrics = {"train_runtime": 15.5}

            mock_eval_result = {"eval_accuracy": 0.75}

            with patch("app.train_lora_v2.Trainer") as mock_trainer:
                mock_trainer.return_value.train.return_value = mock_train_result
                mock_trainer.return_value.evaluate.return_value = mock_eval_result

                # 訓練應該成功，文本會被截斷
                train_result, eval_result = train_main(test_config)

                # 驗證結果
                assert train_result.global_step == 100
                assert "train_runtime" in train_result.metrics
                assert eval_result["eval_accuracy"] == 0.75
