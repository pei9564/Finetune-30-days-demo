.PHONY: setup-conda run-local logs-local data-analyze data-validate data-versions help

# 本地 Conda 環境設置（自動檢測晶片類型）
setup-conda:
	@echo "🔍 檢測系統環境..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先安裝 miniforge：brew install --cask miniforge"; \
		exit 1; \
	fi
	@bash -c '\
		if command -v nvidia-smi &> /dev/null; then \
			echo "🚀 檢測到 NVIDIA GPU 環境"; \
			ENV_NAME="lora-gpu"; \
		elif uname -m | grep -q "arm64"; then \
			echo "🍎 檢測到 Apple Silicon (ARM64) 環境"; \
			ENV_NAME="lora-m3"; \
		else \
			echo "💻 檢測到 x86_64 CPU 環境"; \
			ENV_NAME="lora-cpu"; \
		fi; \
		echo "📦 使用環境名稱: $$ENV_NAME"; \
		if conda env list | grep -q "$$ENV_NAME"; \
		then \
			echo "✅ Conda 環境 \"$$ENV_NAME\" 已存在"; \
		else \
			echo "📦 創建新的 Conda 環境 \"$$ENV_NAME\"..."; \
			conda create --name $$ENV_NAME python=3.11 -y; \
			echo "✅ 環境創建完成！"; \
		fi; \
		echo "📦 安裝依賴..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && pip install --upgrade pip && pip install -r requirements.txt; \
		echo "✅ 依賴安裝完成！"; \
		echo "📋 下一步：make run-local" \
	'

# 本地運行訓練（使用預設配置）
run-local:
	@echo "🚀 檢查並運行 LoRA 訓練（使用預設配置）..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先運行 'make setup-conda'"; \
		exit 1; \
	fi
	@bash -c '\
		if command -v nvidia-smi &> /dev/null; then \
			ENV_NAME="lora-gpu"; \
		elif uname -m | grep -q "arm64"; then \
			ENV_NAME="lora-m3"; \
		else \
			ENV_NAME="lora-cpu"; \
		fi; \
		if ! conda env list | grep -q "$$ENV_NAME"; then \
			echo "❌ Conda 環境 \"$$ENV_NAME\" 不存在，請先運行 \"make setup-conda\""; \
			exit 1; \
		fi; \
		echo "🚀 使用環境 \"$$ENV_NAME\" 開始訓練..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && python -u app/train_lora_v2.py \
	'



# 查看本地訓練 log（優先顯示 v2 的進度日誌）
logs-local:
	@if [ -f logs/training_progress.log ]; then \
		echo "📋 查看訓練進度（最後 20 行）..."; \
		tail -n 20 logs/training_progress.log; \
		echo ""; \
		echo "💡 提示："; \
		echo "  - 使用 'tail -f logs/training_progress.log' 來持續監控 log"; \
	else \
		echo "❌ 沒有找到訓練 log 文件，請先運行 'make run-local'"; \
	fi

# 分析資料集分布 (僅用於測試範例)
data-analyze:
	@echo "📊 分析資料集分布..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先運行 'make setup-conda'"; \
		exit 1; \
	fi
	@bash -c '\
		if command -v nvidia-smi &> /dev/null; then \
			ENV_NAME="lora-gpu"; \
		elif uname -m | grep -q "arm64"; then \
			ENV_NAME="lora-m3"; \
		else \
			ENV_NAME="lora-cpu"; \
		fi; \
		if ! conda env list | grep -q "$$ENV_NAME"; then \
			echo "❌ Conda 環境 \"$$ENV_NAME\" 不存在，請先運行 \"make setup-conda\""; \
			exit 1; \
		fi; \
		echo "📊 使用環境 \"$$ENV_NAME\" 分析資料..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.dataset_analyzer \
	'

# 驗證資料集品質 (僅用於測試範例)
data-validate:
	@echo "🔍 驗證資料集品質..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先運行 'make setup-conda'"; \
		exit 1; \
	fi
	@bash -c '\
		if command -v nvidia-smi &> /dev/null; then \
			ENV_NAME="lora-gpu"; \
		elif uname -m | grep -q "arm64"; then \
			ENV_NAME="lora-m3"; \
		else \
			ENV_NAME="lora-cpu"; \
		fi; \
		if ! conda env list | grep -q "$$ENV_NAME"; then \
			echo "❌ Conda 環境 \"$$ENV_NAME\" 不存在，請先運行 \"make setup-conda\""; \
			exit 1; \
		fi; \
		echo "🔍 使用環境 \"$$ENV_NAME\" 驗證資料..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.data_validator \
	'

# 管理資料版本 (僅用於測試範例)
data-versions:
	@echo "📦 管理資料版本..."
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先運行 'make setup-conda'"; \
		exit 1; \
	fi
	@bash -c '\
		if command -v nvidia-smi &> /dev/null; then \
			ENV_NAME="lora-gpu"; \
		elif uname -m | grep -q "arm64"; then \
			ENV_NAME="lora-m3"; \
		else \
			ENV_NAME="lora-cpu"; \
		fi; \
		if ! conda env list | grep -q "$$ENV_NAME"; then \
			echo "❌ Conda 環境 \"$$ENV_NAME\" 不存在，請先運行 \"make setup-conda\""; \
			exit 1; \
		fi; \
		echo "📦 使用環境 \"$$ENV_NAME\" 管理版本..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.version_manager \
	'

# 顯示幫助
help:
	@echo "🍎 LoRA 訓練環境管理命令"
	@echo ""
	@echo "🚀 基本使用流程："
	@echo "  1. make setup-conda   - 首次使用：檢查並創建 Conda 環境"
	@echo "  2. make run-local     - 執行訓練（使用預設配置）"
	@echo "  3. make logs-local    - 查看訓練進度"
	@echo ""
	@echo "⚙️ 配置說明："
	@echo "  1. 預設配置文件：config/default.yaml"
	@echo "     - 包含所有可調整的參數與預設值"
	@echo "     - 直接修改此文件來更改預設配置"
	@echo ""
	@echo "  2. 命令列參數（優先於預設配置）："
	@echo "     python app/train_lora_v2.py [參數]"
	@echo ""
	@echo "     常用參數："
	@echo "     --experiment_name TEXT    實驗名稱"
	@echo "     --learning_rate FLOAT     學習率"
	@echo "     --epochs INT              訓練輪數"
	@echo "     --train_samples INT       訓練樣本數"
	@echo "     --device TEXT             指定設備 (cuda/mps/cpu)"
	@echo ""
	@echo "     完整範例："
	@echo "     python app/train_lora_v2.py \\"
	@echo "       --experiment_name \"custom_test\" \\"
	@echo "       --learning_rate 0.001 \\"
	@echo "       --epochs 3 \\"
	@echo "       --train_samples 1000"
	@echo ""
	@echo "📊 監控與記錄："
	@echo "  1. 即時監控："
	@echo "     - 使用 'tail -f logs/training_progress.log'"
	@echo "     - 或執行 'make logs-local' 查看最後 20 行"
	@echo ""
	@echo "  2. 實驗記錄："
	@echo "     - 配置記錄：results/configs/{實驗名稱}_{準確率}_{時間戳}.yaml"
	@echo "     - 訓練日誌：logs/training_progress.log"
	@echo "     - 模型保存：results/final_model/"
	@echo ""
	@echo "🔧 資料管理工具（僅供開發測試用）："
	@echo "  註：這些命令會使用預設的 SST-2 範例資料集"
	@echo "  實際訓練時的資料管理已整合在訓練流程中"
	@echo ""
	@echo "  make data-analyze   - 分析資料集分布"
	@echo "  make data-validate  - 驗證資料集品質"
	@echo "  make data-versions  - 管理資料版本"