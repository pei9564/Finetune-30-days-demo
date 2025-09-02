.PHONY: setup-conda run-local logs-local data-analyze data-validate data-versions start-services start-worker start-api help

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
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) python -u app/train_lora_v2.py \
	'



# 查看最新實驗的訓練日誌
logs-local:
	@latest_dir=$$(ls -td results/*/ 2>/dev/null | head -n1); \
	if [ -n "$$latest_dir" ] && [ -f "$$latest_dir/logs.txt" ]; then \
		echo "📋 查看最新實驗日誌（最後 20 行）..."; \
		echo "📂 實驗目錄：$$latest_dir"; \
		tail -n 20 "$$latest_dir/logs.txt"; \
		echo ""; \
		echo "💡 提示："; \
		echo "  - 使用 'tail -f $$latest_dir/logs.txt' 來持續監控日誌"; \
		echo "  - 系統日誌與訓練進度都記錄在此文件中"; \
	else \
		echo "❌ 沒有找到實驗日誌，請先運行 'make run-local'"; \
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
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.dataset_analyzer \
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
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.data_validator \
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
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.version_manager \
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
	@echo "     PYTHONPATH=$(PWD) python app/train_lora_v2.py [參數]"
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
	@echo "     - 執行 'make logs-local' 查看最新實驗的日誌"
	@echo "     - 或使用顯示的 tail -f 命令持續監控"
	@echo ""
	@echo "  2. 實驗記錄："
	@echo "     每次訓練會在 results/ 下創建獨立的實驗目錄："
	@echo "     - 實驗目錄：results/{實驗名稱}_{時間戳}/"
	@echo "     - 系統日誌：logs.txt（包含系統操作和訓練進度）"
	@echo "     - 實驗配置：config.yaml"
	@echo "     - 評估指標：metrics.json"
	@echo "     - 模型文件：artifacts/final_model/"
	@echo ""
	@echo "🔧 資料管理工具（僅供開發測試用）："
	@echo "  註：這些命令會使用預設的 SST-2 範例資料集"
	@echo "  實際訓練時的資料管理已整合在訓練流程中"
	@echo ""
	@echo "  make data-analyze   - 分析資料集分布"
	@echo "  make data-validate  - 驗證資料集品質"
	@echo "  make data-versions  - 管理資料版本"

# 啟動 Redis 服務
start-services:
	@echo "🚀 啟動 Redis 服務..."
	@if ! command -v docker-compose &> /dev/null; then \
		echo "❌ docker-compose 未安裝"; \
		exit 1; \
	fi
	docker-compose up -d

# 啟動 Celery worker
start-worker:
	@echo "👷 啟動 Celery worker..."
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
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 worker..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) celery -A app.tasks worker -l INFO -P solo \
	'

# 啟動 FastAPI 服務
start-api:
	@echo "🚀 啟動 API 服務..."
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
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 API..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) uvicorn app.api:app --reload --host 0.0.0.0 --port 8000 \
	'

	@echo ""
	@echo "🚀 非同步訓練服務："
	@echo "  make start-services - 啟動 Redis 服務"
	@echo "  make start-worker   - 啟動 Celery worker"
	@echo "  make start-api      - 啟動 FastAPI 服務"