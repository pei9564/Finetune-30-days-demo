.PHONY: setup-conda run-local logs-local data-analyze data-validate data-versions start-services start-worker start-api start-ui help

# 通用變量
PYTHON_VERSION := 3.11
PYTHONPATH := $(PWD)

# 定義檢測環境的函數
define detect_env
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
	echo "📦 使用環境名稱: $$ENV_NAME";
endef

# 定義檢查 Conda 環境的函數
define check_conda
	@if ! command -v conda &> /dev/null; then \
		echo "❌ Conda 未安裝，請先安裝 miniforge：brew install --cask miniforge"; \
		exit 1; \
	fi
endef

# 定義檢查環境存在的函數
define check_env_exists
	if ! conda env list | grep -q "$$ENV_NAME"; then \
		echo "❌ Conda 環境 \"$$ENV_NAME\" 不存在，請先運行 \"make setup-conda\""; \
		exit 1; \
	fi;
endef

# 本地 Conda 環境設置
setup-conda:
	@echo "🔍 檢測系統環境..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		if conda env list | grep -q "$$ENV_NAME"; then \
			echo "✅ Conda 環境 \"$$ENV_NAME\" 已存在"; \
		else \
			echo "📦 創建新的 Conda 環境 \"$$ENV_NAME\"..."; \
			conda create --name $$ENV_NAME python=$(PYTHON_VERSION) -y; \
			echo "✅ 環境創建完成！"; \
		fi; \
		echo "📦 安裝依賴..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		pip install --upgrade pip && pip install -r requirements.txt; \
		echo "✅ 依賴安裝完成！"; \
		echo "📋 下一步：make run-local" \
	'

# 本地運行訓練
run-local:
	@echo "🚀 檢查並運行 LoRA 訓練（使用預設配置）..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
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

# 資料管理工具（僅用於測試範例）
define run_data_tool
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		echo "🔧 使用環境 \"$$ENV_NAME\" $(1)..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) PYTHONWARNINGS="ignore::RuntimeWarning" python -m app.data_management.$(2) \
	'
endef

data-analyze:
	@echo "📊 分析資料集分布..."
	$(call run_data_tool,"分析資料","dataset_analyzer")

data-validate:
	@echo "🔍 驗證資料集品質..."
	$(call run_data_tool,"驗證資料","data_validator")

data-versions:
	@echo "📦 管理資料版本..."
	$(call run_data_tool,"管理版本","version_manager")

# 非同步訓練服務
start-services:
	@echo "🚀 啟動 Redis 服務..."
	@if ! command -v docker-compose &> /dev/null; then \
		echo "❌ docker-compose 未安裝"; \
		exit 1; \
	fi
	docker-compose up -d

# 定義啟動服務的函數
define start_service
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 $(1)..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) $(2) \
	'
endef

start-worker:
	@echo "👷 啟動 Celery worker..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 worker..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && \
		PYTHONPATH=$(PWD) python -m celery -A app.tasks worker -l INFO -P solo \
	'

start-api:
	@echo "🚀 啟動 API 服務..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 API..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && \
		PYTHONPATH=$(PWD) python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000 \
	'

start-ui:
	@echo "🚀 啟動進度追蹤 UI..."
	$(check_conda)
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		echo "🚀 使用環境 \"$$ENV_NAME\" 啟動 UI..."; \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && \
		PYTHONPATH=$(PWD) python -m streamlit run app/stepper_ui.py \
	'

# 顯示幫助
help:
	@echo "🍎 LoRA 訓練環境管理命令"
	@echo ""
	@echo "🚀 訓練模式："
	@echo "  1. 本地直接訓練："
	@echo "     make setup-conda   - 首次使用：檢查並創建 Conda 環境"
	@echo "     make run-local     - 執行訓練（使用預設配置）"
	@echo "     make logs-local    - 查看訓練進度"
	@echo ""
	@echo "  2. 非同步訓練服務（需要開啟四個終端）："
	@echo "     make start-services - 啟動 Redis 服務（任務佇列）"
	@echo "     make start-worker   - 啟動 Celery worker（執行訓練）"
	@echo "     make start-api      - 啟動 FastAPI 服務（接收請求）"
	@echo "     make start-ui       - 啟動網頁界面（提交任務與查看進度）"
	@echo ""
	@echo "⚙️ 配置方式："
	@echo "  1. 使用網頁界面（推薦）："
	@echo "     - 訪問 http://localhost:8501"
	@echo "     - 在表單中設置實驗參數"
	@echo "     - 提交任務並追蹤進度"
	@echo ""
	@echo "  2. 使用預設配置："
	@echo "     - 編輯 config/default.yaml"
	@echo "     - 包含所有可調整的參數"
	@echo ""
	@echo "  3. 使用命令列（僅用於本地訓練）："
	@echo "     PYTHONPATH=$(PWD) python app/train_lora_v2.py [參數]"
	@echo ""
	@echo "     常用參數："
	@echo "     --experiment_name TEXT    實驗名稱"
	@echo "     --learning_rate FLOAT     學習率"
	@echo "     --epochs INT              訓練輪數"
	@echo "     --train_samples INT       訓練樣本數"
	@echo "     --device TEXT             指定設備 (cuda/mps/cpu)"
	@echo ""
	@echo "📊 實驗記錄："
	@echo "  1. 實驗目錄結構："
	@echo "     每次訓練（無論本地或非同步）都會創建獨立目錄："
	@echo "     results/{實驗名稱}_{時間戳}/"
	@echo "     ├── logs.txt           - 系統日誌與訓練進度"
	@echo "     ├── config.yaml        - 本次實驗的完整配置"
	@echo "     ├── metrics.json       - 訓練結果與評估指標"
	@echo "     └── artifacts/         - 模型與其他產出"
	@echo "         └── final_model/   - 訓練完成的模型"
	@echo ""
	@echo "  2. 查看方式："
	@echo "     - 本地訓練：使用 make logs-local"
	@echo "     - 非同步訓練：使用網頁界面"
	@echo ""
	@echo "🔧 資料管理工具（僅供開發測試用）："
	@echo "  註：這些命令會使用預設的 SST-2 範例資料集"
	@echo "  實際訓練時的資料管理已整合在訓練流程中"
	@echo ""
	@echo "  make data-analyze   - 分析資料集分布"
	@echo "  make data-validate  - 驗證資料集品質"
	@echo "  make data-versions  - 管理資料版本"