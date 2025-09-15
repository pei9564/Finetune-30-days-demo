.PHONY: setup-conda run-local logs-local analyze-metrics analyze-by-model analyze-by-dataset \
        data-analyze data-validate data-versions db-list \
        start-services stop-services restart-services logs-services logs-service \
        k8s-setup k8s-build k8s-build-fast k8s-deploy k8s-verify k8s-cleanup \
        k8s-status k8s-logs k8s-restart k8s-scale k8s-quick-deploy k8s-full-cleanup \
        help check-docker serve predict-health predict-text predict-positive predict-negative

# ==============================================================================
# 通用變量和函數
# ==============================================================================

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

# ==============================================================================
# 本地訓練相關命令
# ==============================================================================

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
		cd $(PWD) && PYTHONPATH=$(PWD) python -u app/train_lora_v2.py $(ARGS) \
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

# ==============================================================================
# 實驗分析相關命令
# ==============================================================================

# 分析實驗效能
analyze-metrics:
	@echo "📊 分析實驗效能..."
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) python -m app.tools.analyze_metrics $(ARGS) \
	'

# 按模型分析效能
analyze-by-model:
	@echo "📊 按模型分析效能..."
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) python -m app.tools.analyze_metrics --group-by model_name \
	'

# 按資料集分析效能
analyze-by-dataset:
	@echo "📊 按資料集分析效能..."
	@bash -c '\
		$(detect_env) \
		$(check_env_exists) \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $$ENV_NAME && \
		cd $(PWD) && PYTHONPATH=$(PWD) python -m app.tools.analyze_metrics --group-by dataset_name \
	'

# ==============================================================================
# 資料管理相關命令
# ==============================================================================

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

# 分析資料集分布
data-analyze:
	@echo "📊 分析資料集分布..."
	$(call run_data_tool,"分析資料","dataset_analyzer")

# 驗證資料集品質
data-validate:
	@echo "🔍 驗證資料集品質..."
	$(call run_data_tool,"驗證資料","data_validator")

# 管理資料版本
data-versions:
	@echo "📦 管理資料版本..."
	$(call run_data_tool,"管理版本","version_manager")

# 查看實驗記錄
db-list:
	@echo "📊 查看實驗記錄..."
	@if [ ! -f "results/experiments.db" ]; then \
		echo "❌ 資料庫不存在，請先執行訓練"; \
		exit 1; \
	fi
	@sqlite3 results/experiments.db ".mode column" ".headers on" \
		"SELECT name as '實驗名稱', \
		datetime(created_at) as '創建時間', \
		printf('%.2f%%', eval_accuracy * 100) as '準確率', \
		printf('%.1fs', train_runtime) as '訓練時間' \
		FROM experiments ORDER BY created_at DESC;"

# ==============================================================================
# Docker 服務相關命令
# ==============================================================================

# 檢查 Docker 環境
check-docker:
	@if ! command -v docker-compose &> /dev/null; then \
		echo "❌ docker-compose 未安裝"; \
		exit 1; \
	fi

# 啟動所有服務
start-services: check-docker
	@echo "🚀 啟動所有服務..."
	docker compose up --build -d
	@echo "✅ 服務已啟動！"
	@echo "💡 提示："
	@echo "  - API 服務：http://localhost:8000"
	@echo "  - UI 界面：http://localhost:8501"
	@echo "  - Redis：localhost:6379"
	@echo "  - 使用 'make logs-services' 查看服務日誌"

# 停止所有服務
stop-services: check-docker
	@echo "🛑 停止所有服務..."
	docker compose down
	@echo "✅ 服務已停止"

# 重啟所有服務
restart-services: stop-services start-services

# 查看所有服務日誌
logs-services: check-docker
	@echo "📋 查看服務日誌..."
	@echo "提示：按 Ctrl+C 停止查看"
	@echo "---"
	docker compose logs -f

# 查看指定服務日誌
logs-service: check-docker
	@if [ -z "$(service)" ]; then \
		echo "❌ 請指定服務名稱：make logs-service service=<redis|worker|api|ui>"; \
		exit 1; \
	fi
	@echo "📋 查看 $(service) 服務日誌..."
	@echo "提示：按 Ctrl+C 停止查看"
	@echo "---"
	docker compose logs -f $(service)

# ==============================================================================
# Kubernetes 部署相關命令
# ==============================================================================

# 安裝並啟動 minikube
k8s-setup:
	@echo "☸️  設置 minikube 環境..."
	@if ! command -v minikube &> /dev/null; then \
		echo "📦 安裝 minikube..."; \
		if [[ "$$(uname -m)" == "arm64" ]]; then \
			curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-arm64; \
			sudo install minikube-darwin-arm64 /usr/local/bin/minikube; \
		else \
			curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-amd64; \
			sudo install minikube-darwin-amd64 /usr/local/bin/minikube; \
		fi; \
		rm -f minikube-darwin-*; \
	fi
	@echo "🚀 啟動 minikube..."
	@minikube start --driver=docker --memory=4096 --cpus=2
	@echo "✅ minikube 已啟動！"
	@echo "💡 提示：使用 'minikube dashboard' 開啟 Kubernetes 儀表板"

# 建構 Docker 映像
k8s-build:
	@./k8s/k8s.sh build

# 快速建構（輕量版）
k8s-build-fast:
	@./k8s/k8s.sh build-fast

# 部署到 Kubernetes
k8s-deploy:
	@./k8s/k8s.sh deploy

# 一鍵部署
k8s-quick-deploy: k8s-setup k8s-build-fast k8s-deploy
	@echo "🎉 一鍵部署完成！"

# 查看部署狀態
k8s-status:
	@./k8s/k8s.sh status

# 查看服務日誌
k8s-logs:
	@./k8s/k8s.sh logs $(service)

# 重啟服務
k8s-restart:
	@./k8s/k8s.sh restart

# 擴展服務
k8s-scale:
	@./k8s/k8s.sh scale $(replicas)

# 驗證部署
k8s-verify:
	@./k8s/k8s.sh verify

# 清理資源
k8s-cleanup:
	@./k8s/k8s.sh cleanup

# 完全清理
k8s-full-cleanup:
	@./k8s/k8s.sh full-cleanup

# ==============================================================================
# 推理服務相關命令
# ==============================================================================

# 尋找最新的實驗
define find_latest_experiment
	@latest_exp=$$(ls -td results/*/ 2>/dev/null | head -n1 | xargs basename); \
	if [ -z "$$latest_exp" ]; then \
		echo "❌ 找不到訓練好的模型！請先執行訓練。"; \
		exit 1; \
	fi; \
	if [ ! -d "results/$$latest_exp/artifacts/final_model" ]; then \
		echo "❌ 最新實驗 $$latest_exp 中找不到模型！"; \
		exit 1; \
	fi; \
	echo "✅ 找到最新實驗：$$latest_exp"
endef

# 啟動推理服務
serve:
	@echo "🚀 啟動推理服務..."
	@if [ -n "$(exp)" ]; then \
		if [ ! -d "results/$(exp)/artifacts/final_model" ]; then \
			echo "❌ 找不到實驗模型：results/$(exp)/artifacts/final_model"; \
			exit 1; \
		fi; \
		echo "📌 使用實驗：$(exp)"; \
		MODEL_PATH="results/$(exp)/artifacts/final_model" bash -c '\
			$(detect_env) \
			$(check_env_exists) \
			source $$(conda info --base)/etc/profile.d/conda.sh && \
			conda activate $$ENV_NAME && \
			cd $(PWD) && PYTHONPATH=$(PWD) python app/inference_api.py \
		'; \
	else \
		$(find_latest_experiment); \
		echo "📌 使用最新實驗：$$latest_exp"; \
		MODEL_PATH="results/$$latest_exp/artifacts/final_model" bash -c '\
			$(detect_env) \
			$(check_env_exists) \
			source $$(conda info --base)/etc/profile.d/conda.sh && \
			conda activate $$ENV_NAME && \
			cd $(PWD) && PYTHONPATH=$(PWD) python app/inference_api.py \
		'; \
	fi


# 預測服務測試指令
predict-health:
	@echo "🔍 檢查服務狀態..."
	@curl -s http://localhost:8002/health | python3 -m json.tool

predict-text:
	@echo "🔍 測試文本預測..."
	@if [ -z "$(text)" ]; then \
		echo "❌ 請提供測試文本：make predict-text text='This movie was great!'"; \
		exit 1; \
	fi
	@curl -s -X POST http://localhost:8002/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "$(text)"}' | python3 -m json.tool

predict-positive:
	@echo "🔍 測試正面評論範例..."
	@curl -s -X POST http://localhost:8002/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "This movie was fantastic! I really enjoyed it."}' | python3 -m json.tool

predict-negative:
	@echo "🔍 測試負面評論範例..."
	@curl -s -X POST http://localhost:8002/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "This was a terrible movie. Complete waste of time."}' | python3 -m json.tool

# ==============================================================================
# 幫助信息
# ==============================================================================

help:
	@echo "🍎 LoRA 訓練環境管理命令"
	@echo ""
	@echo "🚀 推理服務："
	@echo "  1. 服務管理："
	@echo "     make serve - 使用最新實驗啟動服務"
	@echo "     make serve exp=實驗名稱 - 使用指定實驗啟動"
	@echo "     例如：make serve exp=default_experiment_20250911_233842"
	@echo ""
	@echo "  2. 預測測試："
	@echo "     make predict-health   - 檢查服務狀態"
	@echo "     make predict-text     - 測試自訂文本，例如："
	@echo "     make predict-text text='This movie was great!'"
	@echo "     make predict-positive - 測試正面評論範例"
	@echo "     make predict-negative - 測試負面評論範例"
	@echo ""
	@echo "🚀 訓練模式："
	@echo "  1. 本地直接訓練："
	@echo "     make setup-conda   - 首次使用：檢查並創建 Conda 環境"
	@echo "     make run-local     - 執行訓練（使用預設配置）"
	@echo "     make logs-local    - 查看最新實驗的訓練進度"
	@echo ""
	@echo "  2. 非同步訓練服務（Docker）："
	@echo "     make start-services  - 啟動所有服務"
	@echo "     make stop-services   - 停止所有服務"
	@echo "     make restart-services - 重啟所有服務"
	@echo "     make logs-services   - 查看所有服務日誌"
	@echo "     make logs-service service=<redis|worker|api|ui> - 查看指定服務日誌"
	@echo ""
	@echo "📊 實驗管理："
	@echo "  1. 網頁界面（推薦）："
	@echo "     - 訪問 http://localhost:8501"
	@echo "     - 提交任務：選擇「提交任務」頁籤，設置參數"
	@echo "     - 追蹤進度：選擇「追蹤進度」頁籤，輸入 task_id"
	@echo "     - 實驗記錄：選擇「實驗記錄」頁籤，查看所有實驗"
	@echo ""
	@echo "  2. 命令列工具："
	@echo "     make db-list       - 查看實驗記錄（表格形式）"
	@echo "     make logs-local    - 查看最新實驗的訓練進度"
	@echo "     make analyze-metrics - 分析實驗效能"
	@echo "     make analyze-by-model - 按模型分析效能"
	@echo "     make analyze-by-dataset - 按資料集分析效能"
	@echo ""
	@echo "⚙️ 配置管理："
	@echo "  1. 使用預設配置："
	@echo "     - 編輯 config/default.yaml"
	@echo "     - 包含所有可調整的參數"
	@echo ""
	@echo "  2. 使用命令列參數（僅用於本地訓練）："
	@echo "     PYTHONPATH=$(PWD) python app/train_lora_v2.py [參數]"
	@echo ""
	@echo "     常用參數："
	@echo "     --experiment_name TEXT    實驗名稱"
	@echo "     --learning_rate FLOAT     學習率"
	@echo "     --epochs INT              訓練輪數"
	@echo "     --train_samples INT       訓練樣本數"
	@echo "     --device TEXT             指定設備 (cuda/mps/cpu)"
	@echo ""
	@echo "🔧 資料管理工具（僅供開發測試用）："
	@echo "  註：這些命令會使用預設的 SST-2 範例資料集"
	@echo "  實際訓練時的資料管理已整合在訓練流程中"
	@echo ""
	@echo "  make data-analyze   - 分析資料集分布"
	@echo "  make data-validate  - 驗證資料集品質"
	@echo "  make data-versions  - 管理資料版本"
	@echo ""
	@echo "☸️  Kubernetes 部署："
	@echo "  1. 快速開始："
	@echo "     make k8s-quick-deploy  - 一鍵部署（建構+部署）"
	@echo "     make k8s-setup         - 安裝並啟動 minikube"
	@echo "     make k8s-build         - 建構 Docker 映像"
	@echo "     make k8s-build-fast    - 快速建構（輕量版）"
	@echo "     make k8s-deploy        - 部署到 Kubernetes"
	@echo ""
	@echo "  2. 管理操作："
	@echo "     make k8s-status        - 查看部署狀態"
	@echo "     make k8s-logs          - 查看服務日誌"
	@echo "     make k8s-restart       - 重啟服務"
	@echo "     make k8s-scale         - 擴展服務"
	@echo "     make k8s-verify        - 驗證部署"
	@echo "     make k8s-cleanup       - 清理資源"
	@echo "     make k8s-full-cleanup  - 完全清理（包含映像）"