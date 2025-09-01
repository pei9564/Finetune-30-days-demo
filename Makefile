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

# 本地運行訓練（自動檢測最佳加速方式）
run-local:
	@echo "🚀 檢查並運行本地 LoRA 訓練..."
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
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $$ENV_NAME && python -u app/train_lora.py \
	'

# 查看本地訓練 log
logs-local:
	@if [ -f logs/local_training.log ]; then \
		echo "📋 查看訓練 log（最後 20 行）..."; \
		tail -n 20 logs/local_training.log; \
		echo ""; \
		echo "💡 提示："; \
		echo "  - 使用 'tail -f logs/local_training.log' 來持續監控 log"; \
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
	@echo "🍎 本地 Conda 模式："
	@echo "  setup-conda   - 檢查並創建 Conda 環境"
	@echo "  run-local     - 在本地運行訓練"
	@echo "  logs-local    - 查看訓練 log（最後 20 行）"
	@echo ""
	@echo "📊 資料管理（僅用於測試範例）："
	@echo "  data-analyze  - 分析資料集分布與統計"
	@echo "  data-validate - 驗證資料集品質"
	@echo "  data-versions - 管理資料版本"
	@echo ""
	@echo "📚 其他："
	@echo "  help          - 顯示此幫助信息"
	@echo ""
	@echo "💡 提示："
	@echo "  - 首次使用請先執行 'make setup-conda' 設置環境"
	@echo "  - 然後使用 'make run-local' 開始訓練"
	@echo "  - 想要即時查看 log 請使用 'make run-local' 直接運行"
	@echo "  - 查看 log 文件：'make logs-local'"
	@echo "  - 持續監控 log：'tail -f logs/local_training.log'"
	@echo "  - 資料分析與管理：'make data-analyze', 'make data-validate', 'make data-versions'"


	