#!/bin/bash

# Kubernetes 管理腳本
# 用法: ./k8s.sh <command> [args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切換到專案根目錄
cd "$PROJECT_ROOT"

# 檢查 minikube 是否安裝
check_minikube() {
    if ! command -v minikube &> /dev/null; then
        echo "❌ minikube 未安裝，請先運行 'make k8s-setup'"
        exit 1
    fi
}

# 檢查 minikube 是否運行
check_minikube_running() {
    if ! minikube status > /dev/null 2>&1; then
        echo "❌ minikube 未運行，請先運行 'make k8s-setup'"
        exit 1
    fi
}

# 設置 Docker 環境
setup_docker_env() {
    eval $(minikube docker-env)
    export DOCKER_BUILDKIT=1
}

case "$1" in
    "build")
        echo "📦 建構 Docker 映像..."
        check_minikube
        check_minikube_running
        setup_docker_env
        echo "🔨 建構統一映像..."
        docker build \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --cache-from finetune-app:latest \
            --progress=plain \
            -t finetune-app:latest \
            -f Dockerfile \
            .
        echo "✅ 映像建構完成！"
        ;;
    "build-fast")
        echo "⚡ 快速建構 Docker 映像..."
        check_minikube
        check_minikube_running
        setup_docker_env
        echo "⚡ 建構輕量映像..."
        docker build \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --cache-from finetune-app:latest \
            --progress=plain \
            --target=minimal \
            -t finetune-app:latest \
            -f Dockerfile \
            .
        echo "✅ 快速建構完成！"
        ;;
    "deploy")
        echo "🚀 部署到 Kubernetes..."
        if ! command -v kubectl &> /dev/null; then
            echo "❌ kubectl 未安裝，請先安裝 kubectl"
            exit 1
        fi
        check_minikube_running
        echo "📋 部署 Kubernetes 資源..."
        # 按順序部署，確保依賴關係正確
        echo "1️⃣ 創建 namespace..."
        kubectl apply -f k8s/manifests/namespace.yaml
        echo "2️⃣ 創建 configmap..."
        kubectl apply -f k8s/manifests/configmap.yaml
        echo "3️⃣ 創建 secrets..."
        kubectl apply -f k8s/manifests/secrets.yaml
        echo "4️⃣ 創建 PVC..."
        kubectl apply -f k8s/manifests/pvc.yaml
        echo "5️⃣ 部署 Redis..."
        kubectl apply -f k8s/manifests/redis.yaml
        echo "6️⃣ 部署 Worker..."
        kubectl apply -f k8s/manifests/worker.yaml
        echo "7️⃣ 部署 API..."
        kubectl apply -f k8s/manifests/api.yaml
        echo "8️⃣ 部署 UI..."
        kubectl apply -f k8s/manifests/ui.yaml
        echo "⏳ 等待服務啟動..."
        kubectl wait --for=condition=available --timeout=300s deployment/api -n lora-system
        kubectl wait --for=condition=available --timeout=300s deployment/ui -n lora-system
        echo "✅ 部署完成！"
        echo ""
        echo "🌐 服務訪問："
        echo "  - API: http://localhost:8000"
        echo "  - UI: http://localhost:8501"
        echo "  - Redis: redis:6379 (集群內)"
        ;;
    "status")
        echo "📊 Kubernetes 部署狀態："
        echo ""
        echo "🏷️  Namespace:"
        kubectl get namespaces | grep lora-system || echo "❌ lora-system namespace 不存在"
        echo ""
        echo "🚀 Deployments:"
        kubectl get deployments -n lora-system
        echo ""
        echo "🌐 Services:"
        kubectl get services -n lora-system
        echo ""
        echo "📦 Pods:"
        kubectl get pods -n lora-system
        echo ""
        echo "💾 PVC:"
        kubectl get pvc -n lora-system
        ;;
    "logs")
        if [ -z "$2" ]; then
            echo "📋 所有服務日誌："
            kubectl logs -l app=api -n lora-system --tail=50
            echo ""
            echo "📋 UI 服務日誌："
            kubectl logs -l app=ui -n lora-system --tail=50
            echo ""
            echo "📋 Worker 服務日誌："
            kubectl logs -l app=worker -n lora-system --tail=50
            echo ""
            echo "📋 Redis 服務日誌："
            kubectl logs -l app=redis -n lora-system --tail=50
        else
            echo "📋 $2 服務日誌："
            kubectl logs -l app=$2 -n lora-system --tail=50 -f
        fi
        ;;
    "restart")
        echo "🔄 重啟服務..."
        kubectl rollout restart deployment/api -n lora-system
        kubectl rollout restart deployment/ui -n lora-system
        kubectl rollout restart deployment/worker -n lora-system
        echo "✅ 服務重啟完成！"
        ;;
    "scale")
        if [ -z "$2" ]; then
            echo "❌ 請指定副本數：./k8s.sh scale 3"
            exit 1
        fi
        echo "📈 擴展服務到 $2 個副本..."
        kubectl scale deployment api --replicas=$2 -n lora-system
        kubectl scale deployment ui --replicas=$2 -n lora-system
        kubectl scale deployment worker --replicas=$2 -n lora-system
        echo "✅ 擴展完成！"
        ;;
    "verify")
        echo "🔍 驗證部署..."
        echo "📊 檢查 Pod 狀態："
        kubectl get pods -n lora-system -o wide
        echo ""
        echo "🌐 檢查服務端點："
        kubectl get endpoints -n lora-system
        echo ""
        echo "💾 檢查 PVC 狀態："
        kubectl get pvc -n lora-system
        echo ""
        echo "🔗 檢查服務連接："
        kubectl get services -n lora-system
        ;;
    "cleanup")
        echo "🧹 清理 Kubernetes 資源..."
        kubectl delete -f k8s/manifests/ --ignore-not-found=true
        echo "✅ 資源清理完成！"
        ;;
    "full-cleanup")
        echo "🧹 完全清理（包含映像）..."
        setup_docker_env
        docker rmi finetune-app:latest 2>/dev/null || true
        kubectl delete -f k8s/manifests/ --ignore-not-found=true
        echo "✅ 完全清理完成！"
        ;;
    *)
        echo "☸️  Kubernetes 管理腳本"
        echo ""
        echo "用法: $0 <command> [args...]"
        echo ""
        echo "可用命令："
        echo "  build         - 建構 Docker 映像"
        echo "  build-fast    - 快速建構（輕量版）"
        echo "  deploy        - 部署到 Kubernetes"
        echo "  status        - 查看部署狀態"
        echo "  logs [service] - 查看服務日誌"
        echo "  restart       - 重啟服務"
        echo "  scale <num>   - 擴展服務"
        echo "  verify        - 驗證部署"
        echo "  cleanup       - 清理資源"
        echo "  full-cleanup  - 完全清理（包含映像）"
        echo ""
        echo "範例："
        echo "  $0 build-fast"
        echo "  $0 deploy"
        echo "  $0 logs api"
        echo "  $0 scale 3"
        ;;
esac
