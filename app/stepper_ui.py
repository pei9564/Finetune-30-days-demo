"""
訓練任務管理 UI
使用 Streamlit 實現任務提交和進度追蹤
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st
import yaml

from app.core.settings import API_URL


def load_default_config() -> Dict:
    """載入預設配置

    Returns:
        Dict: 預設配置
    """
    try:
        with open("config/default.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"載入預設配置失敗：{e}")
        return {}


def login(username: str, password: str) -> Optional[Dict]:
    """登入並獲取 JWT token 和用戶資訊

    Args:
        username: 使用者名稱
        password: 密碼

    Returns:
        Dict: 包含 token 和用戶資訊的字典，如果登入失敗則返回 None
    """
    try:
        with st.spinner("登入中..."):  # 添加載入指示器
            # 使用 API_URL 環境變數（在 Docker 中設置為 http://api:8000）
            response = requests.post(
                f"{API_URL}/auth/login",
                json={"username": username, "password": password},
                timeout=5,  # 設置 5 秒超時
            )

        if response.status_code == 200:
            data = response.json()
            return {
                "token": data["token"],
                "role": data["role"],
                "user_id": data["user_id"],
            }
        elif response.status_code == 401:
            st.error("登入失敗：帳號或密碼錯誤")
            return None
        else:
            st.error(f"登入失敗：伺服器錯誤 (狀態碼: {response.status_code})")
            return None

    except requests.exceptions.Timeout:
        st.error("登入超時：伺服器回應時間過長，請稍後再試")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"連接失敗：無法連接到伺服器 ({str(e)})")
        return None
    except Exception as e:
        st.error(f"登入失敗：{str(e)}")
        return None


def get_task_status(task_id: str) -> Optional[Dict]:
    """查詢任務狀態

    Args:
        task_id: 任務 ID

    Returns:
        Dict: 任務狀態資訊，如果請求失敗則返回 None
    """
    try:
        # 從 session state 獲取 token
        token = st.session_state.get("jwt_token")
        if not token:
            st.error("請先登入")
            return None

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_URL}/task/{task_id}", headers=headers)
        return response.json()
    except Exception as e:
        st.error(f"查詢失敗：{e}")
        return None


def render_stepper(status: str):
    """渲染進度指示器

    Args:
        status: 任務狀態
    """
    # 定義所有步驟
    steps = ["PENDING", "STARTED", "SUCCESS/FAILURE"]
    current_step = 0

    # 根據狀態確定當前步驟
    if status == "PENDING":
        current_step = 0
    elif status == "STARTED":
        current_step = 1
    elif status in ["SUCCESS", "FAILURE"]:
        current_step = 2

    # 創建進度條容器
    cols = st.columns(len(steps))

    # 渲染每個步驟
    for i, (step, col) in enumerate(zip(steps, cols)):
        # 設置步驟樣式
        if i < current_step:  # 已完成
            color = "green"
            symbol = "✅"
        elif i == current_step:  # 當前
            color = "blue"
            symbol = "🔄"
            if status == "FAILURE" and i == 2:
                color = "red"
                symbol = "❌"
            elif status == "SUCCESS" and i == 2:
                color = "green"
                symbol = "✅"
        else:  # 未開始
            color = "gray"
            symbol = "⭕️"

        # 在每個 column 中顯示步驟
        with col:
            st.markdown(
                f'<div style="text-align: center; color: {color};">'
                f"<h4>{symbol} {step}</h4>"
                f"</div>",
                unsafe_allow_html=True,
            )


def check_auth() -> bool:
    """檢查是否已登入

    Returns:
        bool: 是否已登入
    """
    if "jwt_token" not in st.session_state:
        st.warning("請先登入")

        # 使用表單進行登入
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("使用者名稱", key="login_username")
            password = st.text_input("密碼", type="password", key="login_password")

            # 使用 columns 來對齊按鈕
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                submitted = st.form_submit_button("登入", use_container_width=True)
            with btn_col2:
                guest_login = st.form_submit_button(
                    "訪客登入", use_container_width=True
                )

            if submitted or guest_login:
                # 如果是訪客登入，使用預設帳密
                if guest_login:
                    username = "guest"
                    password = "guest123"
                elif not username or not password:
                    st.error("請輸入使用者名稱和密碼")
                    return False

                auth_data = login(username, password)
                if auth_data:
                    st.session_state.jwt_token = auth_data["token"]
                    st.session_state.user_role = auth_data["role"]
                    st.session_state.user_id = auth_data["user_id"]
                    st.success(
                        f"登入成功！歡迎 {auth_data['user_id']} ({auth_data['role']})"
                    )
                    st.rerun()
                    return True

        return False
    return True


def submit_training_task(
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    train_samples: Optional[int] = None,
    eval_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
) -> Optional[str]:
    """提交訓練任務

    Args:
        experiment_name: 實驗名稱
        model_name: 預訓練模型名稱
        dataset: 資料集名稱
        train_samples: 訓練樣本數
        eval_samples: 驗證樣本數
        batch_size: 批次大小
        lora_r: LoRA 秩
        lora_alpha: LoRA Alpha
        learning_rate: 學習率
        epochs: 訓練輪數
        device: 訓練設備

    Returns:
        Optional[str]: 任務 ID，如果提交失敗則返回 None
    """
    try:
        # 準備配置
        config = load_default_config()

        # 更新配置
        if experiment_name:
            config["experiment_name"] = experiment_name
        if model_name:
            config["model"]["name"] = model_name
        if dataset:
            # 處理資料集名稱
            if "/" in dataset:
                dataset_name, dataset_config = dataset.split("/")
            else:
                dataset_name = dataset
                dataset_config = None
            config["data"]["dataset_name"] = dataset_name
            if dataset_config:
                config["data"]["dataset_config"] = dataset_config
        if train_samples:
            config["data"]["train_samples"] = train_samples
        if eval_samples:
            config["data"]["eval_samples"] = eval_samples
        if batch_size:
            config["training"]["per_device_train_batch_size"] = batch_size
        if lora_r:
            config["lora"]["r"] = lora_r
        if lora_alpha:
            config["lora"]["lora_alpha"] = lora_alpha
        if learning_rate:
            config["training"]["learning_rate"] = learning_rate
        if epochs:
            config["training"]["num_train_epochs"] = epochs
        if device and device != "auto":
            config["training"]["device"] = device

        # 提交任務
        headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
        response = requests.post(
            f"{API_URL}/train",
            json={"config": config},
            headers=headers,
        )
        result = response.json()
        return result.get("task_id")
    except Exception as e:
        st.error(f"提交任務失敗：{e}")
        return None


def render_task_form():
    """渲染任務提交表單"""
    st.markdown("### 提交訓練任務")

    # 載入預設配置
    default_config = load_default_config()

    # 創建表單
    with st.form("training_form"):
        # 實驗設置
        st.markdown("#### 實驗設置")
        experiment_name = st.text_input(
            "實驗名稱",
            value=default_config.get("experiment_name", "default_experiment"),
            help="為這次實驗取一個有意義的名稱",
        )

        # 模型設置
        st.markdown("#### 模型設置")
        model_name = st.selectbox(
            "預訓練模型",
            options=["distilbert-base-uncased", "bert-base-uncased", "roberta-base"],
            index=0,
            help="選擇要微調的基礎模型",
        )

        # 資料設置
        st.markdown("#### 資料設置")
        col1, col2 = st.columns(2)
        with col1:
            dataset = st.selectbox(
                "資料集",
                options=["glue/sst2", "glue/cola", "imdb"],
                index=0,
                help="選擇訓練資料集（格式：dataset_name/config 或 dataset_name）",
            )
            train_samples = st.number_input(
                "訓練樣本數",
                value=int(default_config.get("data", {}).get("train_samples", 500)),
                min_value=100,
                max_value=10000,
                step=100,
                help="使用多少樣本進行訓練",
            )
        with col2:
            eval_samples = st.number_input(
                "驗證樣本數",
                value=int(default_config.get("data", {}).get("eval_samples", 100)),
                min_value=50,
                max_value=2000,
                step=50,
                help="使用多少樣本進行驗證",
            )
            batch_size = st.number_input(
                "批次大小",
                value=int(
                    default_config.get("training", {}).get(
                        "per_device_train_batch_size", 2
                    )
                ),
                min_value=1,
                max_value=32,
                help="每批處理的樣本數量",
            )

        # LoRA 設置
        st.markdown("#### LoRA 設置")
        col3, col4 = st.columns(2)
        with col3:
            lora_r = st.number_input(
                "LoRA 秩 (r)",
                value=int(default_config.get("lora", {}).get("r", 8)),
                min_value=1,
                max_value=64,
                help="LoRA 矩陣的秩，越大效果越好但參數更多",
            )
            learning_rate = st.number_input(
                "學習率",
                value=float(
                    default_config.get("training", {}).get("learning_rate", 5e-4)
                ),
                format="%.0e",
                help="訓練的學習率",
            )
        with col4:
            lora_alpha = st.number_input(
                "LoRA Alpha",
                value=int(default_config.get("lora", {}).get("lora_alpha", 16)),
                min_value=1,
                max_value=128,
                help="LoRA 的縮放參數",
            )
            epochs = st.number_input(
                "訓練輪數",
                value=int(
                    default_config.get("training", {}).get("num_train_epochs", 1)
                ),
                min_value=1,
                max_value=10,
                help="完整訓練資料集的次數",
            )

        # 系統設置
        st.markdown("#### 系統設置")
        device = st.selectbox(
            "訓練設備",
            options=["auto", "cuda", "mps", "cpu"],
            index=0,
            help="選擇訓練使用的設備，auto 會自動選擇最佳設備",
        )

        # 提交按鈕
        submitted = st.form_submit_button("提交訓練任務")
        if submitted:
            # 提交任務
            task_id = submit_training_task(
                experiment_name=experiment_name,
                model_name=model_name,
                dataset=dataset,
                train_samples=train_samples,
                eval_samples=eval_samples,
                batch_size=batch_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                learning_rate=learning_rate,
                epochs=epochs,
                device=None if device == "auto" else device,
            )

            if task_id:
                st.success(f"任務提交成功！任務 ID：{task_id}")
                # 將任務 ID 存入 session state
                st.session_state.last_task_id = task_id


def render_experiment_list():
    """渲染實驗列表"""
    # 標題列與重新整理按鈕並排
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown("### 實驗記錄")
    with col2:
        if st.button("🔄", help="重新整理資料"):
            st.rerun()

    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning("請先登入後查看實驗記錄")
        return

    # 篩選條件
    with st.expander("篩選條件", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            name_filter = st.text_input("實驗名稱", help="支援模糊搜尋")
            min_accuracy = st.slider(
                "最低準確率",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="篩選達到特定準確率的實驗",
            )
        with col2:
            max_runtime = st.number_input(
                "最長訓練時間（秒）",
                min_value=0,
                value=0,
                help="篩選在特定時間內完成的實驗",
            )
            sort_by = st.selectbox(
                "排序依據",
                options=["created_at", "name", "train_runtime", "eval_accuracy"],
                format_func=lambda x: {
                    "created_at": "創建時間",
                    "name": "實驗名稱",
                    "train_runtime": "訓練時間",
                    "eval_accuracy": "驗證準確率",
                }[x],
            )
        desc = st.checkbox("降序排序", value=True)

    # 發送請求
    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}

    # 先獲取統計資訊
    try:
        stats_response = requests.get(
            f"{API_URL}/experiments/stats", headers=headers, timeout=5
        )
        stats = stats_response.json()

        # 顯示統計資訊
        if stats.get("total_experiments", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("總實驗數", stats["total_experiments"])
            with col2:
                st.metric("平均準確率", f"{stats.get('avg_accuracy', 0):.2%}")
            with col3:
                st.metric("最佳準確率", f"{stats.get('best_accuracy', 0):.2%}")
            with col4:
                st.metric("最短訓練時間", f"{stats.get('min_runtime', 0):.1f}s")
        else:
            st.info("📊 目前還沒有任何實驗記錄")
            st.markdown("---")
            st.markdown("💡 **開始使用：**")
            st.markdown("1. 點擊「提交任務」頁籤")
            st.markdown("2. 設置訓練參數")
            st.markdown("3. 提交訓練任務")
            return

    except Exception:
        st.warning("無法獲取統計資訊，但仍然嘗試載入實驗列表...")

        # 獲取實驗列表
        params = {
            "sort_by": sort_by,
            "desc": desc,
            "limit": 100,
        }
        if name_filter:
            params["name"] = name_filter
        if min_accuracy > 0:
            params["min_accuracy"] = min_accuracy
        if max_runtime > 0:
            params["max_runtime"] = max_runtime

        response = requests.get(
            f"{API_URL}/experiments", params=params, headers=headers, timeout=5
        )
        experiments = response.json()

        # 顯示實驗列表
        if experiments:
            st.markdown("#### 實驗列表")
            # 創建實驗表格
            data = []
            for exp in experiments:
                data.append(
                    {
                        "實驗名稱": exp["name"],
                        "創建時間": datetime.fromisoformat(exp["created_at"]).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "訓練時間": f"{exp['train_runtime']:.1f}s",
                        "準確率": f"{exp['eval_accuracy']:.2%}",
                        "ID": exp["id"],
                    }
                )

            # 顯示表格
            st.dataframe(
                data,
                column_config={
                    "ID": st.column_config.TextColumn(
                        "ID",
                        help="實驗唯一識別碼",
                        width="medium",
                    ),
                },
                hide_index=True,
                width="stretch",
            )
        else:
            if name_filter or min_accuracy > 0 or max_runtime > 0:
                st.info("沒有找到符合篩選條件的實驗")
            else:
                st.info("📊 目前還沒有任何實驗記錄")
                st.markdown("---")
                st.markdown("💡 **開始使用：**")
                st.markdown("1. 點擊「提交任務」頁籤")
                st.markdown("2. 設置訓練參數")
                st.markdown("3. 提交訓練任務")

    except requests.exceptions.Timeout:
        st.error("載入實驗記錄超時，請稍後再試")
    except requests.exceptions.ConnectionError:
        st.error("無法連接到伺服器，請確認服務是否正常運行")
    except Exception as e:
        st.error(f"載入實驗記錄時發生錯誤：{str(e)}")
        st.markdown("---")
        st.markdown("💡 **可能的解決方案：**")
        st.markdown("1. 重新整理頁面")
        st.markdown("2. 檢查網路連接")
        st.markdown("3. 確認服務是否正常運行")


def render_task_progress():
    """渲染任務進度"""
    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning("請先登入後查看任務進度")
        return

    # 如果有最新提交的任務 ID，自動填入
    default_task_id = st.session_state.get("last_task_id", "")

    # 輸入任務 ID
    task_id = st.text_input("請輸入任務 ID", value=default_task_id)
    check_status = st.button("查詢狀態")

    # 如果有輸入 ID 且點擊查詢
    if task_id and check_status:
        status_placeholder = st.empty()

        try:
            while True:
                # 查詢狀態
                result = get_task_status(task_id)
                if not result:
                    st.error("無法獲取任務狀態，請確認任務 ID 是否正確")
                    break

                # 清空佔位元件並顯示新狀態
                with status_placeholder:
                    st.markdown("---")

                    # 安全地獲取狀態
                    status = result.get("status")
                    if not status:
                        st.error("任務狀態資訊不完整")
                        break

                    st.markdown(f"**任務狀態**：{status}")
                    render_stepper(status)

                    # 如果有結果，顯示
                    if "result" in result:
                        st.markdown("---")
                        st.markdown("**訓練結果**：")
                        st.json(result["result"])
                    # 如果有錯誤，顯示
                    elif "error" in result:
                        st.markdown("---")
                        st.error(f"錯誤信息：{result.get('error', '未知錯誤')}")

                    # 根據狀態顯示不同的提示
                    if status == "PENDING":
                        st.info("⏳ 任務正在等待執行...")
                    elif status == "STARTED":
                        st.info("🚀 任務正在執行中...")
                    elif status == "SUCCESS":
                        st.success("✅ 任務已完成！")
                    elif status == "FAILURE":
                        st.error("❌ 任務執行失敗")
                    else:
                        st.warning(f"⚠️ 未知狀態：{status}")

                # 如果任務完成或失敗，停止輪詢
                if status in ["SUCCESS", "FAILURE"]:
                    break

                # 等待 2 秒後再次查詢
                time.sleep(2)

        except Exception as e:
            st.error(f"查詢任務狀態時發生錯誤：{str(e)}")
            st.markdown("---")
            st.markdown("💡 **可能的解決方案：**")
            st.markdown("1. 確認任務 ID 是否正確")
            st.markdown("2. 檢查網路連接")
            st.markdown("3. 稍後再試")


def main():
    """主函數"""
    st.title("LoRA 訓練任務管理")

    # 檢查登入狀態
    if not check_auth():
        return

    # 添加頁籤
    tab1, tab2, tab3 = st.tabs(["提交任務", "追蹤進度", "實驗記錄"])

    # 提交任務頁籤
    with tab1:
        render_task_form()

    # 追蹤進度頁籤
    with tab2:
        render_task_progress()

    # 實驗記錄頁籤
    with tab3:
        render_mlflow_experiments()


def render_mlflow_experiments():
    """渲染實驗記錄頁面"""
    st.title("📊 實驗記錄")

    # 檢查登入狀態
    if "jwt_token" not in st.session_state:
        st.warning("請先登入後再查看實驗資料")
        return

    # API endpoints
    API_BASE = API_URL  # 使用環境變數中的 API URL
    # 使用環境變數或預設值，並確保使用正確的端口
    MLFLOW_UI = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")
    # 如果是容器內部的 URL，轉換為外部訪問的 URL
    if MLFLOW_UI == "http://mlflow:5000":
        MLFLOW_UI = "http://localhost:5001"

    def format_time(timestamp):
        """Format timestamp to readable datetime."""
        if not timestamp:
            return "-"
        try:
            # 嘗試解析 ISO 格式時間字符串
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            try:
                # 如果不是 ISO 格式，嘗試作為數字時間戳處理
                ts = float(timestamp)
                return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                return "-"

    def format_duration(start_time, end_time):
        """Calculate and format duration between timestamps."""
        if not (start_time and end_time):
            return "-"
        try:
            # 嘗試解析 ISO 格式時間字符串
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration = (end_dt - start_dt).total_seconds()
        except (ValueError, AttributeError):
            try:
                # 如果不是 ISO 格式，嘗試作為數字時間戳處理
                start_ts = float(start_time)
                end_ts = float(end_time)
                duration = (end_ts - start_ts) / 1000
            except (ValueError, TypeError):
                return "-"

        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration / 60:.1f}m"
        else:
            return f"{duration / 3600:.1f}h"

    # Sidebar filters
    st.sidebar.title("篩選器")
    status_filter = st.sidebar.selectbox(
        "Run 狀態", ["FINISHED", "RUNNING", "FAILED", "SCHEDULED", "ALL"], index=0
    )
    limit = st.sidebar.number_input("Run 數量", min_value=1, max_value=100, value=10)

    # Fetch experiment runs
    try:
        params = {"limit": limit}
        if status_filter != "ALL":
            params["status"] = status_filter

        # 從 session state 獲取 token
        token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        response = requests.get(
            f"{API_BASE}/mlflow",
            params=params,
            headers=headers,
            timeout=15,  # 增加超時時間到 15 秒
        )

        if response.status_code == 200:
            data = response.json()

            # 移除調試信息顯示

            if not data.get("runs") or len(data["runs"]) == 0:
                st.info("📊 目前沒有找到任何 MLflow 實驗 runs。")
                st.markdown("---")
                st.markdown("💡 **提示：**")
                st.markdown("- 請先執行一些訓練任務來產生實驗資料")
                st.markdown("- 確保 MLflow 服務正在運行")
                st.markdown(f"- 查看 [MLflow UI]({MLFLOW_UI}) 確認實驗資料")
                return
        else:
            st.warning(f"⚠️ API 回應錯誤 (狀態碼: {response.status_code})")
            st.info("📊 目前無法獲取 MLflow 實驗資料。")
            st.markdown("---")
            st.markdown("💡 **可能的原因：**")
            st.markdown("- MLflow 服務可能正在啟動中")
            st.markdown("- API 服務可能需要重新啟動")
            st.markdown(f"- 請檢查 [MLflow UI]({MLFLOW_UI}) 是否可訪問")
            return
    except requests.exceptions.ConnectionError:
        st.error("❌ 無法連接到 API 服務")
        st.info("📊 目前無法獲取 MLflow 實驗資料。")
        st.markdown("---")
        st.markdown("💡 **解決方案：**")
        st.markdown("- 請確認 API 服務正在運行")
        st.markdown("- 嘗試重新啟動服務: `docker-compose restart api`")
        st.markdown(f"- 檢查 [MLflow UI]({MLFLOW_UI}) 是否可訪問")
        return
    except requests.exceptions.Timeout:
        st.warning("⏰ API 請求超時")
        st.info("📊 目前無法獲取 MLflow 實驗資料。")
        st.markdown("---")
        st.markdown("💡 **請稍後再試或檢查服務狀態**")
        return
    except Exception as e:
        st.error(f"❌ 發生未預期的錯誤: {str(e)}")
        st.info("📊 目前無法獲取 MLflow 實驗資料。")
        st.markdown("---")
        st.markdown("💡 **請檢查服務日誌或聯繫管理員**")
        return

    # 如果成功獲取資料，顯示實驗列表
    if data.get("runs") and len(data["runs"]) > 0:
        # Create DataFrame for better display
        runs_data = []
        for run in data["runs"]:
            runs_data.append(
                {
                    "Run ID": f"[{run.get('run_id', '-')}]({MLFLOW_UI}/#/experiments/{data.get('experiment_id', '1')}/runs/{run.get('run_id', '-')})",
                    "狀態": run.get("status", "-"),
                    "開始時間": format_time(run.get("start_time")),
                    "持續時間": format_duration(
                        run.get("start_time"), run.get("end_time")
                    ),
                    "模型": run.get("params", {}).get("model_name", "-"),
                    "準確率": f"{float(run.get('metrics', {}).get('eval_accuracy') or 0):.4f}",
                    "訓練時間": f"{float(run.get('metrics', {}).get('training_time') or 0):.1f}s",
                    "批次大小": run.get("params", {}).get("batch_size", "-"),
                    "學習率": run.get("params", {}).get("learning_rate", "-"),
                }
            )

        df = pd.DataFrame(runs_data)

        # Display metrics summary
        st.subheader("📈 最近實驗")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_acc = pd.to_numeric(df["準確率"], errors="coerce").mean()
            st.metric("平均準確率", f"{avg_acc:.4f}")

        with col2:
            success_rate = (df["狀態"] == "FINISHED").mean() * 100
            st.metric("成功率", f"{success_rate:.1f}%")

        with col3:
            total_runs = len(df)
            st.metric("總 Runs", total_runs)

        # Display runs table
        st.dataframe(
            df,
            column_config={
                "Run ID": st.column_config.Column(
                    "Run ID",
                    help="點擊查看 MLflow UI 中的詳細資訊",
                    width="small",
                ),
                "狀態": st.column_config.Column(
                    "狀態",
                    help="Run 狀態",
                    width="small",
                ),
                "準確率": st.column_config.NumberColumn(
                    "準確率",
                    help="評估準確率",
                    format="%.4f",
                    width="small",
                ),
            },
            hide_index=True,
        )

        # Add MLflow UI link
        st.markdown("---")
        st.markdown(
            f"💡 **提示：** 點擊 Run ID 可以在 MLflow UI 中查看詳細的指標和 artifacts，"
            f"或者直接訪問 [MLflow UI]({MLFLOW_UI})"
        )


if __name__ == "__main__":
    main()
