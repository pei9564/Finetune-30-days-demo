"""
認證與授權相關測試
"""

import time
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth.jwt_utils import (
    JWT_ALGORITHM,
    JWT_SECRET,
    TOKEN_EXPIRE_MINUTES,
    check_admin,
    check_task_owner,
    create_token,
    decode_token,
    get_current_user,
)


class TestJWTAuth:
    """JWT 認證相關測試"""

    def test_create_token(self):
        """測試 Token 生成

        測試場景：
        - 生成的 Token 應該包含正確的 payload
        - Token 應該可以被正確解碼
        - 過期時間應該正確設置
        """
        # 生成 Token
        user_id = "test_user"
        role = "admin"
        current_time = time.time()

        with patch("time.time", return_value=current_time):
            token = create_token(user_id, role)

            # 解碼並驗證
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            assert payload["user_id"] == user_id
            assert payload["role"] == role
            assert payload["exp"] == int(current_time) + TOKEN_EXPIRE_MINUTES * 60

    def test_create_token_different_roles(self):
        """測試不同角色的 Token 生成

        測試場景：
        - admin 角色的 Token
        - user 角色的 Token
        - 自定義角色的 Token
        """
        test_cases = [
            {"user_id": "admin1", "role": "admin"},
            {"user_id": "user1", "role": "user"},
            {"user_id": "guest1", "role": "guest"},
        ]

        for case in test_cases:
            token = create_token(case["user_id"], case["role"])
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            assert payload["user_id"] == case["user_id"]
            assert payload["role"] == case["role"]

    def test_decode_token(self):
        """測試 Token 解碼

        測試場景：
        - 有效的 Token 應該能被正確解碼
        - 過期的 Token 應該拋出異常
        - 無效的 Token 應該拋出異常
        """
        # 測試有效 Token
        token = create_token("test_user", "user")
        payload = decode_token(token)
        assert payload["user_id"] == "test_user"
        assert payload["role"] == "user"

        # 測試過期 Token
        current_time = time.time()
        with patch("time.time", return_value=current_time):
            token = create_token("test_user", "user")

        # 模擬時間前進超過過期時間
        with patch(
            "time.time", return_value=current_time + TOKEN_EXPIRE_MINUTES * 60 + 1
        ):
            with pytest.raises(HTTPException) as exc_info:
                decode_token(token)
            assert exc_info.value.status_code == 401
            assert "Token 已過期" in exc_info.value.detail

        # 測試無效 Token
        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid_token")
        assert exc_info.value.status_code == 401
        assert "Token 無效" in exc_info.value.detail

    def test_check_admin(self):
        """測試管理員權限檢查

        測試場景：
        - admin 用戶應該通過檢查
        - 普通用戶應該被拒絕
        """
        # 模擬管理員
        admin_user = {"user_id": "admin", "role": "admin"}
        result = check_admin(admin_user)
        assert result == admin_user

        # 模擬普通用戶
        normal_user = {"user_id": "user1", "role": "user"}
        with pytest.raises(HTTPException) as exc_info:
            check_admin(normal_user)
        assert exc_info.value.status_code == 403
        assert "管理員權限" in exc_info.value.detail

    def test_get_current_user(self):
        """測試獲取當前用戶

        測試場景：
        - 有效的認證信息應該返回用戶
        - 無認證信息應該拋出 401
        - 無效的 token 應該拋出 401
        - 過期的 token 應該拋出 401
        """
        # 測試有效認證
        valid_token = create_token("test_user", "admin")
        valid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=valid_token
        )
        user = get_current_user(valid_credentials)
        assert user["user_id"] == "test_user"
        assert user["role"] == "admin"

        # 測試無認證信息
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None)
        assert exc_info.value.status_code == 401
        assert "未提供認證信息" in exc_info.value.detail

        # 測試無效 token
        invalid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_token"
        )
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(invalid_credentials)
        assert exc_info.value.status_code == 401
        assert "Token 無效" in exc_info.value.detail

        # 測試過期 token
        current_time = time.time()
        with patch("time.time", return_value=current_time):
            expired_token = create_token("test_user", "admin")

        # 模擬時間前進超過過期時間
        with patch(
            "time.time", return_value=current_time + TOKEN_EXPIRE_MINUTES * 60 + 1
        ):
            expired_credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=expired_token
            )
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(expired_credentials)
            assert exc_info.value.status_code == 401
            assert "Token 已過期" in exc_info.value.detail

    def test_check_task_owner(self):
        """測試任務所有者權限檢查

        測試場景：
        - admin 可以訪問任何任務
        - 用戶只能訪問自己的任務
        - 用戶訪問其他人的任務應該被拒絕
        - 任務 ID 格式不正確時的處理
        """
        # 模擬管理員訪問任務
        admin_user = {"user_id": "admin", "role": "admin"}
        result = check_task_owner("user1_task1", admin_user)
        assert result == admin_user

        # 模擬用戶訪問自己的任務
        user = {"user_id": "user1", "role": "user"}
        result = check_task_owner("user1_task1", user)
        assert result == user

        # 模擬用戶訪問其他人的任務
        with pytest.raises(HTTPException) as exc_info:
            check_task_owner("user2_task1", user)
        assert exc_info.value.status_code == 403
        assert "無權訪問" in exc_info.value.detail

        # 測試任務 ID 格式不正確
        with pytest.raises(HTTPException) as exc_info:
            check_task_owner("invalid_task_id", user)
        assert exc_info.value.status_code == 403
        assert "無權訪問" in exc_info.value.detail
