"""JWT 工具模組

提供 JWT Token 的生成與驗證功能。
"""

import os
import time
from typing import Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

# 配置
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set.")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer(auto_error=False)


def create_token(user_id: str, role: str) -> str:
    """生成 JWT Token

    Args:
        user_id: 使用者 ID
        role: 使用者角色 ("admin" 或 "user")

    Returns:
        str: JWT Token
    """
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": int(time.time()) + TOKEN_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict:
    """解析 JWT Token

    Args:
        token: JWT Token

    Returns:
        Dict: Token payload

    Raises:
        HTTPException: Token 無效或過期
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload["exp"] < time.time():
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Token 已過期",
            )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Token 無效",
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict:
    """FastAPI dependency，用於獲取當前用戶信息

    Args:
        credentials: HTTP Authorization header

    Returns:
        Dict: 用戶信息，包含 user_id 和 role
    """
    if credentials is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="未提供認證信息",
        )
    return decode_token(credentials.credentials)


def check_admin(user: Dict = Depends(get_current_user)) -> Dict:
    """FastAPI dependency，用於驗證管理員權限

    Args:
        user: 用戶信息

    Returns:
        Dict: 用戶信息

    Raises:
        HTTPException: 用戶不是管理員
    """
    if user["role"] != "admin":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="需要管理員權限",
        )
    return user


def check_task_owner(
    task_id: str, user: Dict = Depends(get_current_user)
) -> Optional[Dict]:
    """FastAPI dependency，用於驗證任務所有者

    Args:
        task_id: 任務 ID
        user: 用戶信息

    Returns:
        Optional[Dict]: 用戶信息，如果是管理員或任務所有者

    Raises:
        HTTPException: 用戶無權訪問該任務
    """
    # 管理員可以訪問所有任務
    if user["role"] == "admin":
        return user

    # 普通用戶只能訪問自己的任務
    # 這裡假設任務 ID 的前綴是用戶 ID
    if not task_id.startswith(user["user_id"]):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="無權訪問此任務",
        )

    return user
