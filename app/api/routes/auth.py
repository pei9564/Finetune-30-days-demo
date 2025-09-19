"""
認證相關路由
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth.jwt_utils import create_token

router = APIRouter(tags=["Authentication"])


class LoginRequest(BaseModel):
    """登入請求模型"""

    username: str
    password: str


class LoginResponse(BaseModel):
    """登入響應模型"""

    token: str
    user_id: str
    role: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """用戶登入

    Args:
        request: 包含用戶名和密碼的請求

    Returns:
        LoginResponse: 包含 token 和用戶信息的響應

    Raises:
        HTTPException: 當認證失敗時
    """
    # 這裡應該實現真實的用戶認證邏輯
    # 目前僅作為示例：admin/admin 為管理員，其他為普通用戶
    if request.username == "admin" and request.password == "admin":
        role = "admin"
    else:
        role = "user"

    # 生成 token
    user_id = request.username  # 在實際應用中應該使用真實的用戶 ID
    token = create_token(user_id, role)

    return LoginResponse(token=token, user_id=user_id, role=role)
