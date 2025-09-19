"""審計日誌 API 路由"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.auth.jwt_utils import check_admin
from app.monitor.audit import get_audit_logs

router = APIRouter(prefix="/audit", tags=["Audit Logs"])


class AuditLogEntry(BaseModel):
    """審計日誌條目"""

    id: int
    user_id: str
    role: str
    action: str
    method: str
    path: str
    status_code: int
    timestamp: int


@router.get("/logs", response_model=List[AuditLogEntry])
async def list_audit_logs(
    user: Dict = Depends(check_admin),  # 只有管理員可以查詢日誌
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = Query(100, ge=1, le=1000),
) -> List[AuditLogEntry]:
    """查詢審計日誌
    支援篩選條件：user_id、role、start_time、end_time、limit
    """
    logs = get_audit_logs(user_id, role, start_time, end_time, limit)
    return logs
