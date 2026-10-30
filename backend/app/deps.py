"""FastAPI 依赖：数据库、JWT 用户、管理员权限。"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .db import get_db
from .models import AppUser
from .security import decode_access_token

bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str = "未登录或登录已过期") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> AppUser:
    if credentials is None:
        raise _unauthorized()
    try:
        payload = decode_access_token(credentials.credentials)
    except Exception:
        raise _unauthorized("Token 无效或已过期")
    username = payload.get("sub")
    if not username:
        raise _unauthorized("Token 缺少用户名")
    user = db.query(AppUser).filter(AppUser.username == username, AppUser.enabled.is_(True)).first()
    if not user:
        raise _unauthorized("用户不存在或已禁用")
    return user


def require_admin(user: AppUser = Depends(get_current_user)) -> AppUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可执行此操作")
    return user
