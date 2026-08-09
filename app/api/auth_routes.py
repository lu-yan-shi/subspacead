"""Authentication routes — login / me / change-password.

独立 router（prefix=/api/auth），不带 require_auth 依赖，因此业务路由整体上锁时
登录端点仍可访问。me / change-password 在端点级校验 token。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import (
    create_token,
    get_user,
    update_password,
    verify_password,
)
from ..config import AUTH_TOKEN_EXPIRE_HOURS
from .security import require_auth

auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@auth_router.post("/login", summary="用户登录")
async def login(req: LoginRequest):
    user = get_user(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {
        "token": create_token(user["username"]),
        "username": user["username"],
        "expires_in": AUTH_TOKEN_EXPIRE_HOURS * 3600,
    }


@auth_router.get("/me", summary="当前登录用户")
async def me(auth: dict = Depends(require_auth)):
    return {"username": auth["username"]}


@auth_router.post("/change-password", summary="修改密码")
async def change_password(
    req: ChangePasswordRequest,
    auth: dict = Depends(require_auth),
):
    username = auth["username"]
    user = get_user(username)
    if not user or not verify_password(req.old_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="原密码不正确")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 位")
    update_password(username, req.new_password)
    return {"ok": True}
