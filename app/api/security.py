"""FastAPI 鉴权依赖 — 业务路由统一校验 Bearer token。

独立于此模块：auth.py 保持纯 stdlib（便于无 fastapi 环境本地单测），
需要 fastapi 的依赖放这里。
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from ..auth import verify_token


def require_auth(request: Request) -> dict:
    """校验 Authorization: Bearer <token>，失败抛 401。返回 {"username": ...}。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        username = verify_token(auth[len("Bearer "):])
        if username:
            return {"username": username}
    raise HTTPException(status_code=401, detail="未登录或登录已过期")
