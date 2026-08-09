"""Authentication core — SQLite user store + PBKDF2 password hashing + HMAC stateless tokens.

纯 stdlib 实现（sqlite3 / hashlib / hmac / secrets），不引入第三方依赖：
- 账号密码落在 SQLite `users` 表，默认管理员首次启动自动创建。
- 密码用 PBKDF2-HMAC-SHA256（随机 salt + 200k 迭代）哈希存储，绝不存明文。
- 登录签发 HMAC 签名无状态 token（默认 24h 过期），无需服务端会话表。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from typing import Optional

from .config import (
    AUTH_ADMIN_PASS,
    AUTH_ADMIN_USER,
    AUTH_DB_PATH,
    AUTH_SECRET,
    AUTH_TOKEN_EXPIRE_HOURS,
)

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000
_TOKEN_TTL = AUTH_TOKEN_EXPIRE_HOURS * 3600


# ---------------------------------------------------------------
# 密码哈希（PBKDF2-HMAC-SHA256）
# 存储格式: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
# ---------------------------------------------------------------
def hash_password(password: str) -> str:
    """对密码做加盐 PBKDF2 哈希，返回自描述的存储串。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256$%d$%s$%s" % (_PBKDF2_ITERATIONS, salt.hex(), dk.hex())


def verify_password(password: str, stored: str) -> bool:
    """常数时间比对：校验密码是否匹配存储串。格式不符/异常一律返回 False。"""
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ---------------------------------------------------------------
# Token（HMAC-SHA256 无状态签名，格式 payload.signature）
# ---------------------------------------------------------------
def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def create_token(username: str) -> str:
    """签发 24h 有效期的签名 token。"""
    payload = _b64url(json.dumps({
        "user": username,
        "exp": int(time.time()) + _TOKEN_TTL,
    }).encode("utf-8"))
    sig = _b64url(hmac.new(
        AUTH_SECRET.encode("utf-8"), payload.encode("ascii"), hashlib.sha256,
    ).digest())
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[str]:
    """校验签名与过期时间，返回用户名；非法/过期返回 None。"""
    try:
        payload_b64, sig_b64 = token.split(".")
        expected = _b64url(hmac.new(
            AUTH_SECRET.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256,
        ).digest())
        if not hmac.compare_digest(sig_b64, expected):
            return None
        data = json.loads(base64.urlsafe_b64decode(payload_b64 + "==").decode("utf-8"))
        if int(data.get("exp", 0)) < time.time():
            return None
        return data.get("user")
    except Exception:
        return None


# ---------------------------------------------------------------
# SQLite 用户存储
# ---------------------------------------------------------------
def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or AUTH_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_auth(db_path: Optional[str] = None) -> None:
    """创建 users 表；首次启动自动创建默认管理员。幂等，可重复调用。"""
    conn = _connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"""
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM users WHERE username=?", (AUTH_ADMIN_USER,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (AUTH_ADMIN_USER, hash_password(AUTH_ADMIN_PASS)),
            )
            conn.commit()
            logger.warning(
                "已创建初始账号「%s」（默认密码，请登录后尽快修改）。"
                "可用环境变量 AUTH_ADMIN_USER / AUTH_ADMIN_PASS 覆盖默认凭据。",
                AUTH_ADMIN_USER,
            )
    finally:
        conn.close()


def get_user(username: str, db_path: Optional[str] = None) -> Optional[dict]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_password(username: str, new_password: str, db_path: Optional[str] = None) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (hash_password(new_password), username),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
