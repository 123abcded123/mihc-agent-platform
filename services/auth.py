"""
登录鉴权服务（无外部依赖：pbkdf2 密码哈希 + HMAC 签名 token）

- 密码永不落盘明文；
- token = base64url(payload) + "." + HMAC-SHA256 签名，payload 含用户信息与过期时间；
- 生产环境必须通过 MIHC_AUTH_SECRET 设置强密钥，否则仅用开发默认值。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException

from core.errors import PermissionDeniedError

logger = logging.getLogger(__name__)

PBKDF2_ITERATIONS = 120_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations))
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, AttributeError):
        return False


class TokenService:
    """签发与校验访问 token。"""

    def __init__(self, secret: str, ttl_sec: int = 86400):
        self.secret = secret.encode("utf-8")
        self.ttl_sec = ttl_sec

    @staticmethod
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def _b64url_decode(text: str) -> bytes:
        padding = "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(text + padding)

    def issue(self, user: dict) -> str:
        payload = {
            "sub": user["username"],
            "tenant_id": user["tenant_id"],
            "role": user.get("role", "customer"),
            "exp": int(time.time()) + self.ttl_sec,
        }
        body = self._b64url(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        signature = hmac.new(self.secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{body}.{signature}"

    def verify(self, token: str) -> Optional[dict]:
        try:
            body, signature = token.rsplit(".", 1)
            expected = hmac.new(self.secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                return None
            payload = json.loads(self._b64url_decode(body))
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except (ValueError, json.JSONDecodeError):
            return None


def get_token_service(config) -> TokenService:
    return TokenService(config.auth.secret, config.auth.token_ttl_sec)


def require_user(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI 依赖：从 Authorization: Bearer <token> 解析当前用户。"""
    from app import platform_service

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, detail={"error": {"code": "unauthorized", "message": "请先登录"}})
    payload = platform_service.tokens.verify(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(401, detail={"error": {"code": "token_invalid", "message": "登录已过期，请重新登录"}})
    return payload


def bootstrap_admin(db, config) -> None:
    """启动时按环境变量创建管理员（幂等）。"""
    if not (config.auth.admin_user and config.auth.admin_password):
        return
    if db.get_user(config.auth.admin_user):
        return
    db.create_user(username=config.auth.admin_user,
                   password_hash=hash_password(config.auth.admin_password),
                   tenant_id=config.platform.tenant_id, role="admin",
                   display_name=config.auth.admin_user)
    logger.info("Admin user '%s' bootstrapped", config.auth.admin_user)


def ensure_allowed(requester: dict, tenant_id: str = "mihc") -> None:
    """租户边界：token 中的租户必须与资源租户一致（admin 除外）。"""
    if requester.get("role") == "admin":
        return
    if requester.get("tenant_id") != tenant_id:
        raise PermissionDeniedError("不能访问其他租户的资源")
