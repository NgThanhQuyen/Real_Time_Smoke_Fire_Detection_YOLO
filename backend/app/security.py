from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import RUNTIME, settings
from .db import get_db
from .models import AuthSession, User, utcnow


password_hasher = PasswordHasher()
SESSION_COOKIE = "fire_alert_session"
SESSION_DAYS = 7


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, ValueError):
        return False


def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(48)
    db.add(
        AuthSession(
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=utcnow() + timedelta(days=SESSION_DAYS),
        )
    )
    db.commit()
    return token


def remove_session(db: Session, token: str | None) -> None:
    if token:
        row = db.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
        if row:
            db.delete(row)
            db.commit()


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Vui lòng đăng nhập")
    session = db.get(AuthSession, hashlib.sha256(token.encode()).hexdigest())
    if not session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Phiên đăng nhập không hợp lệ")
    expires = session.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= utcnow():
        db.delete(session)
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Phiên đăng nhập đã hết hạn")
    user = db.get(User, session.user_id)
    if not user or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Tài khoản không hoạt động")
    return user


def require_role(role: str):
    def dependency(user: User = Depends(current_user)) -> User:
        if role not in {item.role for item in user.roles}:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Bạn không có quyền thực hiện")
        return user

    return dependency


def require_user(user: User = Depends(current_user)) -> User:
    roles = {item.role for item in user.roles}
    if "user" not in roles or roles.intersection({"admin_ops", "admin_model"}):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tài khoản này không có quyền dự đoán video")
    return user


def cipher() -> Fernet:
    if settings.app_encryption_key:
        return Fernet(settings.app_encryption_key.encode())
    key_file = RUNTIME / "app.key"
    if not key_file.exists():
        key_file.write_bytes(Fernet.generate_key())
    return Fernet(key_file.read_bytes())


def encrypt_secret(value: str) -> str:
    return cipher().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    return cipher().decrypt(value.encode()).decode()
