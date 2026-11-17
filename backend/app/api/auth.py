from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, UserProfile, UserRole
from ..security import (
    SESSION_COOKIE,
    create_session,
    current_user,
    hash_password,
    remove_session,
    verify_password,
)
from ..serializers import user_data
from ..config import settings
from .schemas import LoginInput, RegisterInput


router = APIRouter(prefix="/api/auth", tags=["auth"])


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=settings.app_origin.startswith("https://"),
        samesite="lax",
        path="/",
    )


@router.post("/register", status_code=201)
def register(payload: RegisterInput, response: Response, db: Session = Depends(get_db)):
    username = payload.username.lower()
    if db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Tên tài khoản đã được sử dụng")
    user = User(username=username, password_hash=hash_password(payload.password))
    db.add(user)
    db.flush()
    db.add_all([UserRole(user_id=user.id, role="user"), UserProfile(user_id=user.id)])
    db.commit()
    db.refresh(user)
    set_session_cookie(response, create_session(db, user))
    return user_data(user)


@router.post("/login")
def login(payload: LoginInput, response: Response, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == payload.username.lower()))
    if not user or not user.active or not verify_password(user.password_hash, payload.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Tài khoản hoặc mật khẩu không đúng")
    set_session_cookie(response, create_session(db, user))
    return user_data(user)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    remove_session(db, request.cookies.get(SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_data(user)
