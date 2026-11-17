from pathlib import Path

import pytest
from fastapi import HTTPException, Request, Response
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.api.auth import set_session_cookie
from backend.app.db import Base
from backend.app.main import clear_auth_sessions
from backend.app.models import AuthSession, User
from backend.app.security import SESSION_COOKIE, create_session, current_user


def test_backend_restart_invalidates_saved_sessions(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'sessions.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            user = User(username="owner", password_hash="unused")
            db.add(user)
            db.commit()
            token = create_session(db, user)
            request = Request({"type": "http", "headers": [(b"cookie", f"{SESSION_COOKIE}={token}".encode())]})
            assert current_user(request, db).id == user.id
            clear_auth_sessions(db)
            db.commit()

            assert db.scalar(select(AuthSession.token_hash)) is None
            with pytest.raises(HTTPException) as error:
                current_user(request, db)
            assert error.value.status_code == 401
    finally:
        engine.dispose()


def test_session_cookie_is_not_persistent():
    response = Response()
    set_session_cookie(response, "sample-token")
    cookie = response.headers["set-cookie"]
    assert "httponly" in cookie.lower()
    assert "max-age" not in cookie.lower()
    assert "expires" not in cookie.lower()
