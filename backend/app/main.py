from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .api.admin import model_admin, ops
from .api.auth import router as auth_router
from .api.realtime import router as realtime_router
from .api.user import router as user_router
from .bootstrap import initialize_database
from .config import settings
from .db import SessionLocal
from .models import AuthSession


def clear_auth_sessions(db: Session) -> None:
    db.execute(delete(AuthSession))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    with SessionLocal.begin() as db:
        clear_auth_sessions(db)
    yield


app = FastAPI(title="Fire Alert YOLO11", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.app_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def check_origin(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin not in {settings.app_origin, settings.api_origin}:
            return JSONResponse(status_code=403, content={"detail": "Nguồn yêu cầu không hợp lệ"})
    return await call_next(request)


app.include_router(auth_router)
app.include_router(user_router)
app.include_router(realtime_router)
app.include_router(ops)
app.include_router(model_admin)


@app.get("/api/health")
def health():
    return {"status": "ok"}
