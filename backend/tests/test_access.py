from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db import Base, get_db
from backend.app.config import INITIAL_MODEL_RELATIVE
from backend.app.main import app
from backend.app.models import ModelVersion, User, UserProfile, UserRole, VideoJob
from backend.app.security import hash_password


def test_user_cannot_read_another_users_video_or_admin_feed(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'access.sqlite').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def test_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = test_db
    try:
        with Session(engine) as db:
            model = ModelVersion(name="test-model", artifact_path=INITIAL_MODEL_RELATIVE, sha256="x" * 64, class_map={"0": "khoi", "1": "lua"}, status="active")
            owner = User(username="owner", password_hash=hash_password("password-123"))
            db.add_all([model, owner])
            db.flush()
            db.add_all([UserProfile(user_id=owner.id), UserRole(user_id=owner.id, role="user")])
            job = VideoJob(
                user_id=owner.id, model_version_id=model.id, original_name="owner.mp4",
                source_path="uploads/owner.mp4", duration_ms=1000, source_fps=10,
                total_frames=10, status="completed",
            )
            db.add(job)
            db.commit()
            owned_id = job.id
        client = TestClient(app)
        response = client.post("/api/auth/register", json={"username": "visitor", "password": "password-123", "password_confirm": "password-123"})
        assert response.status_code == 201
        assert client.get(f"/api/videos/{owned_id}").status_code == 404
        assert client.delete(f"/api/videos/{owned_id}").status_code == 404
        assert client.get("/api/admin/events").status_code == 403
        assert client.get("/api/model-admin/feedback").status_code == 403
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
