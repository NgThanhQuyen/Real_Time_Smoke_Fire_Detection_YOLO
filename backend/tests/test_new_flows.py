from datetime import date, datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api import user as user_api
from backend.app.api.admin import admin_job_detail, list_events
from backend.app.db import Base, get_db
from backend.app.inference.video import output_size
from backend.app.main import app
from backend.app.models import (
    AlertEvent,
    Feedback,
    ModelVersion,
    NotificationLog,
    User,
    UserProfile,
    UserRole,
    VideoJob,
)
from backend.app.security import hash_password
from backend.app.zalo import notification_text


def test_video_size_and_zalo_time_do_not_require_windows_tzdata():
    assert output_size(2160, 3840) == (720, 1280)
    assert output_size(1920, 1080) == (1280, 720)
    job = VideoJob(original_name="clip.mp4", source_path="uploads/clip.mp4", duration_ms=1000, source_fps=10, total_frames=10)
    event_item = AlertEvent(class_id=1, start_ms=500, end_ms=900, peak_confidence=0.8)
    assert "Việt Nam" in notification_text(event_item, job)


def test_registration_confirmation_and_cascading_user_deletion(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.sqlite').as_posix()}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def test_db():
        with session_factory() as db:
            yield db

    monkeypatch.setattr(user_api, "runtime_file", lambda relative: tmp_path / relative)
    monkeypatch.setattr(user_api, "RUNTIME", tmp_path)
    app.dependency_overrides[get_db] = test_db
    try:
        with Session(engine) as db:
            model = ModelVersion(name="test", artifact_path="unused", sha256="x" * 64, class_map={"0": "khoi", "1": "lua"}, status="active")
            user = User(username="owner", password_hash=hash_password("password-123"))
            db.add_all([model, user])
            db.flush()
            db.add_all([UserProfile(user_id=user.id), UserRole(user_id=user.id, role="user")])
            job = VideoJob(user_id=user.id, model_version_id=model.id, original_name="clip.mp4", source_path="uploads/clip.mp4", output_path="rendered/clip.mp4", duration_ms=1000, source_fps=10, total_frames=10, status="completed")
            db.add(job)
            db.flush()
            alert = AlertEvent(job_id=job.id, class_id=1, start_ms=0, end_ms=900, peak_confidence=0.8, contact_snapshot={})
            report = Feedback(job_id=job.id, user_id=user.id, video_time_ms=0, image_path="feedback/report.jpg", type="false_negative")
            extra_report = Feedback(job_id=job.id, user_id=user.id, video_time_ms=500, image_path="feedback/extra.jpg", type="false_negative")
            db.add_all([alert, report, extra_report])
            db.flush()
            db.add(NotificationLog(event_id=alert.id, recipient_user_id=user.id, status="pending"))
            db.commit()
            job_id = job.id
            extra_report_id = extra_report.id
        for relative in ("uploads/clip.mp4", "rendered/clip.mp4", "feedback/report.jpg", "feedback/extra.jpg"):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
        client = TestClient(app)
        mismatch = client.post("/api/auth/register", json={"username": "newuser", "password": "password-123", "password_confirm": "different-123"})
        assert mismatch.status_code == 422
        assert client.post("/api/auth/login", json={"username": "owner", "password": "password-123"}).status_code == 200
        assert client.delete(f"/api/feedback/{extra_report_id}").status_code == 200
        assert not (tmp_path / "feedback/extra.jpg").exists()
        assert client.delete(f"/api/videos/{job_id}").status_code == 200
        assert client.get(f"/api/videos/{job_id}").status_code == 404
        with Session(engine) as db:
            assert db.scalar(select(func.count()).select_from(Feedback)) == 0
            assert db.scalar(select(func.count()).select_from(AlertEvent)) == 0
            assert db.scalar(select(func.count()).select_from(NotificationLog)) == 0
        assert not (tmp_path / "uploads/clip.mp4").exists()
        assert not (tmp_path / "feedback/report.jpg").exists()
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_admin_event_date_filter_uses_vietnam_day(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'dates.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            model = ModelVersion(name="test", artifact_path="unused", sha256="x" * 64, class_map={"0": "khoi", "1": "lua"}, status="active")
            admin = User(username="ops", password_hash="unused")
            owner = User(username="owner", password_hash="unused")
            db.add_all([model, admin, owner])
            db.flush()
            db.add_all([UserProfile(user_id=owner.id), UserRole(user_id=admin.id, role="admin_ops")])
            job = VideoJob(user_id=owner.id, model_version_id=model.id, original_name="clip.mp4", source_path="uploads/clip.mp4", duration_ms=1000, source_fps=10, total_frames=10, status="completed")
            db.add(job)
            db.flush()
            db.add_all([
                AlertEvent(job_id=job.id, class_id=1, start_ms=0, end_ms=100, peak_confidence=0.8, contact_snapshot={}, created_at=datetime(2026, 10, 15, 16, 59, tzinfo=timezone.utc)),
                AlertEvent(job_id=job.id, class_id=1, start_ms=200, end_ms=300, peak_confidence=0.8, contact_snapshot={}, created_at=datetime(2026, 10, 15, 17, 1, tzinfo=timezone.utc)),
            ])
            db.commit()
            filtered = list_events(date_from=date(2026, 10, 16), date_to=date(2026, 10, 16), _actor=admin, db=db)
            assert len(filtered) == 1
            assert filtered[0]["start_ms"] == 200
            detail = admin_job_detail(job.id, _actor=admin, db=db)
            assert detail["original_name"] == "clip.mp4"
            assert len(detail["events"]) == 2
    finally:
        engine.dispose()


