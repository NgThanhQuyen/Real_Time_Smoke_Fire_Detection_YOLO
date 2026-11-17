from __future__ import annotations

import hashlib
import json

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from .config import INITIAL_MODEL_DIR, INITIAL_MODEL_RELATIVE
from .db import Base, engine
from .models import ModelVersion, RealtimeEvent, RealtimeNotificationLog, RealtimeSession, UserRole, VideoJob, utcnow


def migrate_feedback_sources() -> None:
    """Cho phép phản hồi tham chiếu đến video cũ hoặc một phiên webcam."""
    with engine.connect() as connection:
        columns = connection.exec_driver_sql("PRAGMA table_info(feedback)").all()
        if not columns:
            return
        names = {row[1] for row in columns}
        job_column = next(row for row in columns if row[1] == "job_id")
        if {"realtime_session_id", "realtime_event_id"}.issubset(names) and not job_column[3]:
            return

        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.exec_driver_sql(
                """CREATE TABLE feedback__new (
                    id VARCHAR(32) NOT NULL PRIMARY KEY,
                    job_id VARCHAR(32) REFERENCES video_jobs(id) ON DELETE CASCADE,
                    realtime_session_id VARCHAR(32) REFERENCES realtime_sessions(id) ON DELETE CASCADE,
                    realtime_event_id VARCHAR(32) REFERENCES realtime_events(id) ON DELETE SET NULL,
                    user_id VARCHAR(32) NOT NULL REFERENCES users(id),
                    detection_id VARCHAR(32) REFERENCES detections(id),
                    video_time_ms INTEGER NOT NULL,
                    image_path VARCHAR(500) NOT NULL,
                    type VARCHAR(30) NOT NULL,
                    expected_class INTEGER,
                    proposed_box JSON,
                    comment VARCHAR(1000),
                    status VARCHAR(20) NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT ck_feedback_one_source CHECK (
                        (job_id IS NOT NULL AND realtime_session_id IS NULL) OR
                        (job_id IS NULL AND realtime_session_id IS NOT NULL)
                    )
                )"""
            )
            old_columns = [
                "id", "job_id", "user_id", "detection_id", "video_time_ms", "image_path",
                "type", "expected_class", "proposed_box", "comment", "status", "created_at",
            ]
            copied = ", ".join(old_columns)
            connection.exec_driver_sql(
                f"INSERT INTO feedback__new ({copied}) SELECT {copied} FROM feedback"
            )
            connection.exec_driver_sql("DROP TABLE feedback")
            connection.exec_driver_sql("ALTER TABLE feedback__new RENAME TO feedback")
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_feedback_status_created ON feedback(status, created_at)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_feedback_realtime_session_id ON feedback(realtime_session_id)"
            )
            connection.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_feedback_realtime_event_id ON feedback(realtime_event_id)"
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.commit()


def initialize_database() -> None:
    from . import models

    Base.metadata.create_all(engine)
    migrate_feedback_sources()
    with Session(engine) as db:
        review_columns = {
            row[1] for row in db.execute(text("PRAGMA table_info(feedback_reviews)"))
        }
        if "annotation_status" not in review_columns:
            db.execute(text("ALTER TABLE feedback_reviews ADD COLUMN annotation_status VARCHAR(20)"))
        if "annotations" not in review_columns:
            db.execute(text("ALTER TABLE feedback_reviews ADD COLUMN annotations JSON"))
        combined = db.execute(
            select(UserRole.user_id)
            .where(UserRole.role.in_(["admin_ops", "admin_model"]))
            .group_by(UserRole.user_id)
            .having(text("count(*) = 2"))
        ).scalars().all()
        if combined:
            if len(combined) != 1 or db.scalar(select(UserRole.user_id).where(UserRole.role == "admin_model", UserRole.user_id != combined[0])):
                raise RuntimeError("Cần tách các tài khoản admin cũ thủ công trước khi khởi động")
            legacy = db.scalar(select(UserRole).where(UserRole.user_id == combined[0], UserRole.role == "admin_model"))
            db.delete(legacy)
            db.flush()
        db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_single_admin_ops ON user_roles(role) WHERE role = 'admin_ops'"))
        db.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_single_admin_model ON user_roles(role) WHERE role = 'admin_model'"))
        db.execute(
            update(VideoJob)
            .where(VideoJob.status.in_(["processing", "finalizing"]))
            .values(status="interrupted", error_code="worker_restarted")
        )
        db.execute(
            update(RealtimeSession)
            .where(RealtimeSession.status == "running")
            .values(status="interrupted", ended_at=utcnow(), error_code="backend_restarted")
        )
        db.execute(
            update(RealtimeEvent)
            .where(
                RealtimeEvent.status == "active",
                RealtimeEvent.session_id.in_(
                    select(RealtimeSession.id).where(RealtimeSession.status == "interrupted")
                ),
            )
            .values(status="closed")
        )
        db.execute(
            update(RealtimeNotificationLog)
            .where(RealtimeNotificationLog.status == "pending")
            .values(status="failed", error_code="backend_restarted")
        )
        if db.scalar(select(ModelVersion.id).limit(1)) is None:
            model_path = INITIAL_MODEL_DIR / "best.pt"
            if model_path.is_file():
                labels = json.loads((INITIAL_MODEL_DIR / "labelmap.json").read_text(encoding="utf-8"))
                summary = json.loads((INITIAL_MODEL_DIR / "run_summary.json").read_text(encoding="utf-8"))
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                db.add(
                    ModelVersion(
                        name=summary.get("run_id", "yolo11n-initial"),
                        artifact_path=INITIAL_MODEL_RELATIVE,
                        sha256=digest,
                        class_map=labels,
                        metrics=summary.get("metrics", {}),
                        config={
                            "imgsz": summary.get("train_config", {}).get("imgsz", 832),
                            "iou": summary.get("iou_threshold_used", 0.6),
                        },
                        status="active",
                    )
                )
        else:
            model_path = INITIAL_MODEL_DIR / "best.pt"
            if model_path.is_file():
                previous = db.scalar(
                    select(ModelVersion).where(ModelVersion.artifact_path == "best.pt").limit(1)
                )
                if previous and previous.sha256 == hashlib.sha256(model_path.read_bytes()).hexdigest():
                    previous.artifact_path = INITIAL_MODEL_RELATIVE
        db.commit()
