from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid4().hex


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    roles: Mapped[list[UserRole]] = relationship(back_populates="user", cascade="all, delete-orphan")
    profile: Mapped[UserProfile | None] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")


class UserRole(Base):
    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))
    user: Mapped[User] = relationship(back_populates="roles")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    address: Mapped[str | None] = mapped_column(String(300), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    user: Mapped[User] = relationship(back_populates="profile")


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ZaloCredential(Base):
    __tablename__ = "zalo_credentials"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    token_ciphertext: Mapped[str] = mapped_column(Text)
    chat_id_ciphertext: Mapped[str] = mapped_column(Text)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    artifact_path: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    class_map: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VideoJob(Base):
    __tablename__ = "video_jobs"
    __table_args__ = (Index("ix_video_jobs_user_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    original_name: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(String(500))
    output_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer)
    source_fps: Mapped[float] = mapped_column(Float)
    total_frames: Mapped[int] = mapped_column(Integer)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    detected_frames: Mapped[int] = mapped_column(Integer, default=0)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RealtimeSession(Base):
    __tablename__ = "realtime_sessions"
    __table_args__ = (Index("ix_realtime_sessions_user_started", "user_id", "started_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    detected_frame_count: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)


class RealtimeEvent(Base):
    __tablename__ = "realtime_events"
    __table_args__ = (Index("ix_realtime_events_session_time", "session_id", "start_ms"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("realtime_sessions.id", ondelete="CASCADE"), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    peak_confidence: Mapped[float] = mapped_column(Float)
    detections: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class RealtimeEvidence(Base):
    __tablename__ = "realtime_evidence"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("realtime_events.id", ondelete="CASCADE"), index=True)
    image_path: Mapped[str] = mapped_column(String(500))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)


class RealtimeNotificationLog(Base):
    __tablename__ = "realtime_notification_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("realtime_events.id", ondelete="CASCADE"), index=True)
    channel: Mapped[str] = mapped_column(String(20), default="zalo")
    recipient_user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Detection(Base):
    __tablename__ = "detections"
    __table_args__ = (Index("ix_detections_job_time", "job_id", "video_time_ms"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("video_jobs.id", ondelete="CASCADE"))
    frame_index: Mapped[int] = mapped_column(Integer)
    video_time_ms: Mapped[int] = mapped_column(Integer)
    class_id: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    __table_args__ = (Index("ix_alert_events_job_time", "job_id", "start_ms"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("video_jobs.id", ondelete="CASCADE"))
    class_id: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    peak_confidence: Mapped[float] = mapped_column(Float)
    contact_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class EventEvidence(Base):
    __tablename__ = "event_evidence"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("alert_events.id", ondelete="CASCADE"))
    frame_index: Mapped[int] = mapped_column(Integer)
    video_time_ms: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str] = mapped_column(String(500))
    detection_id: Mapped[str | None] = mapped_column(ForeignKey("detections.id"), nullable=True)


class NotificationLog(Base):
    __tablename__ = "notification_logs"
    __table_args__ = (UniqueConstraint("event_id", "channel", "recipient_user_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("alert_events.id", ondelete="CASCADE"))
    channel: Mapped[str] = mapped_column(String(20), default="zalo")
    recipient_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_status_created", "status", "created_at"),
        CheckConstraint(
            "(job_id IS NOT NULL AND realtime_session_id IS NULL) OR "
            "(job_id IS NULL AND realtime_session_id IS NOT NULL)",
            name="ck_feedback_one_source",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("video_jobs.id", ondelete="CASCADE"), nullable=True)
    realtime_session_id: Mapped[str | None] = mapped_column(ForeignKey("realtime_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    realtime_event_id: Mapped[str | None] = mapped_column(ForeignKey("realtime_events.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    detection_id: Mapped[str | None] = mapped_column(ForeignKey("detections.id"), nullable=True)
    video_time_ms: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str] = mapped_column(String(500))
    type: Mapped[str] = mapped_column(String(30))
    expected_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    proposed_box: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    comment: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FeedbackReview(Base):
    __tablename__ = "feedback_reviews"

    feedback_id: Mapped[str] = mapped_column(ForeignKey("feedback.id", ondelete="CASCADE"), primary_key=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    corrected_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corrected_box: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    annotation_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    annotations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    decision: Mapped[str] = mapped_column(String(20))
    review_note: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(80))
    target_type: Mapped[str] = mapped_column(String(50))
    target_id: Mapped[str] = mapped_column(String(50))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
