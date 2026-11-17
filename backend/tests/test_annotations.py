import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.app import bootstrap
from backend.app.api import admin as admin_api
from backend.app.api.schemas import FeedbackReviewInput
from backend.app.db import Base
from backend.app.models import Detection, Feedback, FeedbackReview, ModelVersion, User, VideoJob


def test_model_boxes_are_suggestions_on_original_feedback_image(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'suggestions.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(admin_api, "runtime_file", lambda relative: tmp_path / relative)
    (tmp_path / "feedback").mkdir()
    try:
        with Session(engine, expire_on_commit=False) as db:
            model = ModelVersion(name="test", artifact_path="unused", sha256="x" * 64, class_map={"0": "khoi", "1": "lua"})
            owner = User(username="owner", password_hash="unused")
            db.add_all([model, owner])
            db.flush()
            job = VideoJob(user_id=owner.id, model_version_id=model.id, original_name="clip.mp4", source_path="uploads/clip.mp4", duration_ms=1000, source_fps=30, total_frames=30)
            db.add(job)
            db.flush()
            feedback = Feedback(job_id=job.id, user_id=owner.id, video_time_ms=433, image_path="feedback/frame.jpg", type="false_negative")
            db.add(feedback)
            db.add(Detection(job_id=job.id, frame_index=13, video_time_ms=433, class_id=1, confidence=0.8, x1=96, y1=72, x2=480, y2=360))
            db.add(Detection(job_id=job.id, frame_index=20, video_time_ms=667, class_id=0, confidence=0.8, x1=0, y1=0, x2=100, y2=100))
            db.commit()
            assert cv2.imwrite(str(tmp_path / "feedback" / "frame.jpg"), np.zeros((720, 960, 3), dtype=np.uint8))

            suggestions = admin_api.model_feedback_suggested_boxes(feedback.id, owner, db)
            assert suggestions == [{"class_id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5}]
    finally:
        engine.dispose()


def test_existing_database_gets_annotation_columns(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.sqlite').as_posix()}")
    try:
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE feedback_reviews (feedback_id VARCHAR(32) PRIMARY KEY, reviewer_id VARCHAR(32), "
                "corrected_class INTEGER, corrected_box JSON, decision VARCHAR(20), "
                "review_note VARCHAR(1000), reviewed_at DATETIME)"
            ))
        monkeypatch.setattr(bootstrap, "engine", engine)
        bootstrap.initialize_database()
        with engine.connect() as connection:
            columns = {row[1] for row in connection.execute(text("PRAGMA table_info(feedback_reviews)"))}
        assert {"annotation_status", "annotations"} <= columns
    finally:
        engine.dispose()


def test_export_contains_only_complete_verified_yolo_labels(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'labels.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(admin_api, "runtime_file", lambda relative: tmp_path / relative)
    (tmp_path / "exports").mkdir()
    (tmp_path / "feedback").mkdir()
    try:
        with Session(engine, expire_on_commit=False) as db:
            model = ModelVersion(name="test", artifact_path="unused", sha256="x" * 64, class_map={"0": "khoi", "1": "lua"}, status="active")
            actor = User(username="model-admin", password_hash="unused")
            owner = User(username="owner", password_hash="unused")
            db.add_all([model, actor, owner])
            db.flush()
            job = VideoJob(user_id=owner.id, model_version_id=model.id, original_name="clip.mp4", source_path="uploads/clip.mp4", duration_ms=1000, source_fps=10, total_frames=10, status="completed")
            db.add(job)
            db.flush()
            negative = Feedback(job_id=job.id, user_id=owner.id, video_time_ms=100, image_path="feedback/negative.jpg", type="false_positive")
            positive = Feedback(job_id=job.id, user_id=owner.id, video_time_ms=200, image_path="feedback/positive.jpg", type="false_negative")
            legacy = Feedback(job_id=job.id, user_id=owner.id, video_time_ms=300, image_path="feedback/legacy.jpg", type="false_positive")
            db.add_all([negative, positive, legacy])
            db.flush()
            for name in ("negative", "positive", "legacy"):
                (tmp_path / "feedback" / f"{name}.jpg").write_bytes(b"image")
            db.add(FeedbackReview(feedback_id=legacy.id, reviewer_id=actor.id, decision="accepted"))
            db.commit()

            with pytest.raises(HTTPException) as wrong_background:
                admin_api.review_feedback(negative.id, FeedbackReviewInput(
                    decision="accepted", annotation_status="background",
                    annotations=[{"class_id": 1, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}],
                ), actor, db)
            assert wrong_background.value.status_code == 400
            with pytest.raises(HTTPException) as no_box:
                admin_api.review_feedback(positive.id, FeedbackReviewInput(
                    decision="accepted", annotation_status="objects", annotations=[],
                ), actor, db)
            assert no_box.value.status_code == 400

            admin_api.review_feedback(negative.id, FeedbackReviewInput(
                decision="accepted", annotation_status="background", annotations=[],
            ), actor, db)
            admin_api.review_feedback(positive.id, FeedbackReviewInput(
                decision="accepted", annotation_status="objects", annotations=[
                    {"class_id": 1, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6},
                    {"class_id": 0, "x1": 0.5, "y1": 0.4, "x2": 0.9, "y2": 0.8},
                ],
            ), actor, db)
            response = admin_api.export_feedback(actor, db)
            with zipfile.ZipFile(response.path) as archive:
                names = set(archive.namelist())
                assert f"images/{negative.id}.jpg" in names
                assert f"images/{positive.id}.jpg" in names
                assert f"images/{legacy.id}.jpg" not in names
                assert archive.read(f"labels/{negative.id}.txt") == b""
                assert archive.read(f"labels/{positive.id}.txt").decode().splitlines() == [
                    "1 0.200000 0.400000 0.200000 0.400000",
                    "0 0.700000 0.600000 0.400000 0.400000",
                ]
                manifest = json.loads(archive.read("manifest.json"))
                assert len(manifest) == 2
                assert {entry["annotation_status"] for entry in manifest} == {"background", "objects"}
                assert json.loads(archive.read("classes.json")) == {"0": "khoi", "1": "lua"}
    finally:
        engine.dispose()
