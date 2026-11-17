from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from backend.app.config import INITIAL_MODEL_RELATIVE, RUNTIME, runtime_file
from backend.app.db import Base
from backend.app.inference.video import process_video
from backend.app.models import AlertEvent, ModelVersion, User, UserProfile, UserRole, VideoJob
from backend.app.security import hash_password


def main() -> None:
    source = runtime_file("uploads/smoke-check.mp4")
    output = runtime_file("rendered/smoke-check.mp4")
    db_path = runtime_file("tmp/smoke-check.sqlite3")
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 180))
    if not writer.isOpened():
        raise RuntimeError("Không tạo được video thử")
    for index in range(10):
        frame = np.full((180, 320, 3), 35 + index * 2, dtype=np.uint8)
        cv2.putText(frame, str(index), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (210, 210, 210), 2)
        writer.write(frame)
    writer.release()
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine, expire_on_commit=False) as db:
            user = User(username="smoke-check", password_hash=hash_password("test-password-123"))
            version = ModelVersion(
                name="best-check",
                artifact_path=INITIAL_MODEL_RELATIVE,
                sha256="test",
                class_map={"0": "khoi", "1": "lua"},
                status="active",
            )
            db.add_all([user, version])
            db.flush()
            db.add_all([UserProfile(user_id=user.id), UserRole(user_id=user.id, role="user")])
            job = VideoJob(
                id="smoke-check",
                user_id=user.id,
                model_version_id=version.id,
                original_name="smoke-check.mp4",
                source_path="uploads/smoke-check.mp4",
                duration_ms=2000,
                source_fps=5,
                total_frames=10,
                status="processing",
            )
            db.add(job)
            db.commit()
            process_video(db, job, version)
            event_count = db.scalar(select(func.count()).select_from(AlertEvent))
            print(
                {
                    "status": job.status,
                    "frames": job.processed_frames,
                    "inference_ms": job.inference_ms,
                    "processing_ms": job.processing_ms,
                    "events": event_count,
                    "output_bytes": output.stat().st_size,
                }
            )
            result = cv2.VideoCapture(str(output))
            print({"encoded_video_opens": result.isOpened(), "encoded_frames": result.get(cv2.CAP_PROP_FRAME_COUNT)})
            result.release()
    finally:
        engine.dispose()
        source.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        db_path.unlink(missing_ok=True)
        for evidence in (RUNTIME / "evidence").glob("smoke-check_*"):
            evidence.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
