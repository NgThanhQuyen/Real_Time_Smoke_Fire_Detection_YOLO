from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_torch_lib))
    except Exception:
        pass
try:
    import torch
except Exception:
    pass

import logging
import time
from contextlib import contextmanager

from sqlalchemy import select

from ..bootstrap import initialize_database
from ..config import RUNTIME
from ..db import SessionLocal
from ..inference.coordination import inference_slot
from ..inference.video import process_video
from ..models import ModelVersion, VideoJob, utcnow
from ..zalo import deliver_pending


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@contextmanager
def single_worker():
    path = RUNTIME / "worker.lock"
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Một worker khác đang chạy") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def process_next() -> bool:
    with SessionLocal() as db:
        job = db.scalar(
            select(VideoJob).where(VideoJob.status == "queued").order_by(VideoJob.created_at).limit(1)
        )
        if not job:
            return False
        job.status = "processing"
        job.started_at = utcnow()
        job.error_code = None
        db.commit()
        version = db.get(ModelVersion, job.model_version_id)
        if not version:
            job.status = "failed"
            job.error_code = "model_version_missing"
            db.commit()
            return True
        try:
            logger.info("Bắt đầu xử lý video %s", job.id)
            with inference_slot(blocking=True):
                process_video(db, job, version)
            logger.info("Hoàn tất video %s", job.id)
        except Exception as exc:
            logger.exception("Xử lý video thất bại: %s", job.id)
            db.rollback()
            job.status = "failed"
            job.error_code = type(exc).__name__
            db.commit()
        return True


def main() -> None:
    initialize_database()
    with single_worker():
        logger.info("Worker đã sẵn sàng")
        while True:
            handled = process_next()
            with SessionLocal() as db:
                deliver_pending(db)
            if not handled:
                time.sleep(1.5)


if __name__ == "__main__":
    main()
