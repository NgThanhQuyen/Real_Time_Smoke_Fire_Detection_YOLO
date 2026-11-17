from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import RUNTIME


class InferenceBusyError(RuntimeError):
    """Báo hiệu tiến trình suy luận khác đang giữ tài nguyên model."""


@contextmanager
def inference_slot(blocking: bool = False) -> Iterator[None]:
    """Dùng khóa liên tiến trình để tránh worker và webcam chạy YOLO cùng lúc."""
    lock_path: Path = RUNTIME / "inference.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()

        while True:
            lock_file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if not blocking:
                    raise InferenceBusyError from exc
                time.sleep(0.2)

        try:
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
