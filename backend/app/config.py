from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Đảm bảo PyTorch DLL và OpenMP được khởi tạo đúng trên Windows trước khi bất kỳ module nào nạp OpenCV (cv2)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_torch_lib))
    except Exception:
        pass
try:
    import torch  # nạp sớm c10.dll tránh WinError 1114
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
RUNTIME = ROOT / "runtime"
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME / "ultralytics"))
(RUNTIME / "ultralytics").mkdir(parents=True, exist_ok=True)
DATABASE_PATH = RUNTIME / "db" / "app.sqlite3"
INITIAL_MODEL_RELATIVE = "models/final_sfd_yolo11n/best.pt"
INITIAL_MODEL_DIR = ROOT / "models" / "final_sfd_yolo11n"


@dataclass(frozen=True)
class Settings:
    app_origin: str = os.getenv("APP_ORIGIN", "http://localhost:5173")
    api_origin: str = os.getenv("API_ORIGIN", "http://localhost:8000")
    public_api_origin: str = os.getenv("PUBLIC_API_ORIGIN", "").rstrip("/")
    max_video_seconds: float = float(os.getenv("MAX_VIDEO_SECONDS", "30"))
    max_video_bytes: int = int(os.getenv("MAX_VIDEO_BYTES", str(50 * 1024 * 1024)))
    model_image_size: int = int(os.getenv("MODEL_IMAGE_SIZE", "832"))
    detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    detection_iou: float = float(os.getenv("DETECTION_IOU", "0.6"))
    event_window_seconds: float = float(os.getenv("EVENT_WINDOW_SECONDS", "1.5"))
    event_trigger_ratio: float = float(os.getenv("EVENT_TRIGGER_RATIO", "0.6"))
    event_release_ratio: float = float(os.getenv("EVENT_RELEASE_RATIO", "0.25"))
    event_min_seconds: float = float(os.getenv("EVENT_MIN_SECONDS", "0.7"))
    event_merge_gap_seconds: float = float(os.getenv("EVENT_MERGE_GAP_SECONDS", "1.0"))
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "")
    app_encryption_key: str = os.getenv("APP_ENCRYPTION_KEY", "")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_tracking_username: str = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    mlflow_tracking_password: str = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


settings = Settings()


def prepare_runtime() -> None:
    for relative in (
        "db",
        "uploads",
        "rendered",
        "evidence",
        "feedback",
        "exports",
        "models",
        "ultralytics",
        "tmp",
    ):
        (RUNTIME / relative).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME / "ultralytics"))


def runtime_file(relative: str) -> Path:
    candidate = (RUNTIME / relative).resolve()
    if not candidate.is_relative_to(RUNTIME.resolve()):
        raise ValueError("Đường dẫn media không hợp lệ")
    return candidate
