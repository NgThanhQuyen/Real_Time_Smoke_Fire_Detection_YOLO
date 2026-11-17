from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import ROOT, settings, runtime_file
from ..models import (
    AlertEvent,
    Detection,
    EventEvidence,
    ModelVersion,
    NotificationLog,
    ZaloCredential,
    User,
    UserProfile,
    VideoJob,
    new_id,
    utcnow,
)
from .geometry import output_size
from .smoother import AlertSmoother


logger = logging.getLogger(__name__)
COLORS = {0: (183, 115, 48), 1: (63, 82, 231)}
LABELS = {0: "KHOI", 1: "LUA"}


def model_file(version: ModelVersion) -> Path:
    candidate = (ROOT / version.artifact_path).resolve()
    if not candidate.is_relative_to(ROOT.resolve()) or not candidate.is_file():
        raise FileNotFoundError("Không tìm thấy checkpoint của phiên bản mô hình")
    return candidate


def ffmpeg_binary() -> str:
    return settings.ffmpeg_path or imageio_ffmpeg.get_ffmpeg_exe()


def start_encoder(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    args = [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def draw_boxes(frame, detections: list[Detection]) -> None:
    for item in detections:
        color = COLORS.get(item.class_id, (255, 255, 255))
        x1, y1, x2, y2 = (int(item.x1), int(item.y1), int(item.x2), int(item.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{LABELS.get(item.class_id, 'DOI TUONG')} {item.confidence:.2f}"
        size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        top = max(0, y1 - size[1] - baseline - 8)
        cv2.rectangle(frame, (x1, top), (x1 + size[0] + 12, top + size[1] + baseline + 8), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 6, top + size[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_video(db: Session, job: VideoJob, version: ModelVersion) -> None:
    from ultralytics import YOLO

    source = runtime_file(job.source_path)
    relative_output = f"rendered/{job.id}.mp4"
    output = runtime_file(relative_output)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Không thể mở video nguồn")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = job.source_fps
    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise ValueError("Metadata video nguồn không hợp lệ")
    model = YOLO(str(model_file(version)))
    names = {int(key): value for key, value in model.names.items()}
    if names != {0: "khoi", 1: "lua"}:
        cap.release()
        raise ValueError("Checkpoint không khớp lớp khoi/lua")
    render_width, render_height = output_size(width, height)
    encoder = start_encoder(output, render_width, render_height, fps)
    smoother = AlertSmoother(
        window_ms=round(settings.event_window_seconds * 1000),
        trigger_ratio=settings.event_trigger_ratio,
        release_ratio=settings.event_release_ratio,
        min_event_ms=round(settings.event_min_seconds * 1000),
        merge_gap_ms=round(settings.event_merge_gap_seconds * 1000),
    )
    started = time.perf_counter()
    inference_seconds = 0.0
    frame_index = 0
    detected_frames = 0
    all_evidence_paths: set[str] = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != render_width or frame.shape[0] != render_height:
                frame = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            video_time_ms = round(frame_index / fps * 1000)
            frame_started = time.perf_counter()
            result = model.predict(
                frame,
                imgsz=settings.model_image_size,
                conf=settings.detection_confidence,
                iou=settings.detection_iou,
                device="cpu",
                verbose=False,
            )[0]
            inference_seconds += time.perf_counter() - frame_started
            rows: list[Detection] = []
            class_peaks: dict[int, tuple[float, str | None]] = {}
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id not in (0, 1):
                    continue
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                item = Detection(
                    id=new_id(),
                    job_id=job.id,
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    class_id=class_id,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                db.add(item)
                rows.append(item)
                if confidence > class_peaks.get(class_id, (0.0, None))[0]:
                    class_peaks[class_id] = (confidence, item.id)
            if rows:
                detected_frames += 1
            peaks = smoother.update(video_time_ms, frame_index, class_peaks)
            draw_boxes(frame, rows)
            for event in peaks:
                relative_image = event.image_path or f"evidence/{job.id}_{event.id}.jpg"
                image = runtime_file(relative_image)
                if not cv2.imwrite(str(image), frame):
                    raise OSError("Không ghi được ảnh bằng chứng")
                event.image_path = relative_image
                all_evidence_paths.add(relative_image)
            if encoder.stdin is None:
                raise OSError("Không mở được luồng mã hóa video")
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
            if frame_index % 10 == 0:
                job.processed_frames = frame_index
                job.detected_frames = detected_frames
                db.commit()
        if frame_index == 0:
            raise ValueError("Video không có khung hình đọc được")
        job.status = "finalizing"
        job.processed_frames = frame_index
        job.detected_frames = detected_frames
        db.commit()
        if encoder.stdin:
            encoder.stdin.close()
        error = encoder.stderr.read() if encoder.stderr else b""
        code = encoder.wait(timeout=120)
        if code != 0 or not output.is_file() or output.stat().st_size == 0:
            raise OSError(f"FFmpeg mã hóa thất bại: {error.decode(errors='replace')[-300:]}")
        events, discarded = smoother.close()
        for relative in discarded:
            runtime_file(relative).unlink(missing_ok=True)
            all_evidence_paths.discard(relative)
        user = db.get(User, job.user_id)
        profile = db.get(UserProfile, job.user_id)
        contact = {
            "username": user.username if user else None,
            "full_name": profile.full_name if profile else None,
            "phone": profile.phone if profile else None,
            "address": profile.address if profile else None,
        }
        zalo = db.get(ZaloCredential, job.user_id)
        for item in events:
            event = AlertEvent(
                id=item.id,
                job_id=job.id,
                class_id=item.class_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                peak_confidence=item.peak_confidence,
                contact_snapshot=contact,
            )
            db.add(event)
            if item.image_path:
                db.add(
                    EventEvidence(
                        event_id=event.id,
                        frame_index=item.peak_frame_index,
                        video_time_ms=round(item.peak_frame_index / fps * 1000),
                        image_path=item.image_path,
                        detection_id=item.detection_id,
                    )
                )
            if zalo and zalo.enabled and zalo.verified_at:
                db.add(
                    NotificationLog(
                        event_id=event.id,
                        recipient_user_id=job.user_id,
                        channel="zalo",
                        status="pending",
                    )
                )
        job.output_path = relative_output
        job.inference_ms = round(inference_seconds * 1000, 1)
        job.processing_ms = round((time.perf_counter() - started) * 1000, 1)
        job.completed_at = utcnow()
        job.status = "completed"
        db.commit()
    except Exception:
        db.rollback()
        if encoder.poll() is None:
            encoder.kill()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        encoder.wait()
        output.unlink(missing_ok=True)
        for relative in all_evidence_paths:
            runtime_file(relative).unlink(missing_ok=True)
        db.execute(delete(Detection).where(Detection.job_id == job.id))
        db.commit()
        raise
    finally:
        cap.release()
        if encoder.stderr:
            encoder.stderr.close()
