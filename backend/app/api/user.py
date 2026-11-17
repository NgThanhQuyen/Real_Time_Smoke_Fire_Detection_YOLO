from __future__ import annotations

from pathlib import Path
import logging
from datetime import timedelta, timezone
from io import BytesIO
import shutil

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..config import RUNTIME, settings, runtime_file
from ..db import get_db
from ..media import stream_video
from ..models import (
    AlertEvent,
    Detection,
    EventEvidence,
    Feedback,
    ModelVersion,
    RealtimeEvent,
    RealtimeEvidence,
    RealtimeSession,
    ZaloCredential,
    User,
    UserProfile,
    VideoJob,
    new_id,
    utcnow,
)
from ..public_media import resolve_public_media_token
from ..security import decrypt_secret, encrypt_secret, require_user
from ..serializers import detection_data, event_data, feedback_data, job_data, realtime_event_data, realtime_session_data
from ..zalo import ZaloApiError, discover_private_chat_id, get_webhook_url, send_message, zalo_api
from .schemas import FeedbackInput, ProfileInput, RealtimeEventFeedbackInput, RealtimeFeedbackInput, ZaloBotInput


router = APIRouter(prefix="/api", tags=["user"])
logger = logging.getLogger(__name__)


@router.get("/public/zalo-evidence/{token}", include_in_schema=False)
def public_zalo_evidence(token: str):
    try:
        image_path = resolve_public_media_token(token)
    except Exception:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh minh chứng")
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def remove_media(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Không xóa được media %s", path.name)


def owned_job(job_id: str, user: User, db: Session) -> VideoJob:
    job = db.get(VideoJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy video")
    return job


def owned_realtime_session(session_id: str, user: User, db: Session) -> RealtimeSession:
    item = db.get(RealtimeSession, session_id)
    if not item or item.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy phiên webcam")
    return item


def enforce_feedback_rate_limit(user: User, db: Session) -> None:
    recent = db.scalar(
        select(func.count(Feedback.id)).where(
            Feedback.user_id == user.id,
            Feedback.created_at >= utcnow() - timedelta(minutes=5),
        )
    ) or 0
    if recent >= 10:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Bạn đã gửi nhiều phản hồi trong thời gian ngắn")


@router.get("/realtime-sessions")
def list_realtime_sessions(user: User = Depends(require_user), db: Session = Depends(get_db)):
    sessions = db.scalars(
        select(RealtimeSession).where(RealtimeSession.user_id == user.id)
        .order_by(RealtimeSession.started_at.desc()).limit(50)
    ).all()
    result = []
    for item in sessions:
        events = db.scalars(
            select(RealtimeEvent).where(RealtimeEvent.session_id == item.id)
            .order_by(RealtimeEvent.start_ms)
        ).all()
        event_rows = []
        for event in events:
            evidence = db.scalar(select(RealtimeEvidence).where(RealtimeEvidence.event_id == event.id).limit(1))
            row = realtime_event_data(event, evidence)
            row["evidence_url"] = (
                f"/api/realtime-sessions/{item.id}/evidence/{evidence.id}" if evidence else None
            )
            event_rows.append(row)
        result.append({**realtime_session_data(item, len(events)), "events": event_rows})
    return result


@router.delete("/realtime-sessions/{session_id}")
def delete_realtime_session(
    session_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    session = owned_realtime_session(session_id, user, db)
    enforce_feedback_rate_limit(user, db)
    evidence_paths = db.scalars(
        select(RealtimeEvidence.image_path)
        .join(RealtimeEvent, RealtimeEvidence.event_id == RealtimeEvent.id)
        .where(RealtimeEvent.session_id == session.id)
    ).all()
    feedback_paths = db.scalars(
        select(Feedback.image_path).where(Feedback.realtime_session_id == session.id)
    ).all()
    if session.status == "running":
        raise HTTPException(status.HTTP_409_CONFLICT, "Dừng webcam trước khi xóa phiên")
    db.delete(session)
    db.commit()
    remove_media([runtime_file(relative) for relative in [*evidence_paths, *feedback_paths]])
    return {"deleted": True}


@router.get("/realtime-sessions/{session_id}/evidence/{evidence_id}")
def realtime_evidence_image(
    session_id: str,
    evidence_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    session = owned_realtime_session(session_id, user, db)
    item = db.get(RealtimeEvidence, evidence_id)
    if not item or not db.scalar(
        select(RealtimeEvent.id).where(RealtimeEvent.id == item.event_id, RealtimeEvent.session_id == session.id)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh minh chứng")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/realtime-sessions/{session_id}/feedback", status_code=201)
async def create_realtime_feedback(
    session_id: str,
    elapsed_ms: int = Form(ge=0),
    type: str = Form(),
    expected_class: int | None = Form(default=None, ge=0, le=1),
    comment: str | None = Form(default=None, max_length=1000),
    frame: UploadFile = File(),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    session = owned_realtime_session(session_id, user, db)
    payload = RealtimeFeedbackInput(
        elapsed_ms=elapsed_ms, type=type, expected_class=expected_class, comment=comment
    )
    if payload.type in {"false_negative", "wrong_class"} and payload.expected_class is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần chọn lớp đúng cho phản hồi này")
    allowed_duration = session.duration_ms
    if session.status == "running":
        started_at = session.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        allowed_duration = max(allowed_duration, round((utcnow() - started_at).total_seconds() * 1000))
    if payload.elapsed_ms > allowed_duration + 5000:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Thời điểm khung hình không thuộc phiên webcam")
    raw = await frame.read(5_000_001)
    if len(raw) > 5_000_000:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Ảnh phản hồi vượt quá 5 MB")
    try:
        with Image.open(BytesIO(raw)) as source_image:
            width, height = source_image.size
            if width <= 0 or height <= 0 or width * height > 2_000_000:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ảnh phản hồi có kích thước quá lớn")
            source_image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ảnh phản hồi không hợp lệ")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ảnh phản hồi không hợp lệ hoặc quá lớn")
    feedback_id = new_id()
    relative = f"feedback/{feedback_id}.jpg"
    target = runtime_file(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Không lưu được ảnh phản hồi")
    event_id = db.scalar(
        select(RealtimeEvent.id)
        .where(
            RealtimeEvent.session_id == session.id,
            RealtimeEvent.start_ms <= payload.elapsed_ms,
            RealtimeEvent.end_ms >= payload.elapsed_ms,
        )
        .order_by(RealtimeEvent.created_at.desc()).limit(1)
    )
    item = Feedback(
        id=feedback_id,
        job_id=None,
        realtime_session_id=session.id,
        realtime_event_id=event_id,
        user_id=user.id,
        detection_id=None,
        video_time_ms=payload.elapsed_ms,
        image_path=relative,
        type=payload.type,
        expected_class=payload.expected_class,
        proposed_box=None,
        comment=payload.comment,
    )
    db.add(item)
    db.commit()
    return feedback_data(item)


@router.post("/realtime-events/{event_id}/feedback", status_code=201)
def report_realtime_event(
    event_id: str,
    payload: RealtimeEventFeedbackInput,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    event = db.get(RealtimeEvent, event_id)
    session = db.get(RealtimeSession, event.session_id) if event else None
    if not event or not session or session.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy cảnh báo webcam")
    enforce_feedback_rate_limit(user, db)
    duplicate = db.scalar(
        select(Feedback.id).where(
            Feedback.user_id == user.id,
            Feedback.realtime_event_id == event.id,
            Feedback.type == payload.type,
            Feedback.status == "pending",
        ).limit(1)
    )
    if duplicate:
        raise HTTPException(status.HTTP_409_CONFLICT, "Bạn đã gửi loại phản hồi này cho cảnh báo")
    if payload.type in {"false_negative", "wrong_class"} and payload.expected_class is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần chọn lớp đúng cho phản hồi này")
    evidence = db.scalar(select(RealtimeEvidence).where(RealtimeEvidence.event_id == event.id).limit(1))
    if not evidence:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cảnh báo chưa có ảnh minh chứng để phản hồi")
    feedback_id = new_id()
    relative = f"feedback/{feedback_id}.jpg"
    target = runtime_file(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runtime_file(evidence.image_path), target)
    item = Feedback(
        id=feedback_id,
        job_id=None,
        realtime_session_id=session.id,
        realtime_event_id=event.id,
        user_id=user.id,
        detection_id=None,
        video_time_ms=event.start_ms,
        image_path=relative,
        type=payload.type,
        expected_class=payload.expected_class,
        proposed_box=None,
        comment=payload.comment,
    )
    db.add(item)
    db.commit()
    return feedback_data(item)


@router.get("/me/profile")
def get_profile(user: User = Depends(require_user), db: Session = Depends(get_db)):
    profile = db.get(UserProfile, user.id)
    credential = db.get(ZaloCredential, user.id)
    return {
        "full_name": profile.full_name if profile else None,
        "phone": profile.phone if profile else None,
        "address": profile.address if profile else None,
        "zalo_status": (
            "verified" if credential and credential.verified_at and credential.enabled
            else "saved" if credential else "unconfigured"
        ),
    }


@router.post("/me/zalo/connect")
def connect_zalo(
    payload: ZaloBotInput,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    token = payload.bot_token.strip()
    try:
        zalo_api(token, "getMe", {})
        webhook_url = get_webhook_url(token)
        if webhook_url:
            raise ZaloApiError(
                "webhook_conflict",
                "Bot đang bật webhook nên hệ thống không thể lấy tin nhắn bằng getUpdates. Hãy tắt webhook trong cấu hình bot rồi kết nối lại.",
            )
        chat_id = discover_private_chat_id(token)
    except ZaloApiError as exc:
        logger.warning("Liên kết bot Zalo thất bại: loại=%s", exc.category)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            exc.public_message,
        )
    except Exception as exc:
        logger.warning("Liên kết bot Zalo gặp lỗi nội bộ: %s", type(exc).__name__)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không thể đọc tin nhắn từ bot. Thử kết nối lại.")
    if not chat_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Zalo chưa trả về tin nhắn riêng để xác định cuộc trò chuyện. Kiểm tra đã nhắn đúng bot từ tài khoản sẽ nhận cảnh báo chưa; bot chỉ nhận event qua getUpdates khi không dùng webhook. Nếu có chương trình khác đang polling cùng bot, chương trình đó có thể đã lấy update. Hãy dừng các poller khác rồi thử kết nối lại.",
        )
    try:
        send_message(token, chat_id, "Đã kết nối bot Zalo với hệ thống phát hiện lửa và khói.")
    except Exception:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Đã tìm thấy cuộc trò chuyện nhưng không gửi được tin thử. Kiểm tra quyền gửi tin của bot.",
        )
    credential = db.get(ZaloCredential, user.id)
    if not credential:
        credential = ZaloCredential(user_id=user.id, token_ciphertext="", chat_id_ciphertext="")
        db.add(credential)
    credential.token_ciphertext = encrypt_secret(token)
    credential.chat_id_ciphertext = encrypt_secret(chat_id)
    credential.verified_at = utcnow()
    credential.enabled = True
    credential.updated_at = utcnow()
    db.commit()
    return {"status": "verified"}


@router.post("/me/zalo/test")
def test_zalo(user: User = Depends(require_user), db: Session = Depends(get_db)):
    credential = db.get(ZaloCredential, user.id)
    if not credential:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Bạn chưa kết nối bot Zalo")
    try:
        send_message(
            decrypt_secret(credential.token_ciphertext),
            decrypt_secret(credential.chat_id_ciphertext),
            "Tin nhắn kiểm tra từ hệ thống phát hiện lửa và khói.",
        )
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Gửi thử thất bại. Hãy kết nối lại bot Zalo.")
    credential.verified_at = utcnow()
    credential.enabled = True
    db.commit()
    return {"status": "verified"}


@router.delete("/me/zalo")
def delete_zalo(user: User = Depends(require_user), db: Session = Depends(get_db)):
    credential = db.get(ZaloCredential, user.id)
    if credential:
        db.delete(credential)
        db.commit()
    return {"status": "unconfigured"}


@router.patch("/me/profile")
def update_profile(
    payload: ProfileInput,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    profile = db.get(UserProfile, user.id)
    if not profile:
        profile = UserProfile(user_id=user.id)
        db.add(profile)
    profile.full_name = payload.full_name
    profile.phone = payload.phone
    profile.address = payload.address
    profile.updated_at = utcnow()
    db.commit()
    return get_profile(user, db)


def inspect_video(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không đọc được video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or fps > 60 or frames <= 0 or width <= 0 or height <= 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Metadata video không hợp lệ")
        if max(width, height) > 3840 or min(width, height) > 2160:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Video vượt giới hạn đầu vào 4K")
        duration = frames / fps
        if duration > settings.max_video_seconds:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video vượt giới hạn thời lượng")
        return duration, fps, frames
    finally:
        cap.release()


@router.post("/videos", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    name = Path(file.filename or "video.mp4").name[:255]
    if Path(name).suffix.lower() != ".mp4":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Chỉ nhận video MP4")
    model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
    if not model:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Chưa có mô hình đang hoạt động")
    job_id = new_id()
    relative = f"uploads/{job_id}.mp4"
    path = runtime_file(relative)
    count = 0
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                count += len(chunk)
                if count > settings.max_video_bytes:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Video vượt giới hạn dung lượng")
                target.write(chunk)
        with path.open("rb") as uploaded:
            header = uploaded.read(16)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Tệp không phải MP4 hợp lệ")
        duration, fps, frames = inspect_video(path)
        job = VideoJob(
            id=job_id,
            user_id=user.id,
            model_version_id=model.id,
            original_name=name,
            source_path=relative,
            duration_ms=round(duration * 1000),
            source_fps=fps,
            total_frames=frames,
            status="queued",
        )
        db.add(job)
        db.commit()
        return job_data(job)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/videos")
def list_videos(
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    jobs = db.scalars(
        select(VideoJob).where(VideoJob.user_id == user.id).order_by(VideoJob.created_at.desc()).limit(100)
    ).all()
    return [
        job_data(job, db.scalar(select(func.count()).where(AlertEvent.job_id == job.id)) or 0)
        for job in jobs
    ]


@router.delete("/videos/{job_id}")
def delete_video(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = owned_job(job_id, user, db)
    if job.status in {"processing", "finalizing", "queued"}:
        raise HTTPException(status.HTTP_409_CONFLICT, "Chỉ xóa video sau khi tác vụ kết thúc")
    image_paths = db.scalars(
        select(EventEvidence.image_path)
        .join(AlertEvent, EventEvidence.event_id == AlertEvent.id)
        .where(AlertEvent.job_id == job.id)
    ).all()
    feedback_paths = db.scalars(select(Feedback.image_path).where(Feedback.job_id == job.id)).all()
    paths = [runtime_file(job.source_path)]
    if job.output_path:
        paths.append(runtime_file(job.output_path))
    paths.extend(runtime_file(relative) for relative in [*image_paths, *feedback_paths])
    paths.extend((RUNTIME / "thumbnails").glob(f"{job.id}_*.jpg"))
    db.delete(job)
    db.commit()
    remove_media(paths)
    return {"deleted": True}


@router.get("/videos/{job_id}")
def get_video(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = owned_job(job_id, user, db)
    events = db.scalars(
        select(AlertEvent).where(AlertEvent.job_id == job.id).order_by(AlertEvent.start_ms)
    ).all()
    evidence = {
        item.event_id: item
        for item in db.scalars(
            select(EventEvidence).where(EventEvidence.event_id.in_([event.id for event in events]))
        ).all()
    } if events else {}
    return {
        **job_data(job, len(events)),
        "events": [event_data(event, evidence.get(event.id)) for event in events],
        "video_url": f"/api/videos/{job.id}/media" if job.status == "completed" else None,
    }


@router.post("/videos/{job_id}/retry")
def retry_video(job_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = owned_job(job_id, user, db)
    if job.status not in {"failed", "interrupted"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Chỉ có thể chạy lại tác vụ thất bại hoặc gián đoạn")
    db.execute(delete(Detection).where(Detection.job_id == job.id))
    for evidence in (RUNTIME / "evidence").glob(f"{job.id}_*.jpg"):
        evidence.unlink(missing_ok=True)
    job.status = "queued"
    job.error_code = None
    job.processed_frames = 0
    job.detected_frames = 0
    job.started_at = None
    job.completed_at = None
    db.commit()
    return job_data(job)


@router.get("/videos/{job_id}/detections")
def list_detections(
    job_id: str,
    start_ms: int = 0,
    end_ms: int = 10000,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = owned_job(job_id, user, db)
    if end_ms < start_ms or end_ms - start_ms > 15000:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Khoảng thời gian không hợp lệ")
    items = db.scalars(
        select(Detection)
        .where(
            Detection.job_id == job.id,
            Detection.video_time_ms >= start_ms,
            Detection.video_time_ms <= end_ms,
        )
        .order_by(Detection.video_time_ms)
        .limit(3000)
    ).all()
    return [detection_data(item) for item in items]


@router.get("/videos/{job_id}/detected-frames")
def detected_frames(
    job_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = owned_job(job_id, user, db)
    if job.status != "completed":
        return []
    rows = db.execute(
        select(Detection.frame_index, Detection.video_time_ms, func.max(Detection.confidence))
        .where(Detection.job_id == job.id)
        .group_by(Detection.frame_index, Detection.video_time_ms)
        .order_by(Detection.frame_index)
    ).all()
    slots: dict[int, tuple[int, int, float]] = {}
    for frame_index, video_time_ms, confidence in rows:
        slot = int(video_time_ms // 500)
        if slot not in slots or confidence > slots[slot][2]:
            slots[slot] = (frame_index, video_time_ms, confidence)
    selected = sorted(slots.values(), key=lambda item: item[1])
    if len(selected) > 48:
        selected = [selected[round(index * (len(selected) - 1) / 47)] for index in range(48)]
    return [
        {
            "frame_index": frame_index,
            "video_time_ms": video_time_ms,
            "confidence": confidence,
            "image_url": f"/api/videos/{job.id}/frames/{frame_index}",
        }
        for frame_index, video_time_ms, confidence in selected
    ]


@router.get("/videos/{job_id}/frames/{frame_index}")
def detected_frame_image(
    job_id: str,
    frame_index: int,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = owned_job(job_id, user, db)
    if job.status != "completed" or not job.output_path or frame_index < 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy khung hình")
    has_detection = db.scalar(
        select(Detection.id)
        .where(Detection.job_id == job.id, Detection.frame_index == frame_index)
        .limit(1)
    )
    if not has_detection:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Khung hình không có dự đoán")
    image_path = runtime_file(f"thumbnails/{job.id}_{frame_index}.jpg")
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(runtime_file(job.output_path)))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or not cv2.imwrite(str(image_path), frame):
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Không tạo được ảnh khung hình")
        finally:
            cap.release()
    return FileResponse(image_path, media_type="image/jpeg")


@router.get("/videos/{job_id}/media")
def video_media(job_id: str, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    job = owned_job(job_id, user, db)
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video kết quả chưa sẵn sàng")
    return stream_video(runtime_file(job.output_path), request)


@router.get("/videos/{job_id}/evidence/{evidence_id}")
def evidence_media(
    job_id: str,
    evidence_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    owned_job(job_id, user, db)
    item = db.get(EventEvidence, evidence_id)
    event = db.get(AlertEvent, item.event_id) if item else None
    if not event or event.job_id != job_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def extract_frame(job: VideoJob, video_time_ms: int, target: Path) -> None:
    if not job.output_path:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Video kết quả chưa sẵn sàng")
    cap = cv2.VideoCapture(str(runtime_file(job.source_path)))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, video_time_ms)
        ok, frame = cap.read()
        if not ok and job.total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(job.total_frames - 1, 0))
            ok, frame = cap.read()
        if not ok or not cv2.imwrite(str(target), frame):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không lấy được khung hình tại thời điểm này")
    finally:
        cap.release()


@router.post("/videos/{job_id}/feedback", status_code=201)
def create_feedback(
    job_id: str,
    payload: FeedbackInput,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    job = owned_job(job_id, user, db)
    if job.status != "completed" or payload.video_time_ms > job.duration_ms:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Thời điểm phản hồi không hợp lệ")
    if payload.type == "false_negative" and payload.expected_class is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần chọn lớp bị bỏ sót")
    detection = db.get(Detection, payload.detection_id) if payload.detection_id else None
    if payload.detection_id and (not detection or detection.job_id != job.id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Box được chọn không thuộc video")
    if payload.type != "false_negative" and not detection:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần chọn box dự đoán để báo sai")
    if detection and abs(detection.video_time_ms - payload.video_time_ms) > 500:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Box không khớp thời điểm video")
    feedback_id = new_id()
    relative = f"feedback/{feedback_id}.jpg"
    target = runtime_file(relative)
    extract_frame(job, payload.video_time_ms, target)
    item = Feedback(
        id=feedback_id,
        job_id=job.id,
        user_id=user.id,
        detection_id=payload.detection_id,
        video_time_ms=payload.video_time_ms,
        image_path=relative,
        type=payload.type,
        expected_class=payload.expected_class,
        proposed_box=payload.proposed_box,
        comment=payload.comment,
    )
    db.add(item)
    db.commit()
    return feedback_data(item)


@router.get("/me/feedback")
def my_feedback(user: User = Depends(require_user), db: Session = Depends(get_db)):
    items = db.scalars(
        select(Feedback).where(Feedback.user_id == user.id).order_by(Feedback.created_at.desc()).limit(100)
    ).all()
    return [feedback_data(item) for item in items]


@router.delete("/feedback/{feedback_id}")
def delete_feedback(
    feedback_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    if not item or item.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy phản hồi")
    image = runtime_file(item.image_path)
    db.delete(item)
    db.commit()
    remove_media([image])
    return {"deleted": True}


@router.get("/feedback/{feedback_id}/image")
def feedback_image(
    feedback_id: str,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    if not item or item.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh phản hồi")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
