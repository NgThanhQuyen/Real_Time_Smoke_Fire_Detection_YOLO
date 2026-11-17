from __future__ import annotations

import hashlib
import io
import json
import pickletools
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import RUNTIME, runtime_file
from ..db import get_db
from ..inference.geometry import output_size
from ..media import stream_video
from ..models import (
    AlertEvent,
    AuditLog,
    Detection,
    EventEvidence,
    Feedback,
    FeedbackReview,
    ModelVersion,
    NotificationLog,
    RealtimeEvent,
    RealtimeEvidence,
    RealtimeNotificationLog,
    RealtimeSession,
    User,
    UserProfile,
    VideoJob,
    new_id,
    utcnow,
)
from ..security import hash_password, require_role
from ..serializers import event_data, feedback_data, iso, job_data, model_data, user_data
from ..serializers import realtime_event_data, realtime_session_data
from .schemas import FeedbackReviewInput, ModelActivateInput, UserUpdateInput


ops = APIRouter(prefix="/api/admin", tags=["admin-ops"])
model_admin = APIRouter(prefix="/api/model-admin", tags=["admin-model"])


def audit(db: Session, actor_id: str, action: str, target_type: str, target_id: str, data: dict | None = None):
    db.add(
        AuditLog(
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata_json=data or {},
        )
    )


@ops.get("/events")
def list_events(
    limit: int = 50,
    offset: int = 0,
    date_from: date | None = None,
    date_to: date | None = None,
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    limit = min(max(limit, 1), 100)
    offset = max(offset, 0)
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ngày bắt đầu phải trước ngày kết thúc")
    query = (
        select(AlertEvent, VideoJob, User)
        .join(VideoJob, AlertEvent.job_id == VideoJob.id)
        .join(User, VideoJob.user_id == User.id)
        .order_by(AlertEvent.created_at.desc())
        .limit(100)
    )
    vietnam = timezone(timedelta(hours=7))
    if date_from:
        start = datetime.combine(date_from, time.min, tzinfo=vietnam).astimezone(timezone.utc)
        query = query.where(AlertEvent.created_at >= start)
    if date_to:
        end = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=vietnam).astimezone(timezone.utc)
        query = query.where(AlertEvent.created_at < end)
    rows = db.execute(query).all()
    items = []
    for event, job, user in rows:
        evidence = db.scalar(select(EventEvidence).where(EventEvidence.event_id == event.id).limit(1))
        notification = db.scalar(
            select(NotificationLog).where(
                NotificationLog.event_id == event.id,
                NotificationLog.channel == "zalo",
            ).limit(1)
        )
        items.append(
            {
                **event_data(event, evidence),
                "source_type": "video",
                "video_name": job.original_name,
                "video_status": job.status,
                "user_id": user.id,
                "username": user.username,
                "contact": event.contact_snapshot,
                "notification_status": notification.status if notification else "not_configured",
                "evidence_url": f"/api/admin/events/{event.id}/evidence" if evidence else None,
            }
        )
    realtime_query = (
        select(RealtimeEvent, RealtimeSession, User)
        .join(RealtimeSession, RealtimeEvent.session_id == RealtimeSession.id)
        .join(User, RealtimeSession.user_id == User.id)
        .order_by(RealtimeEvent.created_at.desc())
        .limit(100)
    )
    if date_from:
        realtime_query = realtime_query.where(RealtimeEvent.created_at >= start)
    if date_to:
        realtime_query = realtime_query.where(RealtimeEvent.created_at < end)
    for event, session, user in db.execute(realtime_query).all():
        evidence = db.scalar(select(RealtimeEvidence).where(RealtimeEvidence.event_id == event.id).limit(1))
        notification = db.scalar(
            select(RealtimeNotificationLog).where(
                RealtimeNotificationLog.event_id == event.id,
                RealtimeNotificationLog.channel == "zalo",
            ).order_by(RealtimeNotificationLog.created_at.desc()).limit(1)
        )
        profile = user.profile
        items.append({
            **realtime_event_data(event, evidence),
            "source_type": "webcam",
            "job_id": None,
            "session_id": session.id,
            "video_name": f"Webcam · phiên {session.id[:8]}",
            "video_status": session.status,
            "user_id": user.id,
            "username": user.username,
            "contact": {
                "full_name": profile.full_name if profile else None,
                "phone": profile.phone if profile else None,
                "address": profile.address if profile else None,
            },
            "notification_status": notification.status if notification else "not_configured",
            "evidence_url": f"/api/admin/realtime-events/{event.id}/evidence" if evidence else None,
        })
    items.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return items[offset:offset + limit]


@ops.get("/jobs")
def list_all_jobs(
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(VideoJob, User)
        .join(User, VideoJob.user_id == User.id)
        .order_by(VideoJob.created_at.desc())
        .limit(100)
    ).all()
    return [
        {
            **job_data(job, db.scalar(select(func.count()).where(AlertEvent.job_id == job.id)) or 0),
            "username": user.username,
            "contact": {
                "full_name": user.profile.full_name if user.profile else None,
                "phone": user.profile.phone if user.profile else None,
                "address": user.profile.address if user.profile else None,
            },
        }
        for job, user in rows
    ]


@ops.get("/jobs/{job_id}")
def admin_job_detail(
    job_id: str,
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    job = db.get(VideoJob, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy video")
    user = db.get(User, job.user_id)
    events = db.scalars(
        select(AlertEvent).where(AlertEvent.job_id == job.id).order_by(AlertEvent.start_ms)
    ).all()
    return {
        **job_data(job, len(events)),
        "username": user.username if user else "",
        "contact": {
            "full_name": user.profile.full_name if user and user.profile else None,
            "phone": user.profile.phone if user and user.profile else None,
            "address": user.profile.address if user and user.profile else None,
        },
        "events": [
            {
                **event_data(event),
                "evidence_url": f"/api/admin/events/{event.id}/evidence",
            }
            for event in events
        ],
        "video_url": f"/api/admin/jobs/{job.id}/media" if job.status == "completed" else None,
    }


@ops.get("/jobs/{job_id}/media")
def admin_job_media(
    job_id: str,
    request: Request,
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    job = db.get(VideoJob, job_id)
    if not job or job.status != "completed" or not job.output_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Video kết quả chưa sẵn sàng")
    return stream_video(runtime_file(job.output_path), request)


@ops.get("/events/{event_id}/evidence")
def admin_evidence(
    event_id: str,
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    item = db.scalar(select(EventEvidence).where(EventEvidence.event_id == event_id).limit(1))
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@ops.get("/realtime-events/{event_id}/evidence")
def admin_realtime_evidence(
    event_id: str,
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    item = db.scalar(select(RealtimeEvidence).where(RealtimeEvidence.event_id == event_id).limit(1))
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh minh chứng")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@ops.get("/users")
def list_users(
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    users = db.scalars(select(User).order_by(User.created_at.desc()).limit(200)).all()
    return [user_data(item) for item in users]


@ops.patch("/users/{user_id}")
def update_user(
    user_id: str,
    payload: UserUpdateInput,
    actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy tài khoản")
    if {item.role for item in target.roles} != {"user"}:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tài khoản quản trị chỉ được quản lý bằng lệnh cục bộ")
    data = payload.model_dump(exclude_unset=True)
    if "active" in data:
        target.active = bool(data["active"])
    if data.get("new_password"):
        target.password_hash = hash_password(data["new_password"])
    if any(key in data for key in ("full_name", "phone", "address")):
        profile = target.profile or UserProfile(user_id=target.id)
        if not target.profile:
            db.add(profile)
        for key in ("full_name", "phone", "address"):
            if key in data:
                setattr(profile, key, (data[key] or "").strip() or None)
    audit(db, actor.id, "user_updated", "user", target.id, {"fields": sorted(data)})
    db.commit()
    db.refresh(target)
    return user_data(target)


@ops.delete("/users/{user_id}")
def disable_user(
    user_id: str,
    actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    if user_id == actor.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Không thể tự khóa tài khoản")
    target = db.get(User, user_id)
    if not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy tài khoản")
    if {item.role for item in target.roles} != {"user"}:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Tài khoản quản trị chỉ được quản lý bằng lệnh cục bộ")
    target.active = False
    audit(db, actor.id, "user_disabled", "user", target.id)
    db.commit()
    return {"active": False}


@ops.get("/models")
def list_models(
    _actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    items = db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all()
    return [model_data(item) for item in items]


@ops.post("/models/{model_id}/activate")
def activate_model(
    model_id: str,
    payload: ModelActivateInput,
    actor: User = Depends(require_role("admin_ops")),
    db: Session = Depends(get_db),
):
    target = db.get(ModelVersion, model_id)
    if not target or target.status not in {"candidate", "retired"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Mô hình không ở trạng thái có thể kích hoạt")
    if not target.metrics:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Mô hình thiếu báo cáo đánh giá")
    running = db.scalar(
        select(func.count()).select_from(VideoJob).where(VideoJob.status.in_(["processing", "finalizing"]))
    )
    running_realtime = db.scalar(
        select(func.count()).select_from(RealtimeSession).where(RealtimeSession.status == "running")
    )
    if running or running_realtime:
        raise HTTPException(status.HTTP_409_CONFLICT, "Hãy chờ tác vụ đang xử lý hoàn tất")
    for item in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")).all():
        item.status = "retired"
    target.status = "active"
    target.activated_at = utcnow()
    audit(db, actor.id, "model_activated", "model_version", target.id, {"reason": payload.reason})
    db.commit()
    return model_data(target)


@model_admin.get("/feedback")
def list_feedback(
    state: str = "pending",
    _actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    query = select(Feedback).order_by(Feedback.created_at.desc()).limit(200)
    if state != "all":
        query = query.where(Feedback.status == state)
    items = db.scalars(query).all()
    result = []
    for item in items:
        job = db.get(VideoJob, item.job_id)
        realtime_session = db.get(RealtimeSession, item.realtime_session_id) if item.realtime_session_id else None
        user = db.get(User, item.user_id)
        review = db.get(FeedbackReview, item.id)
        result.append(
            {
                **feedback_data(item),
                "video_name": job.original_name if job else (
                    f"Webcam · phiên {realtime_session.id[:8]}" if realtime_session else "Nguồn không xác định"
                ),
                "source_type": "webcam" if realtime_session else "video",
                "realtime_session_id": item.realtime_session_id,
                "realtime_event_id": item.realtime_event_id,
                "username": user.username if user else None,
                "image_url": f"/api/model-admin/feedback/{item.id}/image",
                "annotation_status": review.annotation_status if review else None,
                "annotations": review.annotations if review and review.annotations else [],
                "review_note": review.review_note if review else None,
            }
        )
    return result


@model_admin.get("/feedback/{feedback_id}/image")
def model_feedback_image(
    feedback_id: str,
    _actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy phản hồi")
    return FileResponse(runtime_file(item.image_path), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@model_admin.get("/feedback/{feedback_id}/suggested-boxes")
def model_feedback_suggested_boxes(
    feedback_id: str,
    _actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy phản hồi")
    if item.realtime_session_id:
        if not item.realtime_event_id:
            return []
        event = db.get(RealtimeEvent, item.realtime_event_id)
        if not event:
            return []
        return [
            {
                "class_id": box["class_id"],
                "x1": box["xyxy"][0], "y1": box["xyxy"][1],
                "x2": box["xyxy"][2], "y2": box["xyxy"][3],
            }
            for box in event.detections
            if box.get("class_id") in (0, 1) and len(box.get("xyxy", [])) == 4
        ]
    job = db.get(VideoJob, item.job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy video phản hồi")
    frame_time = db.scalar(
        select(Detection.video_time_ms)
        .where(Detection.job_id == job.id)
        .order_by(func.abs(Detection.video_time_ms - item.video_time_ms))
        .limit(1)
    )
    tolerance_ms = max(80, round(1000 / max(job.source_fps, 1)))
    if frame_time is None or abs(frame_time - item.video_time_ms) > tolerance_ms:
        return []
    image = cv2.imread(str(runtime_file(item.image_path)))
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy ảnh phản hồi")
    height, width = image.shape[:2]
    render_width, render_height = output_size(width, height)
    detections = db.scalars(
        select(Detection).where(Detection.job_id == job.id, Detection.video_time_ms == frame_time)
    ).all()
    suggestions = []
    for detection in detections:
        x1 = max(0.0, min(1.0, detection.x1 / render_width))
        y1 = max(0.0, min(1.0, detection.y1 / render_height))
        x2 = max(0.0, min(1.0, detection.x2 / render_width))
        y2 = max(0.0, min(1.0, detection.y2 / render_height))
        if detection.class_id in (0, 1) and x2 > x1 and y2 > y1:
            suggestions.append({"class_id": detection.class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return suggestions


@model_admin.get("/feedback/{feedback_id}/video")
def model_feedback_video(
    feedback_id: str,
    request: Request,
    _actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    job = db.get(VideoJob, item.job_id) if item and item.job_id else None
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy video phản hồi")
    return stream_video(runtime_file(job.output_path or job.source_path), request)


@model_admin.patch("/feedback/{feedback_id}")
def review_feedback(
    feedback_id: str,
    payload: FeedbackReviewInput,
    actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    item = db.get(Feedback, feedback_id)
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Không tìm thấy phản hồi")
    if payload.decision == "accepted":
        if payload.annotation_status == "background" and payload.annotations:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ảnh âm tính không được có box")
        if payload.annotation_status == "objects" and not payload.annotations:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ảnh có đối tượng cần ít nhất một box")
        if payload.annotation_status is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần xác nhận đã kiểm tra toàn ảnh")
    review = db.get(FeedbackReview, item.id)
    if not review:
        review = FeedbackReview(feedback_id=item.id, reviewer_id=actor.id, decision=payload.decision)
        db.add(review)
    review.reviewer_id = actor.id
    review.corrected_class = None
    review.corrected_box = None
    review.annotation_status = payload.annotation_status if payload.decision == "accepted" else None
    review.annotations = [box.model_dump() for box in payload.annotations] if payload.decision == "accepted" else None
    review.decision = payload.decision
    review.review_note = payload.review_note
    review.reviewed_at = utcnow()
    item.status = payload.decision
    audit(db, actor.id, "feedback_reviewed", "feedback", item.id, {"decision": payload.decision})
    db.commit()
    return feedback_data(item)


@model_admin.post("/exports")
def export_feedback(
    actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(Feedback, FeedbackReview)
        .join(FeedbackReview, Feedback.id == FeedbackReview.feedback_id)
        .where(
            FeedbackReview.decision == "accepted",
            FeedbackReview.annotation_status.in_(["objects", "background"]),
        )
        .order_by(Feedback.created_at)
    ).all()
    if not rows:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Chưa có ảnh được gắn nhãn và xác nhận toàn ảnh")
    export_id = new_id()
    output = runtime_file(f"exports/{export_id}.zip")
    manifest = []
    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item, review in rows:
                image_name = f"images/{item.id}.jpg"
                label_name = f"labels/{item.id}.txt"
                if review.annotation_status == "objects" and not review.annotations:
                    raise ValueError("Mẫu có đối tượng nhưng chưa có box")
                if review.annotation_status == "background" and review.annotations:
                    raise ValueError("Mẫu âm tính chứa box")
                archive.write(runtime_file(item.image_path), image_name)
                lines = []
                for box in review.annotations or []:
                    width = box["x2"] - box["x1"]
                    height = box["y2"] - box["y1"]
                    cx = box["x1"] + width / 2
                    cy = box["y1"] + height / 2
                    lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
                archive.writestr(label_name, "\n".join(lines) + ("\n" if lines else ""))
                manifest.append(
                    {
                        "feedback_id": item.id,
                        "job_id": item.job_id,
                        "realtime_session_id": item.realtime_session_id,
                        "realtime_event_id": item.realtime_event_id,
                        "source_type": "webcam" if item.realtime_session_id else "video",
                        "video_time_ms": item.video_time_ms,
                        "type": item.type,
                        "reviewed_at": iso(review.reviewed_at),
                        "annotation_status": review.annotation_status,
                        "annotation_count": len(lines),
                        "image": image_name,
                        "label": label_name,
                    }
                )
            archive.writestr("classes.json", json.dumps({"0": "khoi", "1": "lua"}, ensure_ascii=False, indent=2))
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("README.txt", "Ảnh và nhãn YOLO đã được admin kiểm tra toàn khung. Ghép với dữ liệu gốc, chia train/val theo video nguồn và tạo data.yaml trước khi huấn luyện. File nhãn rỗng là ảnh âm tính đã xác nhận.\n")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    audit(db, actor.id, "feedback_exported", "export", export_id, {"count": len(manifest)})
    db.commit()
    return FileResponse(output, media_type="application/zip", filename=f"feedback-{export_id}.zip")


def check_checkpoint(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Checkpoint không có định dạng PyTorch hợp lệ")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Checkpoint bị hỏng")
        names = archive.namelist()
        pickles = [name for name in names if name.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Checkpoint thiếu metadata")
        content = archive.read(pickles[0])
        strings = {
            value
            for opcode, value, _ in pickletools.genops(content)
            if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"} and isinstance(value, str)
        }
        if not {"khoi", "lua"}.issubset(strings):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Checkpoint không khớp hai lớp khoi/lua")


@model_admin.post("/models", status_code=201)
async def submit_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    metrics_json: str = Form(...),
    actor: User = Depends(require_role("admin_model")),
    db: Session = Depends(get_db),
):
    name = name.strip()[:120]
    if not name or db.scalar(select(ModelVersion.id).where(ModelVersion.name == name)):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Tên phiên bản không hợp lệ hoặc đã tồn tại")
    try:
        metrics = json.loads(metrics_json)
    except json.JSONDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Báo cáo metrics phải là JSON hợp lệ")
    if not isinstance(metrics, dict) or not metrics:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cần báo cáo đánh giá mô hình")
    model_id = new_id()
    relative = f"models/{model_id}.pt"
    path = runtime_file(relative)
    count = 0
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                count += len(chunk)
                if count > 100 * 1024 * 1024:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Checkpoint vượt 100 MB")
                target.write(chunk)
        check_checkpoint(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        item = ModelVersion(
            id=model_id,
            name=name,
            artifact_path=f"runtime/{relative}",
            sha256=digest,
            class_map={"0": "khoi", "1": "lua"},
            metrics=metrics,
            config={},
            status="candidate",
            created_by=actor.id,
        )
        db.add(item)
        audit(db, actor.id, "model_submitted", "model_version", item.id)
        db.commit()
        return model_data(item)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
