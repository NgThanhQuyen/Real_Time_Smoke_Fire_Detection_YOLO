from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import threading
import time
from datetime import timezone

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
from sqlalchemy import select

from ..config import runtime_file, settings
from ..db import SessionLocal
from ..inference.coordination import InferenceBusyError, inference_slot
from ..models import (
    AuthSession, ModelVersion, RealtimeEvidence, RealtimeEvent, RealtimeNotificationLog,
    RealtimeSession, ZaloCredential, User, new_id, utcnow,
)
from ..public_media import create_public_media_url
from ..security import SESSION_COOKIE, decrypt_secret
from ..zalo import realtime_notification_text, send_alert
from ..inference.realtime_events import RealtimeEventTracker
from ..inference.video import model_file


router = APIRouter()
logger = logging.getLogger(__name__)
_model_lock = threading.Lock()
_cached_model_path: str | None = None
_cached_model = None
MAX_FRAME_BYTES = 1_000_000
MAX_FRAME_PIXELS = 2_000_000
REALTIME_IMAGE_SIZE = 640
BACKGROUND_ZALO_TASKS: set[asyncio.Task] = set()
ACTIVE_REALTIME_USERS: set[str] = set()
_active_sessions_lock = threading.Lock()


def authenticated_user(
    websocket: WebSocket,
) -> tuple[User | None, ModelVersion | None, tuple[str, str] | None, str]:
    token = websocket.cookies.get(SESSION_COOKIE)
    if not token:
        return None, None, None, "unconfigured"
    digest = hashlib.sha256(token.encode()).hexdigest()
    with SessionLocal() as db:
        session = db.get(AuthSession, digest)
        if not session:
            return None, None, None, "unconfigured"
        expires = session.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= utcnow():
            return None, None, None, "unconfigured"
        user = db.get(User, session.user_id)
        if not user or not user.active:
            return None, None, None, "unconfigured"
        roles = {role.role for role in user.roles}
        if "user" not in roles or roles.intersection({"admin_ops", "admin_model"}):
            return None, None, None, "unconfigured"
        model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
        if not model:
            return user, None, None, "unconfigured"
        db.expunge(model)
        credential = db.get(ZaloCredential, user.id)
        if not credential:
            return user, model, None, "unconfigured"
        if not credential.enabled or not credential.verified_at:
            return user, model, None, "unverified"
        try:
            credentials = (
                decrypt_secret(credential.token_ciphertext),
                decrypt_secret(credential.chat_id_ciphertext),
            )
        except Exception as exc:
            logger.warning("Không đọc được cấu hình Zalo webcam: %s", type(exc).__name__)
            return user, model, None, "error"
        return user, model, credentials, "verified"


def load_model(path: str):
    global _cached_model_path, _cached_model
    with _model_lock:
        if _cached_model_path != path or _cached_model is None:
            from ultralytics import YOLO

            model = YOLO(path)
            names = {int(key): value for key, value in model.names.items()}
            if names != {0: "khoi", 1: "lua"}:
                raise ValueError("Checkpoint không khớp lớp khoi/lua")
            _cached_model = model
            _cached_model_path = path
        return _cached_model


def predict_frame(model, payload: bytes) -> dict:
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Khung hình vượt giới hạn dung lượng")
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_FRAME_PIXELS:
            raise ValueError("Kích thước khung hình không hợp lệ")
        frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    with inference_slot(blocking=False):
        result = model.predict(
            frame,
            imgsz=REALTIME_IMAGE_SIZE,
            conf=settings.detection_confidence,
            iou=settings.detection_iou,
            device="cpu",
            verbose=False,
        )[0]
    boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in (0, 1):
            continue
        boxes.append({
            "class_id": class_id,
            "confidence": float(box.conf[0].item()),
            "xyxy": [float(value) for value in box.xyxy[0].tolist()],
        })
    return {"width": width, "height": height, "detections": boxes}


def save_webcam_evidence(
    payload: bytes, detections: list[dict], relative_path: str | None = None
) -> str:
    """Lưu ảnh webcam có box; mặc định dùng đường dẫn tạm phục vụ gửi Zalo."""
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không giải mã được khung webcam để gửi ảnh cảnh báo")

    height, width = frame.shape[:2]
    scale = max(0.5, min(width / 960, 1.0))
    for detection in detections:
        x1, y1, x2, y2 = [round(value) for value in detection["xyxy"]]
        class_id = int(detection["class_id"])
        label = ("LỬA" if class_id == 1 else "KHÓI") + f" {float(detection['confidence']):.0%}"
        color = (0, 0, 255) if class_id == 1 else (255, 150, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(2 * scale)))
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            max(1, round(2 * scale)),
        )
        text_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(frame, (x1, text_top), (x1 + text_width + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(text_height, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (255, 255, 255),
            max(1, round(2 * scale)),
            cv2.LINE_AA,
        )

    encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded_ok:
        raise OSError("Không mã hóa được ảnh minh chứng webcam")
    relative_path = relative_path or f"tmp/zalo_evidence/{new_id()}.jpg"
    target = runtime_file(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded.tobytes())
    return relative_path


@router.websocket("/api/realtime")
async def realtime(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in {settings.app_origin, settings.api_origin}:
        await websocket.close(code=4403, reason="Nguồn kết nối không hợp lệ")
        return
    user, version, zalo_credentials, zalo_status = authenticated_user(websocket)
    if not user:
        await websocket.close(code=4401, reason="Vui lòng đăng nhập lại")
        return
    if not version:
        await websocket.close(code=1011, reason="Chưa có mô hình đang hoạt động")
        return
    try:
        path = model_file(version)
    except FileNotFoundError:
        await websocket.close(code=1011, reason="Không tìm thấy checkpoint đang hoạt động")
        return

    with _active_sessions_lock:
        if user.id in ACTIVE_REALTIME_USERS:
            await websocket.close(code=4409, reason="Tài khoản đã có một phiên webcam đang chạy")
            return
        ACTIVE_REALTIME_USERS.add(user.id)
    try:
        await websocket.accept()
    except Exception:
        with _active_sessions_lock:
            ACTIVE_REALTIME_USERS.discard(user.id)
        raise
    send_lock = asyncio.Lock()
    tracker = RealtimeEventTracker()
    session_id = new_id()
    session_started_at = utcnow()
    session_clock_started = time.monotonic()
    active_event_ids: dict[int, str] = {}
    last_any_detection: float | None = None
    realtime_paused_at: float | None = None
    try:
        with SessionLocal.begin() as db:
            db.add(RealtimeSession(
                id=session_id,
                user_id=user.id,
                model_version_id=version.id,
                status="running",
                started_at=session_started_at,
                config_snapshot={
                    "image_size": REALTIME_IMAGE_SIZE,
                    "confidence": settings.detection_confidence,
                    "iou": settings.detection_iou,
                    "frame_interval_ms": 500,
                },
            ))
    except Exception:
        with _active_sessions_lock:
            ACTIVE_REALTIME_USERS.discard(user.id)
        await websocket.close(code=1011, reason="Không thể ghi nhận phiên webcam")
        return

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def deliver_zalo(
        peak_confidences: dict[int, float],
        is_reminder: bool,
        frame_payload: bytes,
        detections: list[dict],
        event_id: str,
        notification_ids: list[str],
    ) -> None:
        credentials = zalo_credentials
        if not credentials:
            return
        relative_image = None
        temporary_image = False
        error_code = None
        try:
            if settings.public_api_origin:
                if not is_reminder:
                    with SessionLocal() as db:
                        evidence = db.scalar(
                            select(RealtimeEvidence)
                            .where(RealtimeEvidence.event_id == event_id).limit(1)
                        )
                        relative_image = evidence.image_path if evidence else None
                if not relative_image:
                    relative_image = await asyncio.to_thread(save_webcam_evidence, frame_payload, detections)
                    temporary_image = True
            image_url = create_public_media_url(relative_image) if relative_image else None
            delivery_kind = await asyncio.to_thread(
                send_alert,
                credentials[0],
                credentials[1],
                realtime_notification_text(peak_confidences, is_reminder),
                image_url,
            )
        except Exception as exc:
            logger.warning("Gửi cảnh báo webcam qua Zalo thất bại: %s", type(exc).__name__)
            error_code = type(exc).__name__
            status = "failed"
            message = "Không gửi được thông báo webcam qua Zalo. Hãy kiểm tra bot và kết nối mạng."
        else:
            status = "sent"
            if delivery_kind == "photo":
                message = "Đã gửi cảnh báo webcam kèm ảnh khung hình có box nhận diện qua Zalo."
            elif image_url:
                message = "Đã gửi cảnh báo webcam dạng chữ; Zalo không nhận được ảnh minh chứng."
            else:
                message = "Đã gửi cảnh báo webcam dạng chữ. Muốn đính kèm ảnh, cần cấu hình PUBLIC_API_ORIGIN bằng địa chỉ HTTPS công khai của backend."
            if is_reminder:
                message = "Tin nhắc lại: " + message[0].lower() + message[1:]
        finally:
            if relative_image and temporary_image:
                try:
                    runtime_file(relative_image).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Không xóa được ảnh webcam tạm sau khi gửi Zalo")
            with SessionLocal.begin() as db:
                for notification_id in notification_ids:
                    log_row = db.get(RealtimeNotificationLog, notification_id)
                    if not log_row:
                        continue
                    log_row.status = status
                    log_row.attempt_count += 1
                    log_row.sent_at = utcnow() if status == "sent" else None
                    log_row.error_code = error_code
        try:
            await send_json({"type": "zalo_status", "status": status, "message": message})
        except (WebSocketDisconnect, RuntimeError):
            pass

    session_final_status = "stopped"
    session_error_code = None
    try:
        model = await asyncio.to_thread(load_model, str(path))
        await send_json({
            "type": "ready",
            "model": version.name,
            "zalo_status": zalo_status,
            "session_id": session_id,
            "session_started_at": session_started_at.isoformat(),
        })
        while True:
            payload = await websocket.receive_bytes()
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(predict_frame, model, payload)
                now = time.monotonic()
                if realtime_paused_at is not None:
                    paused_seconds = max(0.0, now - realtime_paused_at)
                    if last_any_detection is not None:
                        last_any_detection += paused_seconds
                    realtime_paused_at = None
                tracker.resume(now)
                confidences: dict[int, float] = {}
                for detection in result["detections"]:
                    class_id = detection["class_id"]
                    confidences[class_id] = max(confidences.get(class_id, 0.0), detection["confidence"])
                event = tracker.update(confidences, now)
                elapsed_ms = max(0, round((now - session_clock_started) * 1000))
                detected_classes = set(confidences)
                if detected_classes:
                    last_any_detection = now
                event_snapshots: list[tuple[str, int, list[dict]]] = []
                with SessionLocal.begin() as db:
                    session_row = db.get(RealtimeSession, session_id)
                    if session_row:
                        session_row.duration_ms = elapsed_ms
                        session_row.frame_count += 1
                        if result["detections"]:
                            session_row.detected_frame_count += 1
                    for class_id, state in tracker.classes.items():
                        if not state.confirmed:
                            continue
                        if class_id not in active_event_ids:
                            event_id = new_id()
                            active_event_ids[class_id] = event_id
                            selected_detections = [
                                item for item in result["detections"] if item["class_id"] == class_id
                            ]
                            event_snapshots.append((event_id, class_id, selected_detections))
                            db.add(RealtimeEvent(
                                id=event_id,
                                session_id=session_id,
                                class_id=class_id,
                                start_ms=elapsed_ms,
                                end_ms=elapsed_ms,
                                peak_confidence=state.peak_confidence,
                                detections=[
                                    {
                                        "class_id": class_id,
                                        "confidence": item["confidence"],
                                        "xyxy": [
                                            item["xyxy"][0] / result["width"],
                                            item["xyxy"][1] / result["height"],
                                            item["xyxy"][2] / result["width"],
                                            item["xyxy"][3] / result["height"],
                                        ],
                                    }
                                    for item in selected_detections
                                ],
                            ))
                        elif class_id in detected_classes:
                            row = db.get(RealtimeEvent, active_event_ids[class_id])
                            if row:
                                row.end_ms = elapsed_ms
                                row.peak_confidence = max(row.peak_confidence, state.peak_confidence)
                                row.detections = [
                                    {
                                        "class_id": class_id,
                                        "confidence": item["confidence"],
                                        "xyxy": [
                                            item["xyxy"][0] / result["width"],
                                            item["xyxy"][1] / result["height"],
                                            item["xyxy"][2] / result["width"],
                                            item["xyxy"][3] / result["height"],
                                        ],
                                    }
                                    for item in result["detections"] if item["class_id"] == class_id
                                ]
                    if (
                        not detected_classes
                        and last_any_detection is not None
                        and now - last_any_detection >= 5.0
                    ):
                        for event_id in active_event_ids.values():
                            row = db.get(RealtimeEvent, event_id)
                            if row:
                                row.status = "closed"
                                row.end_ms = elapsed_ms
                        active_event_ids.clear()
                        last_any_detection = None
                for event_id, class_id, selected_detections in event_snapshots:
                    relative = f"realtime/{session_id}/{event_id}.jpg"
                    try:
                        await asyncio.to_thread(save_webcam_evidence, payload, selected_detections, relative)
                        with SessionLocal.begin() as db:
                            db.add(RealtimeEvidence(
                                id=new_id(), event_id=event_id, image_path=relative,
                                width=result["width"], height=result["height"],
                            ))
                    except Exception:
                        try:
                            runtime_file(relative).unlink(missing_ok=True)
                        except OSError:
                            logger.warning("Không dọn được ảnh webcam lỗi")
                        logger.exception("Không lưu được ảnh minh chứng webcam")
                await send_json({
                    "type": "prediction",
                    **result,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                })
                if event:
                    notification_event_id = next(
                        (active_event_ids[class_id] for class_id in sorted(event.peak_confidences)
                         if class_id in active_event_ids),
                        None,
                    )
                    notification_ids = []
                    for class_id in sorted(event.peak_confidences):
                        class_event_id = active_event_ids.get(class_id)
                        if not class_event_id:
                            continue
                        notification_id = new_id()
                        notification_ids.append(notification_id)
                        with SessionLocal.begin() as db:
                            db.add(RealtimeNotificationLog(
                                id=notification_id,
                                event_id=class_event_id,
                                recipient_user_id=user.id,
                                status="pending" if zalo_credentials else "skipped",
                                error_code=None if zalo_credentials else "zalo_not_configured",
                            ))
                    if zalo_credentials:
                        if not notification_event_id or not notification_ids:
                            continue
                        task = asyncio.create_task(
                            deliver_zalo(
                                event.peak_confidences,
                                event.is_reminder,
                                payload,
                                result["detections"],
                                notification_event_id,
                                notification_ids,
                            )
                        )
                        BACKGROUND_ZALO_TASKS.add(task)
                        task.add_done_callback(BACKGROUND_ZALO_TASKS.discard)
                    else:
                        if zalo_status == "unverified":
                            zalo_message = "Tín hiệu webcam đã được xác nhận nhưng bot Zalo chưa xác minh. Hãy kết nối bot trong hồ sơ rồi khởi động lại webcam."
                        elif zalo_status == "error":
                            zalo_message = "Không thể đọc cấu hình Zalo. Hãy kết nối lại bot trong hồ sơ."
                        else:
                            zalo_message = "Tín hiệu webcam đã được xác nhận nhưng bot Zalo chưa cấu hình. Hãy cấu hình bot rồi khởi động lại webcam."
                        await send_json({
                            "type": "zalo_status",
                            "status": zalo_status,
                            "message": zalo_message,
                        })
            except InferenceBusyError:
                if realtime_paused_at is None:
                    realtime_paused_at = time.monotonic()
                    tracker.pause(realtime_paused_at)
                await send_json({"type": "busy"})
            except (ValueError, OSError) as exc:
                await send_json({"type": "frame_error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        session_final_status = "failed"
        session_error_code = type(exc).__name__
        logger.exception("Luồng dự đoán webcam gặp lỗi")
        try:
            await websocket.close(code=1011, reason="Không thể tiếp tục dự đoán webcam")
        except RuntimeError:
            pass
    finally:
        with SessionLocal.begin() as db:
            session_row = db.get(RealtimeSession, session_id)
            if session_row:
                session_row.status = session_final_status
                session_row.ended_at = utcnow()
                session_row.error_code = session_error_code
                session_row.duration_ms = max(
                    session_row.duration_ms,
                    round((time.monotonic() - session_clock_started) * 1000),
                )
            for event_id in active_event_ids.values():
                event_row = db.get(RealtimeEvent, event_id)
                if event_row:
                    event_row.status = "closed"
        with _active_sessions_lock:
            ACTIVE_REALTIME_USERS.discard(user.id)
