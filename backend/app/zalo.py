from __future__ import annotations

import logging
from datetime import timedelta, timezone

import httpx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .inference.realtime_events import REMINDER_INTERVAL_SECONDS
from .models import AlertEvent, EventEvidence, NotificationLog, VideoJob, ZaloCredential, utcnow
from .public_media import create_public_media_url
from .security import decrypt_secret


logger = logging.getLogger(__name__)
CLASS_NAMES = {0: "khói", 1: "lửa"}
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
API_ROOT = "https://bot-api.zaloplatforms.com"


class ZaloApiError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.public_message = message


def zalo_api(token: str, method: str, payload: dict, timeout: float = 12) -> dict:
    try:
        response = httpx.post(
            f"{API_ROOT}/bot{token}/{method}",
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise ZaloApiError("timeout", "Zalo phản hồi quá thời gian chờ. Hãy thử lại.") from exc
    except httpx.RequestError as exc:
        raise ZaloApiError(
            "network",
            "Backend không kết nối được máy chủ Zalo Bot API. Kiểm tra Internet trên máy chạy backend.",
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise ZaloApiError("invalid_response", "Zalo trả về phản hồi không hợp lệ.") from exc
    if not isinstance(data, dict):
        raise ZaloApiError("invalid_response", "Zalo trả về phản hồi không hợp lệ.")
    if response.status_code in {401, 403}:
        raise ZaloApiError("invalid_token", "Zalo không chấp nhận Bot Token. Hãy sao chép lại nguyên token từ tin nhắn tạo bot.")
    description = str(data.get("description") or "")
    lower_description = description.lower()
    if "webhook" in lower_description or response.status_code == 409:
        raise ZaloApiError(
            "webhook_conflict",
            "Bot đang dùng webhook nên getUpdates không hoạt động. Xóa webhook của bot rồi thử kết nối lại.",
        )
    if method == "getUpdates" and data.get("error_code") == 408:
        return {"ok": True, "result": []}
    if method == "getWebhookInfo" and (response.status_code == 404 or data.get("error_code") == 404):
        logger.info("Zalo Bot API không hỗ trợ getWebhookInfo; tiếp tục kết nối bằng getUpdates")
        return {"ok": True, "result": None}
    if response.status_code >= 400 or not data.get("ok"):
        error_code = data.get("error_code")
        detail = f" (mã {error_code})" if error_code is not None else f" (HTTP {response.status_code})"
        raise ZaloApiError("api_rejected", f"Zalo Bot API từ chối {method}{detail}. Kiểm tra token và cấu hình bot.")
    return data


def send_message(token: str, chat_id: str, message: str) -> None:
    zalo_api(token, "sendMessage", {"chat_id": chat_id, "text": message})


def send_photo(token: str, chat_id: str, image_url: str, caption: str) -> None:
    zalo_api(
        token,
        "sendPhoto",
        {"chat_id": chat_id, "photo": image_url, "caption": caption[:2000]},
    )


def send_alert(token: str, chat_id: str, message: str, image_url: str | None) -> str:
    """Gửi ảnh kèm mô tả; nếu ảnh không gửi được thì vẫn gửi cảnh báo dạng chữ."""
    if image_url:
        try:
            send_photo(token, chat_id, image_url, message)
            return "photo"
        except Exception as exc:
            logger.warning("Gửi ảnh cảnh báo Zalo thất bại; chuyển sang tin chữ: %s", type(exc).__name__)
            message += "\n\nẢnh minh chứng chưa gửi được; hệ thống gửi kèm thông tin dạng chữ."
    else:
        message += "\n\nẢnh chưa được đính kèm vì chưa cấu hình địa chỉ HTTPS công khai cho backend."
    send_message(token, chat_id, message)
    return "text"


def discover_private_chat_id(token: str) -> str | None:
    # Theo API, getUpdates dùng long polling; chờ đủ lâu để nhận event mới phát sinh.
    data = zalo_api(token, "getUpdates", {"timeout": 30}, timeout=35)
    result = data.get("result")
    updates = result if isinstance(result, list) else [result]
    for update in reversed(updates):
        if not isinstance(update, dict):
            continue
        message = update.get("message")
        # Zalo trả result là một event object, không phải lúc nào cũng là list.
        # Một số phiên bản/adapter có thể trả thẳng message thay vì bọc trong event.
        if not isinstance(message, dict) and isinstance(update.get("chat"), dict):
            message = update
        chat = message.get("chat") if isinstance(message, dict) else None
        if isinstance(chat, dict) and str(chat.get("chat_type", "")).upper() == "PRIVATE" and chat.get("id"):
            logger.info("Zalo getUpdates nhận được tin nhắn riêng; không ghi nội dung hoặc chat_id vào log")
            return str(chat["id"])
    logger.info(
        "Zalo getUpdates không có chat riêng phù hợp; result_type=%s",
        type(result).__name__ if result is not None else "empty",
    )
    return None


def get_webhook_url(token: str) -> str | None:
    data = zalo_api(token, "getWebhookInfo", {})
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    url = result.get("url")
    return str(url).strip() if url else None


def notification_text(event: AlertEvent, job: VideoJob) -> str:
    kind = CLASS_NAMES.get(event.class_id, "lửa/khói")
    time_seconds = event.start_ms / 1000
    end_seconds = event.end_ms / 1000
    duration_seconds = max(0, event.end_ms - event.start_ms) / 1000
    captured_at = utcnow().astimezone(VIETNAM_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    return (
        "CẢNH BÁO PHÁT HIỆN LỬA/KHÓI\n"
        "Nguồn: Video bạn đã tải lên\n"
        f"Video: {job.original_name}\n"
        f"Phát hiện: {kind}\n"
        f"Thời điểm trong video: {time_seconds:.1f}–{end_seconds:.1f} giây\n"
        f"Thời lượng sự kiện: {duration_seconds:.1f} giây\n"
        f"Độ tin cậy cao nhất: {event.peak_confidence:.0%}\n"
        f"Thời gian gửi: {captured_at} (giờ Việt Nam)\n"
        "Lưu ý: Đây là dự đoán của mô hình; vui lòng kiểm tra tình hình thực tế."
    )


def realtime_notification_text(peak_confidences: dict[int, float], is_reminder: bool = False) -> str:
    labels = []
    for class_id, label in ((1, "lửa"), (0, "khói")):
        if class_id in peak_confidences:
            labels.append(f"{label} ({peak_confidences[class_id]:.0%})")
    detected_at = utcnow().astimezone(VIETNAM_TIMEZONE).strftime("%d/%m/%Y %H:%M:%S")
    heading = "NHẮC LẠI CẢNH BÁO WEBCAM" if is_reminder else "CẢNH BÁO PHÁT HIỆN LỬA/KHÓI"
    timing_note = f"Tín hiệu còn xuất hiện; hệ thống nhắc lại tối đa mỗi {int(REMINDER_INTERVAL_SECONDS)} giây.\n" if is_reminder else "Điều kiện xác nhận: ít nhất 2 dự đoán liên tiếp trong tối thiểu 0,5 giây.\n"
    time_label = "Thời điểm nhắc lại" if is_reminder else "Thời điểm xác nhận"
    return (
        f"{heading}\n"
        "Nguồn: Webcam trực tiếp\n"
        f"Mô hình nhận diện: {', '.join(labels) if labels else 'lửa/khói'}\n"
        f"{timing_note}"
        f"{time_label}: {detected_at} (giờ Việt Nam)\n"
        "Ảnh đính kèm là khung hình tại thời điểm cảnh báo, có vẽ hộp nhận diện.\n"
        "Đây là dự đoán của mô hình; vui lòng kiểm tra tình hình thực tế."
    )


def deliver_pending(db: Session, limit: int = 10) -> int:
    rows = db.scalars(
        select(NotificationLog)
        .where(
            NotificationLog.channel == "zalo",
            or_(
                NotificationLog.status == "pending",
                (NotificationLog.status == "failed") & (NotificationLog.error_code == "ZoneInfoNotFoundError"),
            ),
        )
        .order_by(NotificationLog.id)
        .limit(limit)
    ).all()
    delivered = 0
    for row in rows:
        event = db.get(AlertEvent, row.event_id)
        job = db.get(VideoJob, event.job_id) if event else None
        credential = db.get(ZaloCredential, row.recipient_user_id)
        if not event or not job or not credential or not credential.enabled or not credential.verified_at:
            row.status = "failed"
            row.error_code = "credential_unavailable"
            db.commit()
            continue
        row.status = "sending"
        row.attempt_count += 1
        db.commit()
        try:
            evidence = db.scalar(select(EventEvidence).where(EventEvidence.event_id == event.id).limit(1))
            image_url = create_public_media_url(evidence.image_path) if evidence else None
            send_alert(
                decrypt_secret(credential.token_ciphertext),
                decrypt_secret(credential.chat_id_ciphertext),
                notification_text(event, job),
                image_url,
            )
        except Exception as exc:
            row.status = "failed"
            row.error_code = type(exc).__name__
            logger.warning("Gửi Zalo thất bại cho thông báo %s: %s", row.id, type(exc).__name__)
        else:
            row.status = "sent"
            row.sent_at = utcnow()
            row.error_code = None
            delivered += 1
        db.commit()
    return delivered
