from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

from .config import runtime_file, settings
from .security import decrypt_secret, encrypt_secret


PUBLIC_MEDIA_ROUTE = "/api/public/zalo-evidence/"
PUBLIC_MEDIA_TTL_SECONDS = 15 * 60


def create_public_media_url(relative_path: str) -> str | None:
    """Tạo URL HTTPS có hạn dùng để Zalo tải ảnh minh chứng."""
    origin = settings.public_api_origin
    parsed_origin = urlsplit(origin)
    if parsed_origin.scheme != "https" or not parsed_origin.netloc:
        return None

    image_path = runtime_file(relative_path)
    if not image_path.is_file():
        return None

    payload = json.dumps(
        {"path": relative_path, "expires_at": int(time.time()) + PUBLIC_MEDIA_TTL_SECONDS},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    token = encrypt_secret(payload)
    return f"{origin}{PUBLIC_MEDIA_ROUTE}{quote(token, safe='')}"


def resolve_public_media_token(token: str) -> Path:
    """Xác thực URL tạm và chỉ trả ảnh nằm trong thư mục runtime."""
    payload = json.loads(decrypt_secret(token))
    relative_path = payload.get("path") if isinstance(payload, dict) else None
    expires_at = payload.get("expires_at") if isinstance(payload, dict) else None
    if not isinstance(relative_path, str) or not isinstance(expires_at, int):
        raise ValueError("Liên kết ảnh không hợp lệ")
    if expires_at < int(time.time()):
        raise ValueError("Liên kết ảnh đã hết hạn")

    image_path = runtime_file(relative_path)
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Định dạng ảnh không được hỗ trợ")
    if not image_path.is_file() or image_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Không tìm thấy ảnh hoặc ảnh vượt giới hạn")
    return image_path
