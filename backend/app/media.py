from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse


RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")
CHUNK_SIZE = 1024 * 1024


def stream_video(path: Path, request: Request) -> StreamingResponse:
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy video")
    size = path.stat().st_size
    if size == 0:
        raise HTTPException(404, "Video trống")
    range_header = request.headers.get("range")
    start, end = 0, size - 1
    code = 200
    if range_header:
        match = RANGE_PATTERN.fullmatch(range_header.strip())
        if not match or not any(match.groups()):
            raise HTTPException(416, "Khoảng byte không hợp lệ", headers={"Content-Range": f"bytes */{size}"})
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            suffix = int(last)
            start = max(size - suffix, 0)
        if start >= size or end < start:
            raise HTTPException(416, "Khoảng byte không hợp lệ", headers={"Content-Range": f"bytes */{size}"})
        code = 206

    def chunks():
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining:
                data = handle.read(min(CHUNK_SIZE, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "private, max-age=0",
    }
    if code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=code, media_type="video/mp4", headers=headers)
