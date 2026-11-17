from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.app.media import stream_video


def test_video_byte_ranges_support_seeking(tmp_path: Path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"0123456789")
    app = FastAPI()

    @app.get("/video")
    def video(request: Request):
        return stream_video(path, request)

    client = TestClient(app)
    part = client.get("/video", headers={"Range": "bytes=3-6"})
    assert part.status_code == 206
    assert part.content == b"3456"
    assert part.headers["Content-Range"] == "bytes 3-6/10"
    assert part.headers["Accept-Ranges"] == "bytes"
    tail = client.get("/video", headers={"Range": "bytes=-3"})
    assert tail.content == b"789"
    assert client.get("/video", headers={"Range": "bytes=20-"}).status_code == 416
