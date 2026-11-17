"""Tạo notebook Kaggle dự đoán sáu video bằng checkpoint từ ZIP train."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "model_training" / "predict_smoke_fire_videos_kaggle.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(r"""
    # Dự đoán lửa và khói trên sáu video Kaggle

    Notebook đọc `best.pt` từ **ZIP kết quả train** đã gắn vào Kaggle Input, rồi xử lý lần lượt `test_1.mp4` đến `test_6.mp4` của dataset **Video_test_YOLO**. Mặc định là run `hardneg500`; có thể đổi `MODEL_RUN_ID` để thử checkpoint run cũ trên cùng video.

    **Trước khi chạy:** gắn hai Input vào notebook: (1) output của notebook train hoặc dataset chứa `*_results.zip`; (2) dataset `Video_test_YOLO`. Bật Internet nếu môi trường cần cài đúng phiên bản Ultralytics. GPU giúp chạy nhanh hơn, CPU vẫn dùng được. Nếu tên thư mục Kaggle khác dự kiến, sửa `MODEL_SOURCE_OVERRIDE` hoặc `VIDEO_ROOT_OVERRIDE` trong ô cấu hình.

    Đầu ra gồm sáu video MP4 có box, CSV từng box theo thời gian, bảng tổng kết, ảnh xem nhanh và thông tin checkpoint. Notebook đóng gói mọi kết quả vào **một ZIP** trong `/kaggle/working`, kiểm tra ZIP rồi dọn đúng thư mục tạm do notebook tạo. Input Kaggle không bị sửa. Số box/khung có phát hiện là thống kê dự đoán, không phải Precision, Recall hoặc số sự kiện cháy thật vì video không có nhãn chuẩn.
    """),
    markdown("## 1. Chuẩn bị môi trường"),
    code(r"""
    from __future__ import annotations

    import csv
    import hashlib
    import importlib.metadata
    import json
    import math
    import platform
    import shutil
    import subprocess
    import sys
    import time
    import zipfile
    from collections import Counter
    from datetime import datetime, timezone
    from pathlib import Path

    REQUIRED_ULTRALYTICS = "8.4.203"
    try:
        installed_ultralytics = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        installed_ultralytics = None
    if installed_ultralytics != REQUIRED_ULTRALYTICS:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet",
            f"ultralytics=={REQUIRED_ULTRALYTICS}",
        ])

    import cv2
    import pandas as pd
    import torch
    import ultralytics
    from IPython.display import FileLink, Image as NotebookImage, Video as NotebookVideo, display
    from ultralytics import YOLO

    print("Python:", platform.python_version())
    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    """),
    markdown("## 2. Chọn checkpoint và video đầu vào"),
    code(r"""
    INPUT_ROOT = Path("/kaggle/input")
    WORK_ROOT = Path("/kaggle/working")

    # Mặc định dùng kết quả lần train thêm 500 ảnh nền khó.
    MODEL_RUN_ID = "smoke_fire_v2_yolo11n_hardneg500_20261110_040250"
    MODEL_SOURCE_OVERRIDE = None  # Đường dẫn best.pt, *_results.zip hoặc thư mục chứa đúng một file đó.
    VIDEO_DATASET_HINT = "video_test_yolo"
    VIDEO_ROOT_OVERRIDE = None  # Đường dẫn thư mục có test_1.mp4 ... test_6.mp4.

    IMAGE_SIZE = 640
    CONFIDENCE = 0.25
    NMS_IOU = 0.7
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    EXPECTED_VIDEO_NAMES = [f"test_{index}.mp4" for index in range(1, 7)]
    RUN_ID = "smoke_fire_video_predict_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RUN_DIR = WORK_ROOT / RUN_ID
    RESULT_ZIP = WORK_ROOT / f"{RUN_ID}_results.zip"

    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Không thấy Kaggle Input: {INPUT_ROOT}")
    if not WORK_ROOT.is_dir():
        raise FileNotFoundError(f"Không thấy Kaggle Working: {WORK_ROOT}")


    def normalize_hint(value: str) -> str:
        return value.lower().replace("-", "_").replace(" ", "_")


    def resolve_model_source() -> Path:
        if MODEL_SOURCE_OVERRIDE:
            target = Path(MODEL_SOURCE_OVERRIDE).expanduser().resolve()
            if target.is_file() and (target.name == "best.pt" or target.name.endswith("_results.zip")):
                return target
            if target.is_dir():
                found = sorted(target.rglob("*_results.zip")) + sorted(target.rglob("best.pt"))
                if len(found) == 1:
                    return found[0]
            raise RuntimeError("MODEL_SOURCE_OVERRIDE phải trỏ đến một best.pt, ZIP kết quả, hoặc thư mục chỉ có một ứng viên.")

        zip_candidates = sorted(
            path for path in INPUT_ROOT.rglob("*_results.zip")
            if MODEL_RUN_ID in path.name
        )
        if len(zip_candidates) == 1:
            return zip_candidates[0]
        if len(zip_candidates) > 1:
            raise RuntimeError(f"Có nhiều ZIP khớp run ID: {zip_candidates}. Hãy đặt MODEL_SOURCE_OVERRIDE.")
        pt_candidates = sorted(
            path for path in INPUT_ROOT.rglob("best.pt")
            if MODEL_RUN_ID in path.as_posix()
        )
        if len(pt_candidates) == 1:
            return pt_candidates[0]
        raise FileNotFoundError(
            f"Không tìm được một output cho run '{MODEL_RUN_ID}' trong {INPUT_ROOT}. "
            "Hãy gắn output train vào Input hoặc đặt MODEL_SOURCE_OVERRIDE."
        )


    def resolve_videos() -> list[Path]:
        search_root = Path(VIDEO_ROOT_OVERRIDE).expanduser().resolve() if VIDEO_ROOT_OVERRIDE else INPUT_ROOT
        if not search_root.is_dir():
            raise FileNotFoundError(f"Không thấy thư mục video: {search_root}")
        discovered = []
        for path in search_root.rglob("*.mp4"):
            if path.name.lower() not in EXPECTED_VIDEO_NAMES:
                continue
            if VIDEO_ROOT_OVERRIDE or VIDEO_DATASET_HINT in normalize_hint(path.as_posix()):
                discovered.append(path)
        result = []
        for name in EXPECTED_VIDEO_NAMES:
            matches = [path for path in discovered if path.name.lower() == name]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Cần đúng một file {name}, tìm thấy {len(matches)}. "
                    "Kiểm tra Input hoặc đặt VIDEO_ROOT_OVERRIDE."
                )
            result.append(matches[0])
        return result


    MODEL_SOURCE = resolve_model_source()
    VIDEO_PATHS = resolve_videos()
    if RUN_DIR.exists() or RESULT_ZIP.exists():
        raise FileExistsError(f"Run đã tồn tại: {RUN_DIR} hoặc {RESULT_ZIP}. Chạy lại ô cấu hình để tạo RUN_ID mới.")
    RUN_DIR.mkdir()
    (RUN_DIR / "videos").mkdir()
    (RUN_DIR / "previews").mkdir()
    (RUN_DIR / "reports").mkdir()
    print("Nguồn model:", MODEL_SOURCE)
    print("Sáu video:")
    for path in VIDEO_PATHS:
        print(" ", path)
    print("Kết quả:", RESULT_ZIP)
    """),
    markdown("## 3. Nạp và xác minh `best.pt`"),
    code(r"""
    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    CHECKPOINT = RUN_DIR / "best.pt"
    train_summary = None
    if MODEL_SOURCE.suffix.lower() == ".zip":
        with zipfile.ZipFile(MODEL_SOURCE) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("ZIP output train có lỗi CRC.")
            if "best.pt" not in archive.namelist():
                raise RuntimeError("ZIP output train không có best.pt ở thư mục gốc.")
            with archive.open("best.pt") as source, CHECKPOINT.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if "reports/run_summary.json" in archive.namelist():
                train_summary = json.loads(archive.read("reports/run_summary.json"))
                if not MODEL_SOURCE_OVERRIDE and train_summary.get("run_id") != MODEL_RUN_ID:
                    raise RuntimeError(f"Run ID trong ZIP không khớp: {train_summary.get('run_id')}")
            if "reports/labelmap.json" in archive.namelist():
                labelmap = json.loads(archive.read("reports/labelmap.json"))
                if [labelmap.get(str(i)) for i in (0, 1)] != ["Fire", "Smoke"]:
                    raise RuntimeError(f"Class map trong ZIP không đúng: {labelmap}")
    else:
        shutil.copy2(MODEL_SOURCE, CHECKPOINT)

    CHECKPOINT_SHA256 = sha256_file(CHECKPOINT)
    if train_summary and train_summary.get("best_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("best.pt không khớp SHA-256 trong run_summary.json.")
    model = YOLO(str(CHECKPOINT))
    names = model.names
    if isinstance(names, dict):
        class_names = [str(names.get(index, names.get(str(index), ""))) for index in (0, 1)]
    else:
        class_names = [str(names[index]) for index in (0, 1)]
    if [name.lower() for name in class_names] != ["fire", "smoke"]:
        raise RuntimeError(f"Checkpoint phải có class 0=Fire, 1=Smoke; thực tế: {names}")

    provenance = {
        "model_source": str(MODEL_SOURCE),
        "model_run_id": train_summary.get("run_id") if train_summary else None,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "class_map": {"0": "Fire", "1": "Smoke"},
        "ultralytics_version": ultralytics.__version__,
        "device": str(DEVICE),
        "imgsz": IMAGE_SIZE,
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
    }
    (RUN_DIR / "reports" / "model_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Checkpoint SHA-256:", CHECKPOINT_SHA256)
    print("Class map:", class_names)
    """),
    markdown(r"""
    ## 4. Dự đoán và xuất MP4

    Đọc từng khung hình để không giữ cả video trong RAM. CSV ghi một dòng cho mỗi box; thời gian tính từ FPS nguồn. MP4 dùng H.264 nếu tìm được FFmpeg, đồng thời giữ âm thanh gốc nếu video có. Nếu không có FFmpeg, notebook giữ bản MP4V và ghi codec trong báo cáo.
    """),
    code(r"""
    def find_ffmpeg() -> str | None:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, OSError):
            return None


    FFMPEG = find_ffmpeg()
    print("FFmpeg:", FFMPEG or "không có; sẽ xuất MP4V")


    def process_video(video_path: Path) -> tuple[dict, list[dict]]:
        video_name = video_path.name
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(f"Metadata video không hợp lệ: {video_path}, fps={fps}, size={width}x{height}")

        temp_video = RUN_DIR / "videos" / f"{video_path.stem}.temporary.mp4"
        output_video = RUN_DIR / "videos" / f"{video_path.stem}_predicted.mp4"
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Không tạo được MP4 tạm: {temp_video}")

        detection_rows = []
        frame_count = 0
        fire_frames = 0
        smoke_frames = 0
        fire_boxes = 0
        smoke_boxes = 0
        max_confidence = 0.0
        first_preview = None
        detection_preview = None
        started = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                prediction = model.predict(
                    source=frame,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    iou=NMS_IOU,
                    device=DEVICE,
                    verbose=False,
                )[0]
                rendered = prediction.plot()
                if rendered.shape[:2] != (height, width):
                    raise RuntimeError(f"Kích thước khung kết quả thay đổi: {video_name}")
                writer.write(rendered)
                if first_preview is None:
                    first_preview = rendered.copy()
                frame_classes = set()
                for box in prediction.boxes:
                    class_id = int(box.cls.item())
                    if class_id not in (0, 1):
                        raise RuntimeError(f"Model dự đoán class ngoài Fire/Smoke: {class_id}")
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    detection_rows.append({
                        "video": video_name,
                        "frame_index": frame_count,
                        "time_seconds": round(frame_count / fps, 4),
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "confidence": round(confidence, 6),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    })
                    frame_classes.add(class_id)
                    max_confidence = max(max_confidence, confidence)
                    if class_id == 0:
                        fire_boxes += 1
                    else:
                        smoke_boxes += 1
                if 0 in frame_classes:
                    fire_frames += 1
                if 1 in frame_classes:
                    smoke_frames += 1
                if frame_classes and detection_preview is None:
                    detection_preview = rendered.copy()
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        if frame_count == 0:
            raise RuntimeError(f"Video không có khung hình giải mã được: {video_path}")
        if declared_frames > 0 and frame_count < declared_frames - 1:
            raise RuntimeError(f"Video bị dừng đọc sớm: {video_name}, đọc {frame_count}/{declared_frames} khung")
        preview = detection_preview if detection_preview is not None else first_preview
        preview_path = RUN_DIR / "previews" / f"{video_path.stem}_preview.jpg"
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"Không lưu được ảnh xem nhanh: {preview_path}")

        if FFMPEG:
            command = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(temp_video), "-i", str(video_path),
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-t", f"{frame_count / fps:.6f}", "-movflags", "+faststart",
                str(output_video),
            ]
            try:
                subprocess.run(command, check=True)
                temp_video.unlink()
                video_codec = "h264"
            except subprocess.CalledProcessError:
                output_video.unlink(missing_ok=True)
                temp_video.rename(output_video)
                video_codec = "mp4v"
                print(f"FFmpeg không mã hóa được H.264 cho {video_name}; giữ MP4V.")
        else:
            temp_video.rename(output_video)
            video_codec = "mp4v"
        verify = cv2.VideoCapture(str(output_video))
        readable, _ = verify.read()
        verify.release()
        if not readable or output_video.stat().st_size == 0:
            raise RuntimeError(f"Video đầu ra không đọc được: {output_video}")

        summary = {
            "video": video_name,
            "source_fps": round(fps, 4),
            "width": width,
            "height": height,
            "frames_processed": frame_count,
            "duration_seconds": round(frame_count / fps, 3),
            "frames_with_fire": fire_frames,
            "frames_with_smoke": smoke_frames,
            "fire_boxes": fire_boxes,
            "smoke_boxes": smoke_boxes,
            "max_confidence": round(max_confidence, 6),
            "processing_seconds": round(time.perf_counter() - started, 2),
            "output_codec": video_codec,
            "output_video": output_video.relative_to(RUN_DIR).as_posix(),
            "preview": preview_path.relative_to(RUN_DIR).as_posix(),
        }
        return summary, detection_rows


    video_summaries = []
    all_detections = []
    for video_path in VIDEO_PATHS:
        summary, rows = process_video(video_path)
        video_summaries.append(summary)
        all_detections.extend(rows)
        print(
            f"{summary['video']}: {summary['frames_processed']} khung, "
            f"Fire {summary['frames_with_fire']} khung/{summary['fire_boxes']} box, "
            f"Smoke {summary['frames_with_smoke']} khung/{summary['smoke_boxes']} box, "
            f"{summary['output_codec']}"
        )
    """),
    markdown("## 5. Lưu báo cáo và xem nhanh kết quả"),
    code(r"""
    report_dir = RUN_DIR / "reports"
    summary_fields = list(video_summaries[0])
    with (report_dir / "video_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(video_summaries)
    detection_fields = [
        "video", "frame_index", "time_seconds", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2",
    ]
    with (report_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=detection_fields)
        writer.writeheader()
        writer.writerows(all_detections)
    report = {
        "run_id": RUN_ID,
        "model": provenance,
        "videos": video_summaries,
        "total_detection_boxes": len(all_detections),
        "note": "Đây là dự đoán trên video không có ground truth; không tính Precision/Recall/mAP hoặc số sự kiện cháy thật.",
    }
    (report_dir / "video_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "README.md").write_text(
        "# Kết quả dự đoán video\n\n"
        "`videos/` chứa MP4 có bounding box. `previews/` chứa ảnh xem nhanh. "
        "`reports/detections.csv` ghi từng box theo khung và giây trong video; "
        "`reports/video_summary.csv` tổng kết từng video. "
        "`reports/model_provenance.json` ghi checkpoint và cấu hình suy luận.\n\n"
        "Không coi số khung/box được dự đoán là độ chính xác hoặc sự kiện cháy thật.\n",
        encoding="utf-8",
    )
    display(pd.DataFrame(video_summaries)[[
        "video", "frames_processed", "frames_with_fire", "frames_with_smoke",
        "fire_boxes", "smoke_boxes", "max_confidence", "output_codec",
    ]])
    for item in video_summaries:
        print(item["video"])
        display(NotebookImage(filename=str(RUN_DIR / item["preview"]), width=640))
    if video_summaries[0]["output_codec"] == "h264":
        print("Xem thử video đầu tiên:")
        display(NotebookVideo(filename=str(RUN_DIR / video_summaries[0]["output_video"]), embed=True, width=640))
    """),
    markdown(r"""
    ## 6. Đóng gói và dọn thư mục tạm

    Chỉ xóa thư mục `RUN_DIR` sau khi ZIP đã tạo và kiểm tra CRC thành công. Nếu bước xử lý lỗi, thư mục tạm vẫn được giữ để kiểm tra.
    """),
    code(r"""
    def package_and_cleanup(run_dir: Path, result_zip: Path, working_root: Path) -> dict:
        base = working_root.resolve()
        target = run_dir.resolve()
        if target.parent != base or run_dir.is_symlink() or not target.name.startswith("smoke_fire_video_predict_"):
            raise RuntimeError(f"Từ chối dọn thư mục ngoài run này: {target}")
        if result_zip.resolve().parent != base or result_zip.exists():
            raise RuntimeError(f"Đường dẫn ZIP không hợp lệ hoặc đã tồn tại: {result_zip}")
        temp_zip = result_zip.with_suffix(".tmp")
        if temp_zip.exists():
            raise FileExistsError(f"ZIP tạm đã tồn tại: {temp_zip}")

        expected = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file())
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for relative in expected:
                archive.write(run_dir / relative, arcname=relative)
        with zipfile.ZipFile(temp_zip) as archive:
            if archive.testzip() is not None or sorted(archive.namelist()) != expected:
                raise RuntimeError("ZIP kết quả không qua kiểm tra CRC/danh sách file; giữ thư mục run để kiểm tra.")
        temp_zip.rename(result_zip)
        size = result_zip.stat().st_size
        shutil.rmtree(target)
        return {"zip": str(result_zip), "files": len(expected), "bytes": size}


    package_result = package_and_cleanup(RUN_DIR, RESULT_ZIP, WORK_ROOT)
    print("Đã lưu:", package_result)
    display(FileLink(str(RESULT_ZIP)))
    """),
]


def main() -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    nbf.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
