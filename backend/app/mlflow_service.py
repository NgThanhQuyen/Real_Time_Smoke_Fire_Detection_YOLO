from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from fastapi import HTTPException, status
from mlflow.tracking import MlflowClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import ROOT, RUNTIME, runtime_file, settings
from .models import AuditLog, ModelVersion, RealtimeSession, VideoJob, new_id, utcnow
from .serializers import model_data


def setup_mlflow_env() -> bool:
    """Thiết lập biến môi trường để MLflow Client xác thực với DagsHub."""
    if not settings.mlflow_tracking_uri:
        return False
    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
    if settings.mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    if settings.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
    return True


def get_client() -> MlflowClient:
    if not setup_mlflow_env():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chưa cấu hình MLFLOW_TRACKING_URI trong file .env",
        )
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def check_mlflow_connection() -> dict[str, Any]:
    if not settings.mlflow_tracking_uri:
        return {
            "configured": False,
            "connected": False,
            "message": "Chưa cấu hình MLFLOW_TRACKING_URI trong file .env",
            "tracking_uri": "",
        }
    try:
        client = get_client()
        experiments = client.search_experiments(max_results=5)
        return {
            "configured": True,
            "connected": True,
            "message": f"Kết nối DagsHub MLflow thành công ({len(experiments)} experiment)",
            "tracking_uri": settings.mlflow_tracking_uri,
        }
    except Exception as exc:
        return {
            "configured": True,
            "connected": False,
            "message": f"Không thể kết nối đến DagsHub MLflow: {str(exc)}",
            "tracking_uri": settings.mlflow_tracking_uri,
        }


def find_checkpoint_in_artifacts(client: MlflowClient, run_id: str, path: str = "") -> str | None:
    """Đệ quy tìm file .pt trong artifact store của run."""
    try:
        artifacts = client.list_artifacts(run_id, path=path)
        for item in artifacts:
            if not item.is_dir and item.path.endswith(".pt"):
                return item.path
            if item.is_dir and item.path in ("model", "weights", "checkpoints", "runs"):
                found = find_checkpoint_in_artifacts(client, run_id, path=item.path)
                if found:
                    return found
    except Exception:
        pass
    return None


def list_remote_runs(db: Session, max_results: int = 20) -> list[dict[str, Any]]:
    client = get_client()
    try:
        experiments = client.search_experiments()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Lỗi khi truy vấn danh sách thí nghiệm từ DagsHub MLflow: {exc}",
        )

    if not experiments:
        return []

    exp_ids = [exp.experiment_id for exp in experiments]
    try:
        runs = client.search_runs(
            experiment_ids=exp_ids,
            order_by=["attribute.start_time DESC"],
            max_results=max_results,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Lỗi khi tìm kiếm runs từ DagsHub MLflow: {exc}",
        )

    # Lấy danh sách tên model đã có trong hệ thống để đánh dấu
    existing_names = set(db.scalars(select(ModelVersion.name)).all())

    results: list[dict[str, Any]] = []
    for run in runs:
        run_id = run.info.run_id
        run_name = run.data.tags.get("mlflow.runName") or run.data.tags.get("run_name") or run_id[:8]
        start_time_iso = None
        if run.info.start_time:
            start_time_iso = datetime.fromtimestamp(run.info.start_time / 1000.0, tz=timezone.utc).isoformat()

        checkpoint_path = find_checkpoint_in_artifacts(client, run_id)

        results.append({
            "run_id": run_id,
            "run_name": run_name,
            "status": run.info.status,
            "start_time": start_time_iso,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "has_checkpoint": bool(checkpoint_path),
            "checkpoint_path": checkpoint_path or "",
            "is_imported": run_name in existing_names,
        })
    return results


def import_run_model(
    run_id: str,
    model_name: str,
    artifact_path: str,
    actor_id: str,
    db: Session,
    check_checkpoint_fn: Any,
    activate_now: bool = True,
) -> dict[str, Any]:
    """Tải checkpoint .pt từ DagsHub MLflow về và đăng ký/kích hoạt ModelVersion."""
    client = get_client()

    name = model_name.strip()[:120]
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tên model không được để trống")
    if db.scalar(select(ModelVersion.id).where(ModelVersion.name == name)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tên phiên bản mô hình '{name}' đã tồn tại trong hệ thống",
        )

    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy MLflow Run với id {run_id}: {exc}",
        )

    target_artifact = artifact_path.strip()
    if not target_artifact:
        # Tự động dò artifact
        detected = find_checkpoint_in_artifacts(client, run_id)
        if not detected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Không tìm thấy file checkpoint (.pt) nào trong artifacts của Run {run_id}",
            )
        target_artifact = detected

    model_id = new_id()
    relative = f"models/{model_id}.pt"
    dest_path = runtime_file(relative)

    temp_dir = Path(tempfile.mkdtemp(dir=str(RUNTIME / "tmp")))
    try:
        try:
            downloaded_file = client.download_artifacts(run_id, target_artifact, dst_path=str(temp_dir))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Không thể tải checkpoint '{target_artifact}' từ MLflow: {exc}",
            )

        downloaded_path = Path(downloaded_file)
        if not downloaded_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Artifact tải về không phải là file hợp lệ: {target_artifact}",
            )

        # Kiểm tra tính toàn vẹn và class map của checkpoint PyTorch
        check_checkpoint_fn(downloaded_path)

        # Di chuyển vào runtime/models/{model_id}.pt
        shutil.move(str(downloaded_path), str(dest_path))

        digest = hashlib.sha256(dest_path.read_bytes()).hexdigest()

        # Tổng hợp metrics từ run
        metrics_dict: dict[str, Any] = dict(run.data.metrics)
        # Chuẩn hóa các metrics thường dùng nếu có
        config_dict: dict[str, Any] = {
            "mlflow_run_id": run_id,
            "mlflow_run_name": run.data.tags.get("mlflow.runName", ""),
            "mlflow_artifact_path": target_artifact,
            "params": run.data.params,
        }
        if "imgsz" in run.data.params:
            try:
                config_dict["imgsz"] = int(run.data.params["imgsz"])
            except ValueError:
                pass

        status_value = "active" if activate_now else "candidate"
        activated_at_value = utcnow() if activate_now else None

        if activate_now:
            running = db.scalar(
                select(func.count()).select_from(VideoJob).where(VideoJob.status.in_(["processing", "finalizing"]))
            )
            running_realtime = db.scalar(
                select(func.count()).select_from(RealtimeSession).where(RealtimeSession.status == "running")
            )
            if running or running_realtime:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "Hãy chờ tác vụ video/webcam đang xử lý hoàn tất trước khi kích hoạt mô hình mới",
                )
            for active_item in db.scalars(select(ModelVersion).where(ModelVersion.status == "active")).all():
                active_item.status = "retired"

        item = ModelVersion(
            id=model_id,
            name=name,
            artifact_path=f"runtime/{relative}",
            sha256=digest,
            class_map={"0": "khoi", "1": "lua"},
            metrics=metrics_dict,
            config=config_dict,
            status=status_value,
            activated_at=activated_at_value,
            created_by=actor_id,
        )
        db.add(item)
        db.add(
            AuditLog(
                actor_id=actor_id,
                action="model_imported_and_activated" if activate_now else "model_imported_mlflow",
                target_type="model_version",
                target_id=item.id,
                metadata_json={
                    "mlflow_run_id": run_id,
                    "model_name": name,
                    "artifact_path": target_artifact,
                    "activated": activate_now,
                },
            )
        )
        db.commit()
        return model_data(item)
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

