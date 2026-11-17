from __future__ import annotations

from datetime import datetime, timezone

from .models import AlertEvent, Detection, EventEvidence, Feedback, ModelVersion, RealtimeEvent, RealtimeEvidence, RealtimeSession, User, VideoJob


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def user_data(user: User, include_profile: bool = True) -> dict:
    data = {
        "id": user.id,
        "username": user.username,
        "active": user.active,
        "roles": sorted(item.role for item in user.roles),
        "created_at": iso(user.created_at),
    }
    if include_profile:
        profile = user.profile
        data["profile"] = {
            "full_name": profile.full_name if profile else None,
            "phone": profile.phone if profile else None,
            "address": profile.address if profile else None,
        }
    return data


def job_data(job: VideoJob, events_count: int = 0) -> dict:
    return {
        "id": job.id,
        "original_name": job.original_name,
        "status": job.status,
        "duration_ms": job.duration_ms,
        "source_fps": job.source_fps,
        "total_frames": job.total_frames,
        "processed_frames": job.processed_frames,
        "detected_frames": job.detected_frames,
        "inference_ms": job.inference_ms,
        "processing_ms": job.processing_ms,
        "model_version_id": job.model_version_id,
        "error_code": job.error_code,
        "created_at": iso(job.created_at),
        "started_at": iso(job.started_at),
        "completed_at": iso(job.completed_at),
        "events_count": events_count,
    }


def detection_data(item: Detection) -> dict:
    return {
        "id": item.id,
        "frame_index": item.frame_index,
        "video_time_ms": item.video_time_ms,
        "class_id": item.class_id,
        "confidence": round(item.confidence, 4),
        "box": [item.x1, item.y1, item.x2, item.y2],
    }


def event_data(item: AlertEvent, evidence: EventEvidence | None = None) -> dict:
    return {
        "id": item.id,
        "job_id": item.job_id,
        "source_type": "video",
        "class_id": item.class_id,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "peak_confidence": round(item.peak_confidence, 4),
        "created_at": iso(item.created_at),
        "evidence_id": evidence.id if evidence else None,
    }


def realtime_session_data(item: RealtimeSession, event_count: int = 0) -> dict:
    return {
        "id": item.id,
        "user_id": item.user_id,
        "model_version_id": item.model_version_id,
        "status": item.status,
        "started_at": iso(item.started_at),
        "ended_at": iso(item.ended_at),
        "duration_ms": item.duration_ms,
        "frame_count": item.frame_count,
        "detected_frame_count": item.detected_frame_count,
        "config_snapshot": item.config_snapshot,
        "error_code": item.error_code,
        "events_count": event_count,
    }


def realtime_event_data(item: RealtimeEvent, evidence: RealtimeEvidence | None = None) -> dict:
    return {
        "id": item.id,
        "session_id": item.session_id,
        "class_id": item.class_id,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
        "peak_confidence": round(item.peak_confidence, 4),
        "status": item.status,
        "created_at": iso(item.created_at),
        "evidence_id": evidence.id if evidence else None,
    }


def feedback_data(item: Feedback) -> dict:
    return {
        "id": item.id,
        "job_id": item.job_id,
        "realtime_session_id": item.realtime_session_id,
        "realtime_event_id": item.realtime_event_id,
        "user_id": item.user_id,
        "detection_id": item.detection_id,
        "video_time_ms": item.video_time_ms,
        "image_url": f"/api/feedback/{item.id}/image",
        "type": item.type,
        "expected_class": item.expected_class,
        "proposed_box": item.proposed_box,
        "comment": item.comment,
        "status": item.status,
        "created_at": iso(item.created_at),
    }


def model_data(item: ModelVersion) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "sha256": item.sha256,
        "class_map": item.class_map,
        "metrics": item.metrics,
        "config": item.config,
        "status": item.status,
        "created_at": iso(item.created_at),
        "activated_at": iso(item.activated_at),
    }
