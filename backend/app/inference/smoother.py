from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..models import new_id


@dataclass
class EventCandidate:
    id: str
    class_id: int
    start_ms: int
    end_ms: int
    peak_confidence: float
    peak_frame_index: int
    detection_id: str | None
    image_path: str | None = None
    last_positive_ms: int = 0


class AlertSmoother:
    def __init__(
        self,
        window_ms: int,
        trigger_ratio: float,
        release_ratio: float,
        min_event_ms: int,
        merge_gap_ms: int,
    ) -> None:
        self.window_ms = window_ms
        self.trigger_ratio = trigger_ratio
        self.release_ratio = release_ratio
        self.min_event_ms = min_event_ms
        self.merge_gap_ms = merge_gap_ms
        self.history: dict[int, deque[tuple[int, float]]] = {0: deque(), 1: deque()}
        self.active: dict[int, EventCandidate | None] = {0: None, 1: None}
        self.finished: list[EventCandidate] = []
        self.discarded_paths: list[str] = []

    def update(
        self,
        time_ms: int,
        frame_index: int,
        confidences: dict[int, tuple[float, str | None]],
    ) -> list[EventCandidate]:
        peaks: list[EventCandidate] = []
        for class_id in (0, 1):
            confidence, detection_id = confidences.get(class_id, (0.0, None))
            history = self.history[class_id]
            history.append((time_ms, confidence))
            while history and history[0][0] < time_ms - self.window_ms:
                history.popleft()
            ratio = sum(value > 0 for _, value in history) / len(history)
            current = self.active[class_id]
            if current is None:
                observed_ms = time_ms - history[0][0]
                if confidence > 0 and len(history) >= 3 and observed_ms >= min(500, self.window_ms // 2) and ratio >= self.trigger_ratio:
                    positives = [point_ms for point_ms, value in history if value > 0]
                    current = EventCandidate(
                        id=new_id(),
                        class_id=class_id,
                        start_ms=positives[0],
                        end_ms=time_ms,
                        peak_confidence=confidence,
                        peak_frame_index=frame_index,
                        detection_id=detection_id,
                        last_positive_ms=time_ms,
                    )
                    self.active[class_id] = current
                    peaks.append(current)
                continue
            if confidence > 0:
                current.last_positive_ms = time_ms
            if confidence > current.peak_confidence:
                current.peak_confidence = confidence
                current.peak_frame_index = frame_index
                current.detection_id = detection_id
                peaks.append(current)
            if ratio <= self.release_ratio:
                current.end_ms = current.last_positive_ms
                if current.end_ms - current.start_ms >= self.min_event_ms:
                    self.finished.append(current)
                elif current.image_path:
                    self.discarded_paths.append(current.image_path)
                self.active[class_id] = None
        return peaks

    def close(self) -> tuple[list[EventCandidate], list[str]]:
        discarded_paths = self.discarded_paths.copy()
        for class_id in (0, 1):
            current = self.active[class_id]
            if current:
                current.end_ms = current.last_positive_ms
                if current.end_ms - current.start_ms >= self.min_event_ms:
                    self.finished.append(current)
                elif current.image_path:
                    discarded_paths.append(current.image_path)
                self.active[class_id] = None
        ordered = sorted(self.finished, key=lambda event: (event.class_id, event.start_ms))
        merged: list[EventCandidate] = []
        for event in ordered:
            if merged and merged[-1].class_id == event.class_id and event.start_ms - merged[-1].end_ms <= self.merge_gap_ms:
                previous = merged[-1]
                previous.end_ms = max(previous.end_ms, event.end_ms)
                if event.peak_confidence > previous.peak_confidence:
                    if previous.image_path:
                        discarded_paths.append(previous.image_path)
                    previous.peak_confidence = event.peak_confidence
                    previous.peak_frame_index = event.peak_frame_index
                    previous.detection_id = event.detection_id
                    previous.image_path = event.image_path
                elif event.image_path:
                    discarded_paths.append(event.image_path)
            else:
                merged.append(event)
        return sorted(merged, key=lambda event: event.start_ms), discarded_paths
