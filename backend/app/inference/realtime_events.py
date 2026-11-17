from __future__ import annotations

from dataclasses import dataclass


MIN_DETECTIONS = 2
MIN_EVENT_SECONDS = 0.5
EVENT_GAP_SECONDS = 5.0
REMINDER_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class RealtimeAlert:
    peak_confidences: dict[int, float]
    is_reminder: bool


@dataclass
class ClassState:
    count: int = 0
    first_seen: float | None = None
    last_seen: float | None = None
    peak_confidence: float = 0.0
    confirmed: bool = False

    def reset(self) -> None:
        self.count = 0
        self.first_seen = None
        self.last_seen = None
        self.peak_confidence = 0.0
        self.confirmed = False


class RealtimeEventTracker:
    """Gom các dự đoán webcam thành một sự kiện để tránh gửi Zalo lặp."""

    def __init__(self) -> None:
        self.classes = {0: ClassState(), 1: ClassState()}
        self.last_any_seen: float | None = None
        self.notification_sent = False
        self.last_notification_at: float | None = None
        self.paused_at: float | None = None

    def pause(self, now: float) -> None:
        if self.paused_at is None:
            self.paused_at = now

    def resume(self, now: float) -> None:
        if self.paused_at is None:
            return
        paused_seconds = max(0.0, now - self.paused_at)
        if self.last_any_seen is not None:
            self.last_any_seen += paused_seconds
        if self.last_notification_at is not None:
            self.last_notification_at += paused_seconds
        for state in self.classes.values():
            if state.first_seen is not None:
                state.first_seen += paused_seconds
            if state.last_seen is not None:
                state.last_seen += paused_seconds
        self.paused_at = None

    def update(self, confidences: dict[int, float], now: float) -> RealtimeAlert | None:
        if (
            not confidences
            and self.last_any_seen is not None
            and now - self.last_any_seen >= EVENT_GAP_SECONDS
        ):
            for state in self.classes.values():
                state.reset()
            self.notification_sent = False
            self.last_notification_at = None

        newly_confirmed: dict[int, float] = {}
        for class_id, state in self.classes.items():
            confidence = confidences.get(class_id)
            if confidence is None:
                if not state.confirmed:
                    state.reset()
                continue

            if state.confirmed:
                state.last_seen = now
                state.peak_confidence = max(state.peak_confidence, confidence)
                continue

            if state.last_seen is None:
                state.count = 1
                state.first_seen = now
                state.peak_confidence = confidence
            else:
                state.count += 1
                state.peak_confidence = max(state.peak_confidence, confidence)
            state.last_seen = now

            if state.count >= MIN_DETECTIONS and state.first_seen is not None:
                if now - state.first_seen >= MIN_EVENT_SECONDS:
                    state.confirmed = True
                    newly_confirmed[class_id] = state.peak_confidence

        if confidences:
            self.last_any_seen = now

        if newly_confirmed and not self.notification_sent:
            self.notification_sent = True
            self.last_notification_at = now
            return RealtimeAlert(
                peak_confidences={
                    class_id: state.peak_confidence
                    for class_id, state in self.classes.items()
                    if state.confirmed
                },
                is_reminder=False,
            )

        if (
            self.notification_sent
            and self.last_notification_at is not None
            and now - self.last_notification_at >= REMINDER_INTERVAL_SECONDS
        ):
            active_confidences = {
                class_id: confidences[class_id]
                for class_id, state in self.classes.items()
                if state.confirmed and class_id in confidences
            }
            if active_confidences:
                self.last_notification_at = now
                return RealtimeAlert(
                    peak_confidences=active_confidences,
                    is_reminder=True,
                )
        return None
