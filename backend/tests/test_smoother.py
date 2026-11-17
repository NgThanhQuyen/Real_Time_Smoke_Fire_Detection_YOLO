from backend.app.inference.smoother import AlertSmoother


def make_smoother() -> AlertSmoother:
    return AlertSmoother(
        window_ms=1200,
        trigger_ratio=0.6,
        release_ratio=0.2,
        min_event_ms=600,
        merge_gap_ms=500,
    )


def test_isolated_detections_do_not_create_an_event():
    smoother = make_smoother()
    for index in range(20):
        confidence = 0.7 if index in {2, 8, 14} else 0.0
        smoother.update(index * 100, index, {0: (confidence, None)})
    events, _ = smoother.close()
    assert events == []


def test_sustained_detection_creates_one_event():
    smoother = make_smoother()
    for index in range(25):
        confidence = 0.7 if 2 <= index <= 17 else 0.0
        smoother.update(index * 100, index, {1: (confidence, None)})
    events, _ = smoother.close()
    assert len(events) == 1
    assert events[0].class_id == 1
    assert events[0].start_ms <= 700
    assert events[0].end_ms >= 1700


def test_sustained_false_positive_is_not_magically_removed():
    smoother = make_smoother()
    for index in range(20):
        smoother.update(index * 100, index, {0: (0.6, None)})
    events, _ = smoother.close()
    assert len(events) == 1
