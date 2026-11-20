from __future__ import annotations

import tempfile
from pathlib import Path

from backend.app.benchmark_service import (
    compare_benchmarks_metrics,
    ensure_dataset_yaml,
    get_dataset_status,
)


def test_dataset_status_detection():
    status = get_dataset_status()
    assert isinstance(status, dict)
    assert "is_ready" in status
    assert "images_count" in status
    assert "labels_count" in status
    assert "folder_path" in status
    assert status["recommended_count"] == 3857


def test_ensure_dataset_yaml(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    yaml_path = tmp_path / "dataset.yaml"

    res = ensure_dataset_yaml(images_dir, yaml_path)
    assert res.is_file()
    content = res.read_text(encoding="utf-8")
    assert "khoi" in content
    assert "lua" in content
    assert "test: \"images\"" in content


def test_compare_benchmarks_metrics_improved():
    candidate_metrics = {
        "overall": {"precision": 0.88, "recall": 0.85, "map50": 0.90, "map50_95": 0.68},
        "classes": {
            "khoi": {"precision": 0.87, "recall": 0.84, "map50": 0.89, "map50_95": 0.66},
            "lua": {"precision": 0.89, "recall": 0.86, "map50": 0.91, "map50_95": 0.70},
        },
        "speed": {"inference_ms": 7.5},
    }
    base_metrics = {
        "overall": {"precision": 0.84, "recall": 0.81, "map50": 0.86, "map50_95": 0.63},
        "classes": {
            "khoi": {"precision": 0.83, "recall": 0.80, "map50": 0.85, "map50_95": 0.61},
            "lua": {"precision": 0.85, "recall": 0.82, "map50": 0.87, "map50_95": 0.65},
        },
        "speed": {"inference_ms": 8.0},
    }

    res = compare_benchmarks_metrics(candidate_metrics, base_metrics)
    assert res["verdict"] == "improved"
    assert res["overall"]["map50"]["delta"] == 0.04
    assert res["overall"]["map50"]["pct"] > 0
    assert res["classes"]["khoi"]["delta_map50"] == 0.04
    assert res["classes"]["lua"]["delta_map50"] == 0.04


def test_compare_benchmarks_metrics_regressed():
    candidate_metrics = {
        "overall": {"precision": 0.70, "recall": 0.70, "map50": 0.72, "map50_95": 0.50},
        "classes": {"khoi": {"map50": 0.70}, "lua": {"map50": 0.74}},
        "speed": {"inference_ms": 10.0},
    }
    base_metrics = {
        "overall": {"precision": 0.85, "recall": 0.80, "map50": 0.88, "map50_95": 0.65},
        "classes": {"khoi": {"map50": 0.87}, "lua": {"map50": 0.89}},
        "speed": {"inference_ms": 8.0},
    }

    res = compare_benchmarks_metrics(candidate_metrics, base_metrics)
    assert res["verdict"] == "regressed"
    assert res["overall"]["map50"]["delta"] < 0


def test_benchmark_run_db_lifecycle(tmp_path: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.db import Base
    from backend.app.models import BenchmarkRun, ModelVersion

    engine = create_engine(f"sqlite:///{(tmp_path / 'bench.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as db:
        m = ModelVersion(name="test_model", artifact_path="m.pt", sha256="0" * 64, class_map={"0": "khoi", "1": "lua"}, status="active")
        db.add(m)
        db.commit()

        b = BenchmarkRun(model_version_id=m.id, status="completed", image_count=100)
        db.add(b)
        db.commit()

        assert db.get(BenchmarkRun, b.id) is not None
        db.delete(b)
        db.commit()
        assert db.get(BenchmarkRun, b.id) is None


