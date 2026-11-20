import json
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db import Base
from backend.app.models import Feedback, ModelVersion, RetrainRun
from backend.app.retrain_service import (
    build_feedback_bundle,
    generate_retrain_notebook_content,
    get_retrain_statistics,
    retrain_run_data,
)


def test_retrain_stats_and_notebook_generation(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test_retrain.sqlite').as_posix()}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    with session_factory() as db:
        # Create active model
        active_model = ModelVersion(
            name="final_sfd_yolo11n_test",
            artifact_path="models/test/best.pt",
            sha256="0" * 64,
            class_map={"0": "khoi", "1": "lua"},
            status="active",
        )
        db.add(active_model)
        db.commit()

        # Check stats with 0 feedbacks
        stats = get_retrain_statistics(db)
        assert stats["accepted_total"] == 0
        assert stats["hard_negatives"] == 0
        assert stats["hard_positives"] == 0
        assert stats["active_model_name"] == "final_sfd_yolo11n_test"

        # Bundle empty raises ValueError
        import pytest
        with pytest.raises(ValueError, match="Chưa có phản hồi nào"):
            build_feedback_bundle(db)

        # Generate notebook
        nb_json_str = generate_retrain_notebook_content(
            feedback_b64="dummy_b64",
            epochs=15,
            run_tag="test_run",
        )
        nb = json.loads(nb_json_str)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) >= 7

        # Check RetrainRun serializer
        run = RetrainRun(
            id="test-run-123",
            kernel_slug="user/test-kernel",
            status="running",
            kaggle_status="running",
            feedback_count=0,
            base_model_name=active_model.name,
        )
        db.add(run)
        db.commit()

        run_dict = retrain_run_data(run)
        assert run_dict["id"] == "test-run-123"
        assert run_dict["status"] == "running"
        assert run_dict["kaggle_url"] == "https://www.kaggle.com/code/user/test-kernel"
