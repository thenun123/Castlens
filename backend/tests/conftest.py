import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Random-weight model: tests the pipeline and API contract, not accuracy."""
    from export_onnx import build_model, export
    d = tmp_path_factory.mktemp("model")
    torch.manual_seed(0)
    torch.save({"state_dict": build_model().state_dict()}, d / "effnet_b0_finetune.pt")
    export(d / "effnet_b0_finetune.pt", d / "effnet_b0_finetune.onnx")
    (d / "inference_config.json").write_text(json.dumps({
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "normalize_brightness": False,
        "models": {"effnet_b0_finetune": {"checkpoint": "effnet_b0_finetune.pt", "threshold": 0.21}}}))
    return d


@pytest.fixture(scope="session")
def client(model_dir):
    import os
    os.environ.update(MODEL_DIR=str(model_dir), MAX_UPLOAD_MB="1", RATE_LIMIT_PER_MINUTE="1000", STATIC_DIR=str(model_dir / "none"))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c
