import io

import numpy as np
from PIL import Image

from app.inference import Classifier, decide


def png_bytes(size=(300, 300), color=128):
    buf = io.BytesIO()
    Image.new("RGB", size, (color,) * 3).save(buf, "PNG")
    return buf.getvalue()


def test_decide_borderline():
    assert decide(0.30, 0.21) == {"verdict": "defect", "highest": "ok", "borderline": True}
    assert decide(0.10, 0.21) == {"verdict": "ok", "highest": "ok", "borderline": False}
    assert decide(0.90, 0.21) == {"verdict": "defect", "highest": "defect", "borderline": False}


def test_health_and_model(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    m = client.get("/api/model").json()
    assert m["threshold"] == 0.21 and m["input_size"] == 224


def test_predict_contract(client):
    r = client.post("/api/predict", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    p = j["probabilities"]
    assert abs(p["ok"] + p["defect"] - 1) < 1e-6
    assert j["highest"] == ("defect" if p["defect"] >= 0.5 else "ok")
    assert j["verdict"] == ("defect" if p["defect"] >= 0.21 else "ok")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 422


def test_rejects_oversize(client):
    r = client.post("/api/predict", files={"file": ("big.png", b"0" * (2 * 1024 * 1024), "image/png")})
    assert r.status_code == 413


def test_warns_on_small_non_square(client):
    j = client.post("/api/predict", files={"file": ("s.png", png_bytes((100, 60)), "image/png")}).json()
    assert len(j["warnings"]) == 2


def test_preprocess_matches_torchvision(client):
    """Guards train/serve skew: our PIL+numpy pipeline must equal the notebook's torchvision transforms."""
    from torchvision import transforms as T
    clf: Classifier = client.app.state.clf
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (317, 289, 3), dtype=np.uint8))
    ref = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img).numpy()[None]
    assert np.abs(ref - clf.preprocess(img)).max() < 1e-4
