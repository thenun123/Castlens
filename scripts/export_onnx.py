"""Export the trained PyTorch checkpoint to ONNX (runtime image stays torch-free) and verify numerical parity.

    python scripts/export_onnx.py                       # model/effnet_b0_finetune.pt -> model/effnet_b0_finetune.onnx
    python scripts/export_onnx.py --checkpoint X.pt --out Y.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]


def build_model() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)  # same head as the training notebook
    return m


def export(checkpoint: Path, out: Path, img_size: int = 224) -> float:
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model()
    model.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
    model.eval()
    x = torch.randn(2, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, x, str(out), dynamo=False, **kwargs)  # legacy exporter: no extra deps
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(model, x, str(out), **kwargs)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = torch.softmax(model(x), 1).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    got = np.exp(got - got.max(1, keepdims=True)); got /= got.sum(1, keepdims=True)
    diff = float(np.abs(ref - got).max())
    if diff > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff:.2e} - refusing to ship this export.")
    return diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "model" / "effnet_b0_finetune.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "model" / "effnet_b0_finetune.onnx")
    a = ap.parse_args()
    print(f"exported {a.out} (max probability diff vs PyTorch: {export(a.checkpoint, a.out):.2e})")
