"""ONNX inference that mirrors the notebook's eval pipeline exactly:
RGB -> bilinear resize to IMG_SIZE x IMG_SIZE -> /255 -> ImageNet normalisation (+ optional brightness equalisation)."""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 25_000_000  # decompression-bomb guard
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class InvalidImage(ValueError):
    pass


def decide(p_defect: float, threshold: float) -> dict:
    """verdict uses the cost-tuned threshold; `highest` is plain argmax. They can differ (borderline)."""
    verdict = "defect" if p_defect >= threshold else "ok"
    highest = "defect" if p_defect >= 0.5 else "ok"
    return {"verdict": verdict, "highest": highest, "borderline": verdict != highest}


class Classifier:
    def __init__(self, onnx_path: Path, config_path: Path, model_key: str):
        cfg = json.loads(Path(config_path).read_text())
        if model_key not in cfg["models"]:
            raise KeyError(f"{model_key!r} not in {config_path} (has {list(cfg['models'])})")
        self.model_key = model_key
        self.threshold = float(cfg["models"][model_key]["threshold"])
        self.size = int(cfg["img_size"])
        self.mean = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
        self.equalize_target = float(cfg["train_mean_luminance"]) if cfg.get("normalize_brightness") else None
        self.version = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()[:12]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._run(np.zeros((1, 3, self.size, self.size), np.float32))  # warm-up

    def _run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        a = np.asarray(img, np.float32)
        if self.equalize_target is not None:  # same as training-time EqualizeBrightness
            a = np.clip(a * (self.equalize_target / max(float(a.mean()), 1e-6)), 0, 255).astype(np.uint8).astype(np.float32)
        a = a.transpose(2, 0, 1) / 255.0
        return ((a - self.mean) / self.std)[None].astype(np.float32)

    @staticmethod
    def decode(data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as e:
            raise InvalidImage("This file is not a readable image. Upload a JPEG, PNG, WebP or BMP photo.") from e
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImage(f"{fmt} images are not supported. Upload a JPEG, PNG, WebP or BMP photo.")
        return img

    def predict_bytes(self, data: bytes) -> dict:
        img = self.decode(data)
        warnings = []
        w, h = img.size
        if min(w, h) < 150:
            warnings.append(f"Image is small ({w}x{h}px). The model was trained on 300x300px photos, so accuracy may drop.")
        if max(w, h) / min(w, h) > 1.3:
            warnings.append("Image is not square. It will be stretched to a square, which can distort the part.")
        t0 = time.perf_counter()
        logits = self._run(self.preprocess(img))[0].astype(np.float64)
        e = np.exp(logits - logits.max())
        p = e / e.sum()  # index 0 = ok_front, 1 = def_front (same as training LABEL_MAP)
        ms = (time.perf_counter() - t0) * 1000
        p_def = float(p[1])
        return {
            **decide(p_def, self.threshold),
            "probabilities": {"ok": float(p[0]), "defect": p_def},
            "threshold": self.threshold,
            "warnings": warnings,
            "model": {"name": self.model_key, "version": self.version},
            "inference_ms": round(ms, 1),
        }
