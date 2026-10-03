from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .inference import Classifier, InvalidImage

log = logging.getLogger("castlens")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "model"))
MODEL_KEY = os.getenv("MODEL_KEY", "effnet_b0_finetune")
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024)
RATE_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", ROOT / "static"))
CORS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]


class RateLimiter:
    """Per-IP sliding window, in memory. Per instance and reset on restart: fine for a demo, not a security boundary."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window, self.hits = limit, window, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


limiter = RateLimiter(RATE_PER_MIN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    onnx = MODEL_DIR / f"{MODEL_KEY}.onnx"
    cfg = MODEL_DIR / "inference_config.json"
    for f in (onnx, cfg):  # fail fast so a bad deploy never goes healthy
        if not f.exists():
            raise RuntimeError(f"Missing {f}. See model/README.md for how to add the trained model.")
    app.state.clf = Classifier(onnx, cfg, MODEL_KEY)
    log.info("model loaded: %s version=%s threshold=%.2f", MODEL_KEY, app.state.clf.version, app.state.clf.threshold)
    yield


app = FastAPI(title="CastLens", version="1.0.0", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
if CORS:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=CORS, allow_methods=["GET", "POST"], allow_headers=["*"])

CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.url.path == "/api/predict":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BYTES + 4096:
            return JSONResponse({"detail": f"File is larger than {MAX_BYTES // 1024 // 1024} MB."}, status_code=413)
    resp = await call_next(request)
    resp.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                         "Content-Security-Policy": CSP, "X-Frame-Options": "DENY"})
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model")
def model_info(request: Request):
    c: Classifier = request.app.state.clf
    return {"name": c.model_key, "version": c.version, "threshold": c.threshold, "input_size": c.size,
            "classes": {"0": "ok_front", "1": "def_front"}}


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    ip = request.client.host if request.client else "unknown"
    if not limiter.allow(ip):
        raise HTTPException(429, "Too many requests. Wait a minute and try again.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // 1024 // 1024} MB.")
    clf: Classifier = request.app.state.clf
    try:
        result = await run_in_threadpool(clf.predict_bytes, data)
    except InvalidImage as e:
        raise HTTPException(422, str(e))
    log.info("predict verdict=%s p_defect=%.3f ms=%.0f bytes=%d", result["verdict"],
             result["probabilities"]["defect"], result["inference_ms"], len(data))  # never log image content
    return result


if STATIC_DIR.is_dir():  # built frontend (production image). Mounted last so /api/* wins.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
