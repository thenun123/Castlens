# CastLens

Upload a photo of a casting and get **defect / OK**, the probability of both, and which one is higher.
FastAPI + ONNX Runtime backend, React + Framer Motion frontend, one Docker image, deployable on Render.

```
castlens/
├── backend/            FastAPI app (app/), tests/, requirements*.txt
├── frontend/           Vite + React + TypeScript + Framer Motion
├── model/              trained model (.pt source, .onnx runtime, inference_config.json)  <- you add these
├── notebooks/          01_eda.ipynb, 02_training.ipynb (the run that produced the model)
├── scripts/            export_onnx.py  (.pt -> .onnx + parity check)
├── docs/               design-system.md
├── Dockerfile  render.yaml  Makefile  .github/workflows/ci.yml
```

## Quick start
1. Copy `effnet_b0_finetune.pt` and `inference_config.json` from Drive (`casting_v2/baseline/`) into `model/`.
2. `pip install -r backend/requirements-dev.txt && make export` (creates `model/effnet_b0_finetune.onnx`).
3. Run: `make dev-api` and, in another terminal, `cd frontend && npm install && make dev-web` (http://localhost:5173).
4. Tests: `make test`. Docker: `make build && make run` (http://localhost:10000).

## Deploy on Render
Push the repo (with the three model files committed) to GitHub, then Render -> **New -> Blueprint** -> pick the repo. `render.yaml` sets the
Docker runtime, health check (`/api/health`) and env vars. The free plan (512 MB) is enough because the runtime uses ONNX Runtime, not PyTorch,
but it sleeps after idle time; use `starter` for always-on.

## API
| method | path | |
|---|---|---|
| POST | `/api/predict` | multipart field `file` -> verdict, `probabilities {ok, defect}`, `highest`, `borderline`, `threshold`, `warnings`, `model` |
| GET | `/api/model` | model name, build hash, threshold |
| GET | `/api/health` | liveness (the app does not start without a model) |

**`verdict` vs `highest`.** `highest` is the larger probability. `verdict` applies the cost-tuned threshold (0.21, chosen on validation with a
missed defect costing 5x a false alarm). A part with 30% defect probability is therefore flagged `defect` although OK is higher; the API
returns `borderline: true` and the UI explains it. Do not "simplify" this to argmax without re-deciding the cost trade-off.

Env vars: `MODEL_KEY`, `MODEL_DIR`, `MAX_UPLOAD_MB` (8), `RATE_LIMIT_PER_MINUTE` (30), `CORS_ORIGINS`, `LOG_LEVEL`.

## Known limits (read before relying on it)
* Trained on one part type and camera setup; there is **no out-of-distribution check**, so a photo of anything else still gets a confident score.
  The API only warns on small or non-square images.
* Test accuracy was near-saturated (1 error in 651), and the false-alarm rate on genuinely new parts is the least certain number.
* Rate limiting is in-memory per instance; there is no authentication.
* Uploads are processed in memory and never written to disk or logged.
