.PHONY: export test dev-api dev-web build run
export:      ## .pt -> .onnx (needs backend/requirements-dev.txt)
	python scripts/export_onnx.py
test:
	cd backend && PYTHONPATH=. pytest -q
dev-api:
	cd backend && MODEL_DIR=../model uvicorn app.main:app --reload --port 8000
dev-web:
	cd frontend && npm run dev
build:
	docker build -t castlens .
run:
	docker run --rm -p 10000:10000 castlens
