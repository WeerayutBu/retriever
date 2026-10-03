.DEFAULT_GOAL := help

PYTHON ?= python3
IMAGE ?= retriever
HOST ?= 0.0.0.0
PORT ?= 18763
GPU_ARGS ?= --gpus all
CSV_PATH ?= data/data.csv

.PHONY: help build up demo start dev models check health

help:
	@printf '%s\n' 'make up      Build and start the API with the full dataset' 'make demo    Build and start the API with the demo dataset' 'make start   Alias for make up' 'make build   Build the Docker image' 'make dev     Start the API with local Python dependencies' 'make models  Download embedding and reranker models' 'make health  Check the running API'

build:
	docker build -t $(IMAGE) .

check:
	@test -f "$(CSV_PATH)" || { echo "Missing $(CSV_PATH)"; exit 1; }
	@test -f .cache/embeddings/Qwen3-Embedding-0.6B/config.json || { echo "Missing embedding model; run make models"; exit 1; }
	@test -f .cache/embeddings/bge-reranker-v2-m3/config.json || { echo "Missing reranker model; run make models"; exit 1; }

start: up

demo: CSV_PATH = data/demo.csv
demo: up

up: check build
	docker run --rm $(GPU_ARGS) -p $(PORT):$(PORT) --shm-size=4g -e CSV_PATH="$(CSV_PATH)" -v "$(CURDIR)/.cache:/app/.cache" $(IMAGE) python -m uvicorn api.main:app --host $(HOST) --port $(PORT)

dev: check
	CSV_PATH="$(CSV_PATH)" $(PYTHON) -m uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

models:
	hf download Qwen/Qwen3-Embedding-0.6B --local-dir .cache/embeddings/Qwen3-Embedding-0.6B
	hf download BAAI/bge-reranker-v2-m3 --local-dir .cache/embeddings/bge-reranker-v2-m3

health:
	curl --fail --show-error http://127.0.0.1:$(PORT)/
