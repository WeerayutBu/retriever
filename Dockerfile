FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: python, pip, build tools, git, curl (often needed), and common libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    build-essential \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make `python` point to python3 (handy)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip tooling
RUN python -m pip install --upgrade pip
RUN pip install "pyarrow==20.0.0" -U
RUN PIP_ONLY_BINARY=:all: python -m pip install -U "tiktoken==0.7.0" -i https://pypi.org/simple -v
RUN python -m pip install -U "llama-index-postprocessor-flag-embedding-reranker"
RUN pip install -U llama-index

# LlamaIndex
RUN git clone https://github.com/run-llama/llama_index.git
RUN pip install -e llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-lancedb

RUN python -m pip install --only-binary=:all: sentencepiece
RUN pip install -U llama-index llama-index-embeddings-huggingface llama-index-vector-stores-lancedb sentence-transformers

RUN python -m pip install --only-binary=:all: --force-reinstall numpy==1.26.4 pandas==2.2.3 
RUN python -m pip install --only-binary=:all: "datasets==3.3.2"

RUN pip install "pyzmq<25.0" 
RUN pip install "ipykernel==6.25.2" "tornado<6.3" "traitlets<5.11"
RUN pip install "git+https://github.com/FlagOpen/FlagEmbedding.git"

# Install Python deps (adjust if you use requirements.txt)
RUN pip install uvicorn fastapi
RUN python -m pip install -U "transformers>=4.51.0,<5" safetensors huggingface_hub

# Copy project files
COPY . /app