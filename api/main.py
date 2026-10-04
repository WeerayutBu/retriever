import os

from fastapi import FastAPI, Request
from core.retriever import Retriever

app = FastAPI()


@app.on_event("startup")
def startup():
    retrieve_k = int(os.environ.get("RETRIEVE_K", "64"))
    top_k = int(os.environ.get("TOP_K", "16"))
    if not 1 <= top_k <= retrieve_k:
        raise ValueError("Require 1 <= TOP_K <= RETRIEVE_K")
    app.state.retriever = Retriever(
        csv_path=os.environ.get("CSV_PATH", "data/data.csv"),
        model_name="./.cache/embeddings/Qwen3-Embedding-0.6B",
        retrieve_k=retrieve_k,
        reranker_name="./.cache/embeddings/bge-reranker-v2-m3",
        top_k=top_k,
        rebuild_index=False,
    )
    app.state.retriever.load()
    print("Startup: Done")


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/retrieve")
def retrieve(q: str, request: Request):
    nodes = request.app.state.retriever.search(q)
    return {
        "results": [
            {"id": n.metadata["id"], "text": n.text, "score": n.score}
            for n in nodes
        ]
    }
