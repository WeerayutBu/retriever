import os

from fastapi import FastAPI, Request
from core.retriever import Retriever

app = FastAPI()


@app.on_event("startup")
def startup():
    app.state.retriever = Retriever(
        csv_path=os.environ.get("CSV_PATH", "data/data.csv"),
        model_name="./.cache/embeddings/Qwen3-Embedding-0.6B", retrieve_k=64,
        reranker_name="./.cache/embeddings/bge-reranker-v2-m3",  top_k=8, # optional
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
