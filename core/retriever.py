from pathlib import Path
from typing import Optional, List

import pandas as pd
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.lancedb import LanceDBVectorStore


class Retriever:
    def __init__(
        self,
        csv_path: str,
        model_name: str,
        reranker_name: str | None = None,
        cache_dir: str = "./.cache",
        device: str = "cuda",
        top_k: int = 8,
        retrieve_k: int = 64,
        rebuild_index: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.retrieve_k = retrieve_k
        self.rebuild_index = rebuild_index
        self._index: Optional[VectorStoreIndex] = None

        csv_name = csv_path.rsplit('/', 1)[-1].removesuffix('.csv')
        model = model_name.rsplit('/', 1)[-1]
        self.table = f"{csv_name}_{model}".lower()

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            normalize=True,
        )

        self.reranker = (
            FlagEmbeddingReranker(model=reranker_name, top_n=top_k, use_fp16=True)
            if reranker_name
            else None
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_nodes(self) -> List[TextNode]:
        df = pd.read_csv(self.csv_path)
        return [
            TextNode(text=row.text, metadata={"id": getattr(row, "id", None)})
            for row in df.itertuples(index=False)
            if pd.notna(row.text)
        ]

    def load(self) -> VectorStoreIndex:
        if self._index:
            return self._index

        uri = str(self.cache_dir / "lancedb")

        if self.rebuild_index:
            import shutil
            shutil.rmtree(uri, ignore_errors=True)

        vector_store = LanceDBVectorStore(uri=uri, table_name=self.table)
        storage = StorageContext.from_defaults(vector_store=vector_store)

        if not self.rebuild_index:
            try:
                self._index = VectorStoreIndex.from_vector_store(vector_store)
                return self._index
            except Exception:
                pass

        self._index = VectorStoreIndex(
            self._load_nodes(),
            storage_context=storage,
            show_progress=True,
        )
        return self._index

    def search(self, query: str) -> List[NodeWithScore]:
        nodes = self.load().as_retriever(
            similarity_top_k=self.retrieve_k
        ).retrieve(query)

        if self.reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=query)

        return nodes
