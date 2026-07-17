"""Vector-store abstraction with brute-force and ChromaDB implementations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.retriever import SearchCallback, brute_force_search
from src.utils import Chunk, SearchResult


class BruteForceStore:
    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray):
        self.chunks = list(chunks)
        self.embeddings = np.asarray(embeddings)

    def search(self, query_vector: np.ndarray, top_k: int, callback: SearchCallback | None = None):
        return brute_force_search(query_vector, self.embeddings, self.chunks, top_k, callback)


class ChromaStore:
    """Ephemeral Chroma collection; embedding remains owned by the application."""

    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray):
        import chromadb

        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self.client = chromadb.EphemeralClient()
        try:
            self.client.delete_collection("rag_explorer")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="rag_explorer", metadata={"hnsw:space": "cosine"}
        )
        if chunks:
            self.collection.upsert(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[{
                    "chunk_id": chunk.global_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "start": chunk.start,
                    "end": chunk.end,
                } for chunk in chunks],
                embeddings=np.asarray(embeddings).tolist(),
            )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        count = self.collection.count()
        if not count:
            return []
        response = self.collection.query(
            query_embeddings=[np.asarray(query_vector).tolist()],
            n_results=min(top_k, count),
            include=["distances"],
        )
        ids = response["ids"][0]
        distances = response["distances"][0]
        return [
            SearchResult(self.chunks_by_id[chunk_id], 1.0 - float(distance), rank)
            for rank, (chunk_id, distance) in enumerate(zip(ids, distances, strict=True), start=1)
        ]
