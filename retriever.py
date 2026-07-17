"""Retrieval algorithms and observable brute-force search."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from src.utils import Chunk, SearchResult

SearchCallback = Callable[[int, int, Chunk, float, bool], None]


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(matrix @ query, denominator, out=np.zeros(len(matrix)), where=denominator != 0)


def brute_force_search(
    query_vector: np.ndarray,
    embeddings: np.ndarray,
    chunks: Sequence[Chunk],
    top_k: int = 5,
    callback: SearchCallback | None = None,
) -> list[SearchResult]:
    if len(embeddings) != len(chunks):
        raise ValueError("Each chunk must have exactly one embedding")
    if not chunks:
        return []

    scores = cosine_similarity(query_vector, embeddings)
    best = float("-inf")
    for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True), start=1):
        improved = float(score) > best
        best = max(best, float(score))
        if callback:
            callback(index, len(chunks), chunk, float(score), improved)

    indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    return [
        SearchResult(chunk=chunks[int(i)], score=float(scores[i]), rank=rank)
        for rank, i in enumerate(indices, start=1)
    ]

