"""Sentence-transformer embedding adapter."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_model(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Sequence[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = load_model(model_name).encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    return embed_texts([query], model_name)[0]

