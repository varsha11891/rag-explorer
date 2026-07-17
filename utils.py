"""Shared data models and utilities for RAG Explorer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    page: int


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    start: int
    end: int
    strategy: str = "Fixed size"
    parent_id: str | None = None
    level: str = "chunk"
    global_id: int = 0

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def format_duration(seconds: float) -> str:
    return f"{seconds * 1_000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"
