"""Reusable text chunking strategies."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from src.utils import Chunk, Document, stable_id


def _chunk(document: Document, text: str, start: int, end: int, strategy: str,
           parent_id: str | None = None, level: str = "chunk") -> Chunk:
    return Chunk(
        id=stable_id(document.source, document.page, start, end, text, strategy, parent_id),
        text=text.strip(), source=document.source, page=document.page, start=start, end=end,
        strategy=strategy, parent_id=parent_id, level=level,
    )


def fixed_size_chunks(
    documents: Iterable[Document], chunk_size: int = 500, overlap: int = 0,
    strategy: str = "Fixed size",
) -> list[Chunk]:
    """Split documents into character-sized chunks while retaining provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = " ".join(document.text.split())
        for start in range(0, len(text), step):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            chunks.append(_chunk(document, chunk_text, start, end, strategy))
            if end == len(text):
                break
    return chunks


def recursive_chunks(documents: Iterable[Document], chunk_size: int) -> list[Chunk]:
    """Recursively split on paragraphs, sentences, words, then characters."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split(text: str, separator_index: int = 0) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        separator = separators[separator_index]
        if not separator:
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        parts, groups, current = text.split(separator), [], ""
        for part in parts:
            candidate = (current + separator + part).strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    groups.append(current)
                groups.extend(split(part, separator_index + 1)) if len(part) > chunk_size else None
                current = "" if len(part) > chunk_size else part
        if current:
            groups.append(current)
        return groups

    output = []
    for document in documents:
        cursor = 0
        for text in split(document.text.strip()):
            start = document.text.find(text, cursor)
            start = cursor if start < 0 else start
            end, cursor = start + len(text), start + len(text)
            output.append(_chunk(document, text, start, end, "Recursive"))
    return output


def semantic_chunks(documents: Iterable[Document], chunk_size: int) -> list[Chunk]:
    """Group adjacent sentences until a semantic shift or size boundary."""
    import numpy as np

    from src.embedder import embed_texts

    output = []
    for document in documents:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", document.text) if s.strip()]
        if not sentences:
            continue
        vectors = embed_texts(sentences)
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1) if len(vectors) > 1 else np.array([])
        threshold = float(np.percentile(similarities, 30)) if len(similarities) else -1.0
        groups, current = [], sentences[0]
        for index, sentence in enumerate(sentences[1:]):
            shift = similarities[index] < threshold
            if shift or len(current) + len(sentence) + 1 > chunk_size:
                groups.append(current)
                current = sentence
            else:
                current += " " + sentence
        groups.append(current)
        cursor = 0
        for text in groups:
            start = document.text.find(text, cursor)
            start = cursor if start < 0 else start
            end, cursor = start + len(text), start + len(text)
            output.append(_chunk(document, text, start, end, "Semantic"))
    return output


def parent_child_chunks(documents: Iterable[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    """Create large parent contexts and smaller retrievable child chunks."""
    output = []
    parent_size = max(chunk_size * 3, chunk_size + 1)
    for document in documents:
        parents = fixed_size_chunks([document], parent_size, 0, "Parent-child")
        for parent in parents:
            parent_chunk = Chunk(**{**parent.metadata(), "level": "parent"})
            output.append(parent_chunk)
            parent_doc = Document(parent.text, document.source, document.page)
            children = fixed_size_chunks([parent_doc], chunk_size, overlap, "Parent-child")
            for child in children:
                output.append(Chunk(**{
                    **child.metadata(), "id": stable_id(parent.id, child.start, child.text),
                    "start": parent.start + child.start, "end": parent.start + child.end,
                    "parent_id": parent.id, "level": "child",
                }))
    return output


def proposition_chunks(documents: Iterable[Document]) -> list[Chunk]:
    """Split sentences into compact, independently retrievable propositions."""
    output = []
    splitter = re.compile(r"(?<=[.!?])\s+|\s*[;]\s*|,\s+(?=(?:and|but|while|whereas)\b)", re.I)
    for document in documents:
        cursor = 0
        for proposition in splitter.split(document.text):
            text = proposition.strip(" ,")
            if len(text.split()) < 3:
                continue
            start = document.text.find(text, cursor)
            start = cursor if start < 0 else start
            end, cursor = start + len(text), start + len(text)
            output.append(_chunk(document, text, start, end, "Fact & proposition", level="proposition"))
    return output


def chunk_documents(
    documents: Iterable[Document], strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    documents = list(documents)
    if strategy == "Fixed size":
        chunks = fixed_size_chunks(documents, chunk_size, 0, strategy)
    elif strategy == "Fixed size with overlap":
        chunks = fixed_size_chunks(documents, chunk_size, overlap, strategy)
    elif strategy == "Recursive":
        chunks = recursive_chunks(documents, chunk_size)
    elif strategy == "Semantic":
        chunks = semantic_chunks(documents, chunk_size)
    elif strategy == "Parent-child":
        chunks = parent_child_chunks(documents, chunk_size, overlap)
    elif strategy == "Fact & proposition":
        chunks = proposition_chunks(documents)
    else:
        raise ValueError(f"Unsupported chunk strategy: {strategy}")
    return [replace(chunk, global_id=index) for index, chunk in enumerate(chunks, start=1)]
