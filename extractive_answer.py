"""Local, no-LLM extractive answers from smart-reranked evidence."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.embedder import DEFAULT_MODEL, embed_query, embed_texts
from src.smart_retriever import decompose_query, detect_query_intent, inspect_subquestion_evidence

INSUFFICIENT_EXTRACTIVE_MESSAGE = (
    "I could not find a sufficiently supported answer in the retrieved text."
)
MIN_SENTENCE_SIMILARITY = 0.18


def split_chunks_into_sentences(retrieved_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split canonical retrieval chunks while retaining exact provenance."""
    sentences: list[dict[str, Any]] = []
    for result in retrieved_results:
        chunk = result.get("chunk", {}) if isinstance(result, dict) else {}
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        for part in parts:
            sentence = " ".join(part.split()).strip()
            if len(sentence) < 20:
                continue
            sentences.append(
                {
                    "text": sentence,
                    "source": str(chunk.get("book", "Unknown source")),
                    "chunk_id": int(chunk.get("chunk_id", 0) or 0),
                    "page": chunk.get("page"),
                }
            )
    return sentences


def _intent_preference(query: str, sentence: str) -> tuple[float, list[str]]:
    intent = detect_query_intent(query)
    lowered = sentence.casefold()
    subject = intent["extracted_entity_or_subject"].casefold()
    boost = 0.0
    reasons: list[str] = []
    if subject and subject in lowered:
        boost += 0.08
        reasons.append("exact subject match")
    markers = {
        "who_is": (" is ", " was ", "professor", "teacher", "head of", "works as"),
        "who": ("gave", "sent", "provided", "arranged", "responsible"),
        "what_is": (" is ", "means", "refers to", "used for", "purpose"),
        "where_is": ("located", "situated", "near", "inside", "outside"),
        "where": ("located", "situated", "near", "inside", "outside"),
        "when_did": ("before", "after", "during", "year", "age"),
        "when": ("before", "after", "during", "year", "age"),
        "why_did": ("because", "reason", "so that", "in order to", "due to"),
        "why": ("because", "reason", "so that", "in order to", "due to"),
        "how_did": (" by ", "through", "method", "steps"),
        "how": (" by ", "through", "method", "steps"),
    }
    matches = [marker.strip() for marker in markers.get(intent["intent_type"], ()) if marker in lowered]
    if matches:
        boost += min(0.12, 0.04 * len(matches))
        reasons.append("intent evidence: " + ", ".join(matches[:3]))
    return boost, reasons


def rank_sentences(
    query: str,
    sentences: Sequence[dict[str, Any]],
    embedding_model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Rank sentence records by local cosine similarity plus deterministic intent preference."""
    if not query.strip() or not sentences:
        return []
    vectors = embed_texts([sentence["text"] for sentence in sentences], embedding_model)
    query_vector = embed_query(query, embedding_model)
    similarities = np.asarray(vectors @ query_vector, dtype=float)
    ranked: list[dict[str, Any]] = []
    for sentence, similarity in zip(sentences, similarities, strict=True):
        boost, reasons = _intent_preference(query, sentence["text"])
        ranked.append(
            {
                **sentence,
                "similarity": float(similarity),
                "ranking_score": float(similarity + boost),
                "boost_reasons": reasons,
            }
        )
    return sorted(ranked, key=lambda item: item["ranking_score"], reverse=True)


def _select_distinct(ranked: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        normalized = re.sub(r"\W+", " ", item["text"].casefold()).strip()
        if item["similarity"] < MIN_SENTENCE_SIMILARITY or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def generate_extractive_answer(
    query: str,
    retrieved_results: Sequence[dict[str, Any]],
    max_sentences: int = 3,
    subquestions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select verbatim supporting sentences locally; never synthesize or paraphrase."""
    started = time.perf_counter()
    sentences = split_chunks_into_sentences(retrieved_results)
    questions = list(subquestions or decompose_query(query) or [query])
    per_question_limit = max(1, max_sentences if len(questions) == 1 else 1)
    parts: list[dict[str, Any]] = []
    all_used: list[dict[str, Any]] = []
    for question in questions:
        inspection = inspect_subquestion_evidence(question, retrieved_results)
        supporting_keys = {
            (record["source"], int(record["chunk_id"]))
            for record in inspection["supporting_chunks"]
        }
        eligible_sentences = [
            sentence for sentence in sentences
            if (sentence["source"], sentence["chunk_id"]) in supporting_keys
        ]
        used = (
            _select_distinct(
                rank_sentences(question, eligible_sentences, DEFAULT_MODEL), per_question_limit
            )
            if inspection["coverage_status"] != "Unsupported"
            else []
        )
        if used:
            answer = " ".join(
                f"{item['text']} [Source: {item['source']} | Chunk: {item['chunk_id']}]"
                for item in used
            )
            coverage = inspection["coverage_status"]
        else:
            answer = INSUFFICIENT_EXTRACTIVE_MESSAGE
            coverage = "Unsupported"
        parts.append(
            {
                "sub_question": question,
                "coverage_status": coverage,
                "answer": answer,
                "sentences_used": used,
            }
        )
        all_used.extend(used)
    combined = (
        parts[0]["answer"]
        if len(parts) == 1
        else "\n\n".join(
            f"**{index}. {part['sub_question']}**\n\n{part['answer']}"
            for index, part in enumerate(parts, start=1)
        )
    )
    unique_sources = list(
        {
            (item["source"], item["chunk_id"]): {
                "book": item["source"],
                "chunk_id": item["chunk_id"],
                "page": item.get("page"),
            }
            for item in all_used
        }.values()
    )
    latency = (time.perf_counter() - started) * 1_000
    return {
        "answer": combined,
        "mode": "Extractive",
        "model": f"{DEFAULT_MODEL} (local extractive)",
        "sentences_used": all_used,
        "sources_used": unique_sources,
        "latency_ms": latency,
        "extractive_latency_ms": latency,
        "prompt_tokens": None,
        "output_tokens": None,
        "subquestion_answers": parts,
    }
