"""Session-scoped RAG experiment recording, comparison, export, and diagnosis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


def initialize_experiment_history() -> list[dict[str, Any]]:
    return st.session_state.setdefault("experiment_history", [])


def build_knowledge_base_version(files, settings: dict, embedding_model: str) -> str:
    payload = {
        "files": sorted(
            (str(file.name), int(getattr(file, "size", 0) or 0)) for file in files
        ),
        "chunk_strategy": settings["strategy"],
        "chunk_size": settings["chunk_size"],
        "overlap": settings["overlap"],
        "embedding_model": embedding_model,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def create_experiment_record(
    query: str,
    smart_run: dict[str, Any],
    settings: dict[str, Any],
    embedding_model: str,
    embedding_dimension: int,
    knowledge_base_version: str,
    retrieved_results: list[dict[str, Any]],
    vector_count: int,
    embedding_previews: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    top = retrieved_results[0] if retrieved_results else {}
    top_chunk = top.get("chunk", {})
    embedding_ms = float(smart_run.get("embedding_latency", 0)) * 1_000
    retrieval_ms = float(smart_run.get("retrieval_latency", 0)) * 1_000
    stored_results = []
    for result in retrieved_results:
        copy = {**result, "chunk": dict(result.get("chunk", {}))}
        chunk_id = int(copy["chunk"].get("chunk_id", 0) or 0)
        copy["embedding_preview"] = list((embedding_previews or {}).get(chunk_id, []))
        stored_results.append(copy)
    return {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": query,
        "original_query": smart_run.get("original_query", query),
        "transformed_query": smart_run.get("transformed_retrieval_query", query),
        "detected_intent": smart_run.get("intent_type", "unknown"),
        "chunk_strategy": settings["strategy"],
        "chunk_size": int(settings["chunk_size"]) if settings.get("chunk_size") is not None else None,
        "overlap": int(settings["overlap"]) if settings.get("overlap") is not None else None,
        "search_engine": settings["engine"],
        "top_k": int(settings["top_k"]),
        "candidate_count": len(smart_run.get("candidates", [])),
        "embedding_model": embedding_model,
        "embedding_dimension": int(embedding_dimension),
        "embedding_latency_ms": embedding_ms,
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": None,
        "total_latency_ms": embedding_ms + retrieval_ms,
        "top_1_chunk_id": top_chunk.get("chunk_id"),
        "top_1_source": top_chunk.get("book", ""),
        "top_1_semantic_score": float(top.get("semantic_score", 0.0)) if top else 0.0,
        "top_1_answerability_score": float(top.get("answerability_score", 0.0)) if top else None,
        "top_1_final_score": float(top.get("final_score", 0.0)) if top else 0.0,
        "retrieved_chunk_ids": [item.get("chunk", {}).get("chunk_id") for item in retrieved_results],
        "retrieved_sources": [item.get("chunk", {}).get("book", "") for item in retrieved_results],
        "answer_mode": "Retrieval Only",
        "generated_answer": None,
        "answer_quality": None,
        "evidence_quality": None,
        "correct_chunk_found": None,
        "notes": "",
        "knowledge_base_version": knowledge_base_version,
        "retrieved_results": stored_results,
        "query_embedding_available": bool(smart_run.get("query_embedding") is not None),
        "smart_reranking_applied": bool(retrieved_results),
        "explicit_evidence_support": any(
            float(item.get("answerability_score", 0.0)) > 0 and item.get("boost_reasons")
            for item in retrieved_results
        ),
        "generator_available": None,
        "answer_produced": False,
        "vectors_compared": vector_count if settings["engine"] == "Brute Force" else None,
        "indexed_vectors": vector_count if settings["engine"] == "ChromaDB" else None,
    }


def append_experiment(record: dict[str, Any]) -> str:
    history = initialize_experiment_history()
    history.append(record)
    st.session_state["latest_experiment_id"] = record["experiment_id"]
    return record["experiment_id"]


def update_experiment_generation(experiment_id: str, generation: dict[str, Any]) -> None:
    for record in initialize_experiment_history():
        if record["experiment_id"] == experiment_id:
            latency = float(generation.get("latency_ms", 0.0) or 0.0)
            record.update(
                answer_mode=generation.get("mode", "Gemini"),
                generated_answer=generation.get("answer"),
                generation_latency_ms=latency,
                total_latency_ms=record["embedding_latency_ms"] + record["retrieval_latency_ms"] + latency,
                generator_available=generation.get("mode") == "Gemini",
                answer_produced=bool(generation.get("answer")),
            )
            return


def update_experiment_rating(
    experiment_id: str,
    answer_quality: str,
    evidence_quality: str,
    correct_chunk_found: bool | None,
    notes: str,
) -> bool:
    for record in initialize_experiment_history():
        if record["experiment_id"] == experiment_id:
            record.update(
                answer_quality=answer_quality,
                evidence_quality=evidence_quality,
                correct_chunk_found=correct_chunk_found,
                notes=notes,
            )
            return True
    return False


def filter_experiments(history: list[dict[str, Any]], **filters) -> list[dict[str, Any]]:
    records = history
    for field, selected in filters.items():
        if not selected:
            continue
        values = set(selected if isinstance(selected, (list, tuple, set)) else [selected])
        records = [record for record in records if record.get(field) in values]
    return records


def compare_configurations(records: list[dict[str, Any]]) -> dict[str, Any]:
    quality_present = any(
        record.get("correct_chunk_found") is not None
        or record.get("answer_quality") not in (None, "Not rated")
        or record.get("evidence_quality") not in (None, "Not rated")
        for record in records
    )
    if len(records) < 2 or not quality_present:
        return {"winner_id": None, "message": "Not enough quality ratings to select a reliable winner."}
    answer_rank = {None: 0, "Not rated": 0, "Incorrect": 1, "Partially correct": 2, "Correct": 3}
    evidence_rank = {None: 0, "Not rated": 0, "Weak": 1, "Partial": 2, "Strong": 3}
    correct_rank = {None: 1, False: 0, True: 2}
    winner = max(
        records,
        key=lambda item: (
            correct_rank[item.get("correct_chunk_found")],
            answer_rank[item.get("answer_quality")],
            evidence_rank[item.get("evidence_quality")],
            -float(item.get("retrieval_latency_ms", 0)),
            float(item.get("top_1_final_score", 0)),
        ),
    )
    return {"winner_id": winner["experiment_id"], "message": "Winner selected using human quality ratings first."}


def build_chunking_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def compare_search_engines(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (
            record["question"], record["chunk_strategy"], record["chunk_size"], record["overlap"],
            record["top_k"], record["embedding_model"], record["knowledge_base_version"],
        )
        groups.setdefault(key, {})[record["search_engine"]] = record
    pairs = []
    for engines in groups.values():
        if {"Brute Force", "ChromaDB"}.issubset(engines):
            brute, chroma = engines["Brute Force"], engines["ChromaDB"]
            brute_ids, chroma_ids = brute["retrieved_chunk_ids"], chroma["retrieved_chunk_ids"]
            overlap = len(set(brute_ids) & set(chroma_ids))
            baseline = float(brute["retrieval_latency_ms"])
            difference = float(chroma["retrieval_latency_ms"]) - baseline
            pairs.append(
                {
                    "brute_force": brute,
                    "chromadb": chroma,
                    "latency_difference_ms": difference,
                    "latency_percentage_change": (difference / baseline * 100) if baseline else None,
                    "top_k_overlap_count": overlap,
                    "top_k_overlap_percentage": overlap / max(int(brute["top_k"]), 1) * 100,
                    "same_top_1": brute["top_1_chunk_id"] == chroma["top_1_chunk_id"],
                }
            )
    return pairs


def diagnose_experiment(record: dict[str, Any]) -> dict[str, Any]:
    stages = {
        "Query embedded": "Passed" if record.get("query_embedding_available") else "Failed",
        "Candidates retrieved": "Passed" if record.get("candidate_count", 0) else "Failed",
        "Smart reranking applied": "Passed" if record.get("smart_reranking_applied") else "Failed",
        "Relevant evidence found": (
            "Passed" if record.get("correct_chunk_found") is True or record.get("explicit_evidence_support")
            else "Failed" if record.get("correct_chunk_found") is False
            else "Warning"
        ),
        "Generator available": "Not applicable" if record.get("answer_mode") == "Retrieval Only" else (
            "Passed" if record.get("generator_available") else "Warning"
        ),
        "Answer produced": "Not applicable" if record.get("answer_mode") == "Retrieval Only" else (
            "Passed" if record.get("answer_produced") else "Failed"
        ),
    }
    if record.get("answer_mode") == "Retrieval Only":
        summary = "Retrieval succeeded; generation was not requested."
    elif record.get("correct_chunk_found") is False:
        summary = "The correct chunk was not found in Top K."
    elif record.get("evidence_quality") == "Weak":
        summary = "Retrieval returned chunks, but evidence quality was weak."
    elif record.get("correct_chunk_found") is True and record.get("answer_quality") == "Correct":
        summary = "Relevant evidence was found and the answer was rated correct."
    else:
        summary = "Review the retrieved evidence and add human quality ratings."
    return {"stages": stages, "summary": summary}


def _export_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {"retrieved_results"}
    return [
        {
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in record.items() if key not in excluded
        }
        for record in history
    ]


def export_experiments_csv(history: list[dict[str, Any]]) -> str:
    rows = _export_rows(history)
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def export_experiments_json(history: list[dict[str, Any]]) -> str:
    return json.dumps(history, indent=2, ensure_ascii=False)


def save_experiment_history_locally(
    history: list[dict[str, Any]], path: str = "data/experiment_history.json"
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(export_experiments_json(history), encoding="utf-8")
    return destination
