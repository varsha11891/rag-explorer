import json

from src.experiment_tracker import (
    compare_configurations,
    compare_search_engines,
    diagnose_experiment,
    export_experiments_csv,
    export_experiments_json,
)


def _record(engine="Brute Force", run="run-1"):
    return {
        "experiment_id": run,
        "question": "Who is McGonagall?",
        "chunk_strategy": "Semantic",
        "chunk_size": 800,
        "overlap": 100,
        "search_engine": engine,
        "top_k": 5,
        "embedding_model": "all-MiniLM-L6-v2",
        "knowledge_base_version": "abc123",
        "retrieval_latency_ms": 12.0 if engine == "Brute Force" else 8.0,
        "top_1_chunk_id": 481,
        "top_1_final_score": 0.72,
        "retrieved_chunk_ids": [481, 22, 7],
        "correct_chunk_found": None,
        "answer_quality": None,
        "evidence_quality": None,
        "candidate_count": 15,
        "query_embedding_available": True,
        "smart_reranking_applied": True,
        "explicit_evidence_support": True,
        "answer_mode": "Retrieval Only",
    }


def test_comparison_never_selects_winner_without_quality_ratings():
    comparison = compare_configurations([_record(run="a"), _record("ChromaDB", "b")])
    assert comparison["winner_id"] is None
    assert "Not enough quality ratings" in comparison["message"]


def test_search_engine_comparison_requires_compatible_configuration():
    pairs = compare_search_engines([_record(), _record("ChromaDB", "run-2")])
    assert len(pairs) == 1
    assert pairs[0]["same_top_1"] is True
    assert pairs[0]["top_k_overlap_count"] == 3
    assert pairs[0]["latency_difference_ms"] == -4.0


def test_retrieval_only_diagnosis_marks_generation_not_applicable():
    diagnosis = diagnose_experiment(_record())
    assert diagnosis["stages"]["Query embedded"] == "Passed"
    assert diagnosis["stages"]["Generator available"] == "Not applicable"
    assert diagnosis["stages"]["Answer produced"] == "Not applicable"


def test_experiment_exports_exclude_inspector_chunk_payload_from_csv():
    record = {**_record(), "retrieved_results": [{"chunk": {"text": "evidence"}}]}
    assert "retrieved_results" not in export_experiments_csv([record]).splitlines()[0]
    assert json.loads(export_experiments_json([record]))[0]["retrieved_results"]
