import numpy as np

from src.smart_retriever import decompose_query, detect_query_intent, rerank_candidates, smart_retrieve
from src.utils import Chunk, SearchResult


def _result(text: str, score: float, chunk_id: int) -> SearchResult:
    chunk = Chunk(
        id=str(chunk_id), text=text, source="Harry Potter.pdf", page=1,
        start=0, end=len(text), global_id=chunk_id,
    )
    return SearchResult(chunk=chunk, score=score, rank=chunk_id)


def test_supported_intents_and_transformations():
    cases = {
        "Who is McGonagall?": ("who_is", "McGonagall", "Professor Minerva McGonagall"),
        "Who is Rubeus Hagrid?": ("who_is", "Rubeus Hagrid", "Rubeus Hagrid"),
        "What is a Horcrux?": ("what_is", "Horcrux", "definition"),
        "Where is Hogwarts?": ("where_is", "Hogwarts", "location"),
        "Why did Harry go to Hogwarts?": ("why_did", "Harry go to Hogwarts", "reason"),
    }
    for query, (intent, subject, expansion) in cases.items():
        detected = detect_query_intent(query)
        assert detected["intent_type"] == intent
        assert detected["extracted_entity_or_subject"] == subject
        assert expansion in detected["transformed_retrieval_query"]


def test_who_is_promotes_role_evidence_above_action_mentions():
    candidates = [
        _result("Professor McGonagall appeared and walked toward Harry.", 0.90, 1),
        _result(
            "Professor McGonagall was a teacher, head of Gryffindor house, and responsible for students.",
            0.88,
            2,
        ),
    ]
    reranked = rerank_candidates("Who is McGonagall?", "who_is", "McGonagall", candidates)
    assert reranked[0]["chunk"].global_id == 2
    assert reranked[0]["answerability_score"] > reranked[1]["answerability_score"]


def test_unknown_intent_preserves_query_and_fewer_candidates_are_returned():
    class FakeStore:
        def search(self, query_vector, top_k):
            assert top_k == 15
            assert np.array_equal(query_vector, np.array([1.0, 0.0]))
            return [_result("An otherwise relevant passage.", 0.5, 1)]

    query = "Compare the two houses"
    run = smart_retrieve(query, FakeStore(), lambda value: np.array([1.0, 0.0]))
    assert run["intent_type"] == "unknown"
    assert run["transformed_retrieval_query"] == query
    assert len(run["candidates"]) == 1
    assert len(run["results"]) == 1


def test_multipart_broom_question_is_decomposed_and_covered():
    query = "Who gave Harry his first broom, and why was he allowed to have it?"
    expected_subqueries = [
        "Who gave Harry his first broom?",
        "Why was Harry allowed to have a broom?",
    ]
    assert decompose_query(query) == expected_subqueries

    embedded_queries = []

    def embed(value):
        embedded_queries.append(value)
        return np.array([float(len(embedded_queries)), 0.0])

    class FakeStore:
        def search(self, query_vector, top_k):
            assert top_k == 8
            if query_vector[0] == 1.0:
                return [
                    _result("Professor McGonagall gave Harry his first broomstick, a Nimbus Two Thousand.", 0.91, 10),
                    _result("Harry carried his broom through the corridor.", 0.70, 11),
                ]
            return [
                _result("Harry was allowed a broom because he had joined the Gryffindor Quidditch team.", 0.90, 12),
                _result("Harry carried his broom through the corridor.", 0.69, 11),
            ]

    run = smart_retrieve(query, FakeStore(), embed)
    assert run["is_multipart"] is True
    assert run["decomposed_subqueries"] == expected_subqueries
    assert len(run["candidates_by_subquery"]) == 2
    assert len(run["merged_candidates"]) == 3
    assert run["coverage_score"] == 2
    supported = {subquery for item in run["results"] for subquery in item["originating_subqueries"]}
    assert supported == set(expected_subqueries)


def test_elliptical_and_why_resolves_subject_and_object():
    assert decompose_query("Who gave Harry his first broom and why?") == [
        "Who gave Harry his first broom?",
        "Why was Harry allowed to have his first broom?",
    ]
