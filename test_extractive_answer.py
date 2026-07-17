import numpy as np
from unittest.mock import patch

from src.extractive_answer import (
    INSUFFICIENT_EXTRACTIVE_MESSAGE,
    generate_extractive_answer,
    rank_sentences,
    split_chunks_into_sentences,
)


def _result(text, chunk_id=1):
    return {
        "similarity": 0.8,
        "chunk": {
            "text": text,
            "book": "book.pdf",
            "chunk_id": chunk_id,
            "page": 1,
            "start": 0,
            "end": len(text),
        },
    }


def test_sentence_split_preserves_source_and_chunk_id():
    sentences = split_chunks_into_sentences(
        [_result("Professor McGonagall gave Harry the broom. It was a Nimbus Two Thousand.", 481)]
    )
    assert [sentence["chunk_id"] for sentence in sentences] == [481, 481]
    assert all(sentence["source"] == "book.pdf" for sentence in sentences)


def test_sentence_ranking_uses_local_cosine_similarity():
    sentences = [
        {"text": "The corridor was empty that night.", "source": "book.pdf", "chunk_id": 1},
        {"text": "McGonagall gave Harry his first broom.", "source": "book.pdf", "chunk_id": 2},
    ]
    with patch(
        "src.extractive_answer.embed_texts",
        return_value=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    ), patch(
        "src.extractive_answer.embed_query",
        return_value=np.array([1.0, 0.0], dtype=np.float32),
    ):
        ranked = rank_sentences("Who gave Harry his first broom?", sentences)
    assert ranked[0]["chunk_id"] == 2
    assert ranked[0]["similarity"] == 1.0


def test_extractive_multipart_abstains_only_for_unsupported_part():
    questions = ["Who gave Harry his first broom?", "Why was Harry allowed to have his first broom?"]
    results = [_result("Professor McGonagall gave Harry his first broom.", 481)]
    with patch(
        "src.extractive_answer.embed_texts",
        return_value=np.array([[1.0, 0.0]], dtype=np.float32),
    ), patch(
        "src.extractive_answer.embed_query",
        return_value=np.array([1.0, 0.0], dtype=np.float32),
    ):
        answer = generate_extractive_answer(
            "Who gave Harry his first broom and why?", results, subquestions=questions
        )
    assert answer["mode"] == "Extractive"
    assert answer["subquestion_answers"][0]["coverage_status"] == "Supported"
    assert "McGonagall" in answer["subquestion_answers"][0]["answer"]
    assert answer["subquestion_answers"][1]["coverage_status"] == "Unsupported"
    assert answer["subquestion_answers"][1]["answer"] == INSUFFICIENT_EXTRACTIVE_MESSAGE
