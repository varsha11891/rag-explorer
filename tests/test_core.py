import numpy as np
from unittest.mock import Mock, patch

from src.chunker import chunk_documents, fixed_size_chunks
from src.generator import PART_INSUFFICIENT_CONTEXT_MESSAGE, build_grounded_prompt, generate_answer
from src.retriever import brute_force_search
from src.utils import Document


def test_chunking_retains_provenance_and_overlap():
    chunks = fixed_size_chunks([Document("abcdefghij", "sample.pdf", 2)], chunk_size=6, overlap=2)
    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert all(chunk.source == "sample.pdf" and chunk.page == 2 for chunk in chunks)


def test_brute_force_ranks_cosine_similarity():
    chunks = fixed_size_chunks([Document("one two three four", "sample.pdf", 1)], 7, 0)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    results = brute_force_search(np.array([1.0, 0.0]), embeddings, chunks, top_k=2)
    assert results[0].chunk == chunks[0]
    assert results[0].score == 1.0


def test_generation_formats_cited_context_and_returns_gemini_metrics():
    chunk = chunk_documents([Document("Paris is the capital of France.", "facts.pdf", 1)], "Fixed size", 100, 0)[0]
    prompt = build_grounded_prompt("What is France's capital?", [chunk])
    assert "[Rank: 1 | Source: facts.pdf | Chunk: 1 | Similarity: 0.0000" in prompt
    response = Mock()
    response.text = "Paris. [Source: facts.pdf | Chunk: 1]"
    response.usage_metadata = Mock(prompt_token_count=30, candidates_token_count=10)
    with patch("src.generator.get_gemini_api_key", return_value="test-key"), patch(
        "src.generator.genai.Client"
    ) as client:
        client.return_value.models.generate_content.return_value = response
        result = generate_answer("What is France's capital?", [chunk])
    assert result["model"] == "gemini-3.5-flash"
    assert result["prompt_tokens"] == 30
    assert result["output_tokens"] == 10
    assert "Chunk: 1" in result["answer"]


def test_generation_uses_every_canonical_top_k_result_in_full():
    results = [
        {
            "similarity": 0.91,
            "chunk": {"text": "A" * 120, "book": "one.pdf", "chunk_id": 10, "start": 0, "end": 120},
        },
        {
            "similarity": 0.82,
            "chunk": {"text": "B" * 130, "book": "two.pdf", "chunk_id": 11, "start": 120, "end": 250},
        },
    ]
    prompt = build_grounded_prompt("What happened?", results)
    assert "[Rank: 1 | Source: one.pdf | Chunk: 10 | Similarity: 0.9100" in prompt
    assert "A" * 120 in prompt
    assert "[Rank: 2 | Source: two.pdf | Chunk: 11 | Similarity: 0.8200" in prompt
    assert "B" * 130 in prompt
    assert "Answer every sub-question separately" in prompt


def test_multipart_generation_answers_supported_part_without_refusing_all():
    question = "Who gave Harry his first broom, and why was he allowed to have it?"
    subquestions = [
        "Who gave Harry his first broom?",
        "Why was Harry allowed to have a broom?",
    ]
    results = [
        {
            "rank": 1,
            "similarity": 0.82,
            "semantic_score": 0.78,
            "answerability_score": 0.28,
            "originating_subqueries": [subquestions[0]],
            "chunk": {
                "text": "Professor McGonagall arranged for Harry to receive his first broom, a Nimbus Two Thousand. " * 3,
                "book": "Harry Potter 1.pdf",
                "chunk_id": 481,
                "start": 0,
                "end": 225,
                "page": 120,
            },
        }
    ]
    response = Mock()
    response.text = "I could not find enough information in the retrieved context."
    response.usage_metadata = Mock(prompt_token_count=40, candidates_token_count=18)
    with patch("src.generator.get_gemini_api_key", return_value="test-key"), patch(
        "src.generator.genai.Client"
    ) as client:
        client.return_value.models.generate_content.return_value = response
        generated = generate_answer(question, results, subquestions=subquestions)
    assert client.return_value.models.generate_content.call_count == 1
    assert generated["subquestion_answers"][0]["coverage_status"] == "Supported"
    assert generated["subquestion_answers"][1]["coverage_status"] == "Unsupported"
    assert generated["subquestion_answers"][1]["answer"] == PART_INSUFFICIENT_CONTEXT_MESSAGE
    assert "Professor McGonagall" in generated["answer"]
    assert "[Source: Harry Potter 1.pdf | Chunk: 481]" in generated["subquestion_answers"][0]["answer"]
    assert "I could not find enough information in the retrieved context." not in generated["subquestion_answers"][0]["answer"]
    assert PART_INSUFFICIENT_CONTEXT_MESSAGE in generated["answer"]


def _broom_result(text, chunk_id):
    return {
        "rank": chunk_id,
        "similarity": 0.8,
        "semantic_score": 0.75,
        "answerability_score": 0.2,
        "originating_subqueries": [],
        "chunk": {
            "text": text * 3,
            "book": "Harry Potter 1.pdf",
            "chunk_id": chunk_id,
            "start": 0,
            "end": len(text) * 3,
            "page": 100 + chunk_id,
        },
    }


def test_fully_supported_two_part_generation():
    question = "Who gave Harry his first broom and why?"
    subquestions = ["Who gave Harry his first broom?", "Why was Harry allowed to have his first broom?"]
    results = [
        _broom_result("Professor McGonagall arranged for Harry to receive his first broom. ", 1),
        _broom_result("Harry was allowed his first broom because he had joined the Gryffindor Quidditch team. ", 2),
    ]
    responses = []
    for text in ("McGonagall gave it.", "He was allowed because he joined the team."):
        response = Mock(text=text, usage_metadata=Mock(prompt_token_count=10, candidates_token_count=5))
        responses.append(response)
    with patch("src.generator.get_gemini_api_key", return_value="test-key"), patch(
        "src.generator.genai.Client"
    ) as client:
        client.return_value.models.generate_content.side_effect = responses
        generated = generate_answer(question, results, subquestions=subquestions)
    assert [part["coverage_status"] for part in generated["subquestion_answers"]] == ["Supported", "Supported"]
    assert client.return_value.models.generate_content.call_count == 2


def test_partially_supported_two_part_generation_states_unverified_detail():
    question = "Who gave Harry his first broom and why?"
    subquestions = ["Who gave Harry his first broom?", "Why was Harry allowed to have his first broom?"]
    results = [
        _broom_result("Professor McGonagall gave Harry his first broom. ", 1),
        _broom_result("Harry was allowed to have his first broom. ", 2),
    ]
    response = Mock(text="The passage says Harry was allowed a broom.", usage_metadata=Mock(prompt_token_count=10, candidates_token_count=5))
    with patch("src.generator.get_gemini_api_key", return_value="test-key"), patch(
        "src.generator.genai.Client"
    ) as client:
        client.return_value.models.generate_content.return_value = response
        generated = generate_answer(question, results, subquestions=subquestions)
    assert generated["subquestion_answers"][1]["coverage_status"] == "Partially supported"
    assert "could not be verified" in generated["subquestion_answers"][1]["answer"]


def test_fully_unsupported_multi_part_generation_skips_gemini():
    question = "Who gave Harry his first broom and why?"
    subquestions = ["Who gave Harry his first broom?", "Why was Harry allowed to have his first broom?"]
    results = [_broom_result("The castle corridor was quiet and empty on that winter evening. ", 9)]
    with patch("src.generator.get_gemini_api_key", return_value="test-key"), patch(
        "src.generator.genai.Client"
    ) as client:
        generated = generate_answer(question, results, subquestions=subquestions)
    assert client.return_value.models.generate_content.call_count == 0
    assert all(part["coverage_status"] == "Unsupported" for part in generated["subquestion_answers"])
    assert all(part["answer"] == PART_INSUFFICIENT_CONTEXT_MESSAGE for part in generated["subquestion_answers"])


def test_fixed_size_has_no_overlap():
    chunks = chunk_documents([Document("abcdefghij", "x.pdf", 1)], "Fixed size", 6, 3)
    assert [chunk.text for chunk in chunks] == ["abcdef", "ghij"]


def test_parent_child_links_children():
    chunks = chunk_documents([Document("a " * 100, "x.pdf", 1)], "Parent-child", 20, 5)
    parents = [chunk for chunk in chunks if chunk.level == "parent"]
    children = [chunk for chunk in chunks if chunk.level == "child"]
    assert parents and children
    assert all(child.parent_id in {parent.id for parent in parents} for child in children)


def test_chunk_ids_are_global_across_documents():
    documents = [Document("abcdefgh", "a.pdf", 1), Document("ijklmnop", "b.pdf", 1)]
    chunks = chunk_documents(documents, "Fixed size", 4, 0)
    assert [chunk.global_id for chunk in chunks] == [1, 2, 3, 4]
    assert chunks[2].source == "b.pdf"
