"""Grounded answer generation with the hosted Gemini API."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import types

from src.smart_retriever import inspect_subquestion_evidence
from src.utils import Chunk, SearchResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL = "gemini-3.5-flash"
MAX_OUTPUT_TOKENS = 512
REQUEST_TIMEOUT_MS = 120_000
INSUFFICIENT_CONTEXT_MESSAGE = "I could not find enough information in the retrieved context."
PART_INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find enough information in the retrieved context for this part."
)

SYSTEM_PROMPT = f"""You are a RAG assistant.

Answer the question directly in 2–4 sentences.
Combine explicitly supported facts across the retrieved chunks into a natural, beginner-friendly explanation.
Use simple language, short sentences, and avoid unnecessary jargon.
Do not begin with "Based on the retrieved context" or "According to the provided text".
Use ONLY the supplied retrieved context. Never use outside knowledge or unsupported assumptions.
For multi-part questions, answer every listed sub-question separately.
If evidence supports only some parts, clearly identify which parts are supported and which cannot be answered.

If the answer cannot be found, say exactly:
'{INSUFFICIENT_CONTEXT_MESSAGE}'

Place supporting citations at the end of the answer using:
[Source: filename | Chunk: number]

Never invent facts, books, chunk IDs, or citations."""


class GenerationBackendError(RuntimeError):
    """Raised when Gemini generation cannot complete."""


def is_fallback_eligible_generation_error(error: BaseException) -> bool:
    """Return true only for missing or transient Gemini backend failures."""
    message = str(error).casefold()
    markers = (
        "api_key is missing", "rate limit", "quota", "resource_exhausted", "429",
        "timeout", "timed out", "deadline", "temporar", "unavailable", "500", "502",
        "503", "504",
    )
    return isinstance(error, GenerationBackendError) and any(marker in message for marker in markers)


def _secret(name: str) -> str | None:
    """Read a value only from Streamlit secrets without exposing it."""
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def get_gemini_api_key() -> str | None:
    key = _secret("GEMINI_API_KEY")
    return None if key == "replace-with-your-gemini-api-key" else key


def get_model_name() -> str:
    return _secret("GEMINI_MODEL") or DEFAULT_MODEL


def _normalize_result(item: SearchResult | Chunk | dict[str, Any], rank: int) -> dict[str, Any]:
    """Normalize object and session-state results to the canonical payload."""
    if isinstance(item, dict):
        chunk = item.get("chunk", {})
        return {
            "rank": int(item.get("rank", rank)),
            "similarity": float(item.get("similarity", 0.0)),
            "chunk": {
                "text": str(chunk.get("text", "")),
                "book": str(chunk.get("book", "")),
                "chunk_id": int(chunk.get("chunk_id", 0)),
                "start": int(chunk.get("start", 0)),
                "end": int(chunk.get("end", 0)),
                "page": int(chunk.get("page", 0)),
            },
            "originating_subqueries": list(item.get("originating_subqueries", [])),
        }
    chunk = item.chunk if isinstance(item, SearchResult) else item
    similarity = item.score if isinstance(item, SearchResult) else 0.0
    return {
        "rank": item.rank if isinstance(item, SearchResult) else rank,
        "similarity": float(similarity),
        "chunk": {
            "text": chunk.text,
            "book": chunk.source,
            "chunk_id": chunk.global_id,
            "start": chunk.start,
            "end": chunk.end,
            "page": chunk.page,
        },
        "originating_subqueries": [],
    }


def format_retrieved_context(
    retrieved_results: Sequence[SearchResult | Chunk | dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Format the latest top-K chunks and collect their source metadata."""
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    for position, item in enumerate(retrieved_results, start=1):
        normalized = _normalize_result(item, position)
        chunk = normalized["chunk"]
        similarity = normalized["similarity"]
        blocks.append(
            f"[Rank: {normalized['rank']} | Source: {chunk['book']} | Chunk: {chunk['chunk_id']} | "
            f"Similarity: {similarity:.4f} | Supports: "
            f"{'; '.join(normalized.get('originating_subqueries', [])) or 'original question'}]\n"
            f"{chunk['text']}"
        )
        sources.append(
            {
                "book": chunk["book"],
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "similarity": similarity,
            }
        )
    context = "\n\n".join(blocks)
    logger.info("Final Gemini context length: %d characters", len(context))
    return context, sources


def build_grounded_prompt(
    query: str,
    retrieved_results: Sequence[SearchResult | Chunk | dict[str, Any]],
    subquestions: Sequence[str] | None = None,
) -> str:
    context, _ = format_retrieved_context(retrieved_results)
    questions = list(subquestions or [query])
    subquestion_block = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    return f"""User question:
{query}

Sub-questions that must each be addressed:
{subquestion_block}

Retrieved context:
{context}

Answer every sub-question separately using only its supported evidence. If a sub-question lacks evidence, say which part could not be answered. Keep the complete response concise and put Source and Chunk ID citations at the end."""


def classify_subquestion_coverage(
    subquestion: str,
    retrieved_results: Sequence[SearchResult | Chunk | dict[str, Any]],
) -> tuple[str, list[SearchResult | Chunk | dict[str, Any]]]:
    """Compatibility wrapper around deterministic chunk-content inspection."""
    inspection = inspect_subquestion_evidence(
        subquestion, [item for item in retrieved_results if isinstance(item, dict)]
    )
    return inspection["coverage_status"], [
        record["result"] for record in inspection["supporting_chunks"]
    ]


def _evidence_citations(records: Sequence[dict[str, Any]]) -> str:
    return " ".join(dict.fromkeys(
        f"[Source: {record['source']} | Chunk: {record['chunk_id']}]" for record in records
    ))


def _is_abstention(answer: str) -> bool:
    lowered = answer.casefold()
    return (
        INSUFFICIENT_CONTEXT_MESSAGE.casefold() in lowered
        or PART_INSUFFICIENT_CONTEXT_MESSAGE.casefold() in lowered
    )


def _extractive_evidence_answer(records: Sequence[dict[str, Any]], partial: bool = False) -> str:
    if not records:
        return PART_INSUFFICIENT_CONTEXT_MESSAGE
    answer = records[0]["excerpt"].strip() + " " + _evidence_citations(records)
    if partial:
        answer += " The remaining details could not be verified from the retrieved context."
    return answer


def _gemini_request(client, model: str, prompt: str, api_key: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
    except Exception as error:
        message = str(error).replace(api_key, "[redacted]")
        if "timeout" in message.lower() or "timed out" in message.lower():
            raise GenerationBackendError("Gemini generation timed out. Please try again.") from error
        raise GenerationBackendError(f"Gemini generation failed: {message}") from error
    try:
        answer = str(response.text or "").strip()
    except Exception as error:
        raise GenerationBackendError("Gemini did not return a readable text answer.") from error
    if not answer:
        raise GenerationBackendError("Gemini returned an empty answer.")
    usage = getattr(response, "usage_metadata", None)
    return {
        "answer": answer,
        "latency_ms": (time.perf_counter() - started) * 1_000,
        "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def generate_answer(
    query: str,
    retrieved_results: Sequence[SearchResult | Chunk | dict[str, Any]],
    model_name: str | None = None,
    subquestions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate a grounded answer using the latest retrieved top-K chunks."""
    if not query.strip():
        raise ValueError("Run a retrieval search before generating an answer.")
    if not retrieved_results:
        raise ValueError("No retrieved chunks are available. Run a retrieval search first.")
    context, sources = format_retrieved_context(retrieved_results)
    if not context.strip():
        raise ValueError("The retrieved context is empty. Run retrieval again before generation.")
    if len(context) < 100:
        raise ValueError(
            f"The retrieved context is too short ({len(context)} characters). At least 100 characters are required."
        )
    api_key = get_gemini_api_key()
    if not api_key:
        raise GenerationBackendError(
            "GEMINI_API_KEY is missing. The app owner must configure it in Streamlit secrets."
        )

    model = model_name or get_model_name()
    prompt = build_grounded_prompt(query, retrieved_results, subquestions)
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    questions = list(subquestions or [query])
    if len(questions) > 1:
        part_results: list[dict[str, Any]] = []
        prompts_sent: list[str] = []
        total_latency = 0.0
        prompt_counts: list[int] = []
        output_counts: list[int] = []
        all_sources: list[dict[str, Any]] = []
        for index, subquestion in enumerate(questions, start=1):
            inspection = inspect_subquestion_evidence(
                subquestion, [item for item in retrieved_results if isinstance(item, dict)]
            )
            coverage = inspection["coverage_status"]
            evidence_records = inspection["supporting_chunks"]
            evidence = [record["result"] for record in evidence_records]
            public_supporting_chunks = [
                {key: value for key, value in record.items() if key != "result"}
                for record in evidence_records
            ]
            if coverage == "Unsupported":
                part_answer = PART_INSUFFICIENT_CONTEXT_MESSAGE
                part_sources: list[dict[str, Any]] = []
            else:
                part_prompt = build_grounded_prompt(subquestion, evidence, [subquestion]) + (
                    f"\n\nThis part is classified as {coverage}. Never fill missing details from outside "
                    f"knowledge. If the supplied evidence cannot answer it, return exactly: "
                    f"{PART_INSUFFICIENT_CONTEXT_MESSAGE}"
                )
                prompts_sent.append(f"SUB-QUESTION {index}\n\n{part_prompt}")
                generated = _gemini_request(client, model, part_prompt, api_key)
                part_answer = generated["answer"]
                if _is_abstention(part_answer):
                    part_answer = _extractive_evidence_answer(
                        evidence_records, partial=coverage == "Partially supported"
                    )
                citations = _evidence_citations(evidence_records)
                if citations and citations not in part_answer:
                    part_answer = f"{part_answer.rstrip()} {citations}"
                if coverage == "Partially supported" and "could not be verified" not in part_answer.casefold():
                    part_answer += " The remaining details could not be verified from the retrieved context."
                total_latency += generated["latency_ms"]
                if generated["prompt_tokens"] is not None:
                    prompt_counts.append(generated["prompt_tokens"])
                if generated["output_tokens"] is not None:
                    output_counts.append(generated["output_tokens"])
                _, part_sources = format_retrieved_context(evidence)
                all_sources.extend(part_sources)
            part_results.append(
                {
                    "sub_question": subquestion,
                    "coverage_status": coverage,
                    "supporting_chunks": public_supporting_chunks,
                    "evidence_summary": inspection["evidence_summary"],
                    "answer": part_answer,
                }
            )
        answer = "\n\n".join(
            f"**{index}. {part['sub_question']}**\n\n"
            f"Evidence coverage: {part['coverage_status']}\n\n{part['answer']}"
            for index, part in enumerate(part_results, start=1)
        )
        unique_sources = list({
            (source["book"], source["chunk_id"]): source for source in all_sources
        }.values())
        return {
            "answer": answer,
            "latency_ms": total_latency,
            "model": model,
            "prompt_tokens": sum(prompt_counts) if prompt_counts else 0,
            "output_tokens": sum(output_counts) if output_counts else 0,
            "sources_used": unique_sources,
            "subquestion_answers": part_results,
            "prompt": "\n\n---\n\n".join(prompts_sent),
        }

    generated = _gemini_request(client, model, prompt, api_key)
    return {
        "answer": generated["answer"],
        "latency_ms": generated["latency_ms"],
        "model": model,
        "prompt_tokens": generated["prompt_tokens"],
        "output_tokens": generated["output_tokens"],
        "sources_used": sources,
        "prompt": prompt,
    }
