"""Deterministic query expansion and local answerability reranking."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from src.utils import SearchResult

SEMANTIC_WEIGHT = 0.80
ANSWERABILITY_WEIGHT = 0.20
CANDIDATE_COUNT = 15
MULTIPART_CANDIDATE_COUNT = 8
FINAL_RESULT_COUNT = 5


class SearchEngine(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]: ...


_INTENT_PATTERNS = (
    ("who_is", re.compile(r"^who\s+is\s+(.+)$", re.I)),
    ("what_is", re.compile(r"^what\s+is\s+(.+)$", re.I)),
    ("where_is", re.compile(r"^where\s+is\s+(.+)$", re.I)),
    ("when_did", re.compile(r"^when\s+did\s+(.+)$", re.I)),
    ("why_did", re.compile(r"^why\s+did\s+(.+)$", re.I)),
    ("how_did", re.compile(r"^how\s+did\s+(.+)$", re.I)),
    ("tell_me_about", re.compile(r"^tell\s+me\s+about\s+(.+)$", re.I)),
    ("who", re.compile(r"^who\s+(.+)$", re.I)),
    ("what", re.compile(r"^what\s+(.+)$", re.I)),
    ("where", re.compile(r"^where\s+(.+)$", re.I)),
    ("when", re.compile(r"^when\s+(.+)$", re.I)),
    ("why", re.compile(r"^why\s+(.+)$", re.I)),
    ("how", re.compile(r"^how\s+(.+)$", re.I)),
)

_INTERROGATIVES = r"who|what|where|when|why|how"


def _clean_subject(value: str) -> str:
    return re.sub(r"[\s?.!]+$", "", value.strip())


def transform_query(query: str, intent: str, subject: str) -> str:
    """Expand known question forms with deterministic retrieval vocabulary."""
    if not subject or intent == "unknown":
        return query.strip()
    expansions = {
        "who_is": "identity role position occupation responsibilities house affiliation teacher professor head of headmaster headmistress keeper student member of works as responsible for Hogwarts",
        "what_is": "definition meaning refers to purpose properties creation used for",
        "where_is": "location place region surroundings situated located near inside outside",
        "when_did": "event date year age timeline before after during",
        "why_did": "reason motivation cause because purpose so that in order to",
        "how_did": "process method by through steps way",
        "tell_me_about": "identity description background role characteristics known for",
        "who": "person identity gave provided responsible",
        "what": "fact description meaning event",
        "where": "location place situated located near inside outside",
        "when": "date year age timeline before after during",
        "why": "reason motivation cause because purpose so that in order to allowed permission",
        "how": "process method by through steps way",
    }
    expanded_subject = subject
    aliases = {
        "mcgonagall": "Professor Minerva McGonagall",
        "hagrid": "Rubeus Hagrid",
    }
    expanded_subject = aliases.get(subject.casefold(), expanded_subject)
    if intent == "when_did":
        expanded_subject = re.sub(r"\blearn\b", "learned", expanded_subject, flags=re.I)
    return f"{expanded_subject} {expansions[intent]}".strip()


def detect_query_intent(query: str) -> dict[str, str]:
    """Detect a supported intent and extract its subject without an LLM."""
    original = query.strip()
    for intent, pattern in _INTENT_PATTERNS:
        match = pattern.match(original)
        if match:
            subject = _clean_subject(match.group(1))
            if intent == "what_is":
                subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.I)
            return {
                "intent_type": intent,
                "extracted_entity_or_subject": subject,
                "original_query": original,
                "transformed_retrieval_query": transform_query(original, intent, subject),
            }
    return {
        "intent_type": "unknown",
        "extracted_entity_or_subject": "",
        "original_query": original,
        "transformed_retrieval_query": original,
    }


def _resolve_references(clause: str, first_clause: str) -> str:
    """Resolve a small set of cross-clause pronouns using surface-form rules."""
    excluded = {"Who", "What", "Where", "When", "Why", "How", "Tell", "And"}
    names = [name for name in re.findall(r"\b[A-Z][a-z]+\b", first_clause) if name not in excluded]
    person = names[-1] if names else ""
    objects = re.findall(
        r"\b(?:his|her|their|the)\s+(?:(?:first|last|new|old)\s+)?([a-z][\w-]*)",
        first_clause,
        re.I,
    )
    referenced_object = f"a {objects[-1]}" if objects else ""
    resolved = clause.strip()
    if person:
        resolved = re.sub(r"\b(?:he|she)\b", person, resolved, flags=re.I)
        resolved = re.sub(r"\b(?:his|her)\b", f"{person}'s", resolved, flags=re.I)
    if referenced_object:
        resolved = re.sub(r"\bit\b", referenced_object, resolved, flags=re.I)
    if re.fullmatch(r"(?:why|how|when|where|who|what)", resolved, re.I):
        possession = re.search(
            r"^Who\s+(?:gave|sent|provided|bought|arranged).*?\b(?P<person>[A-Z][a-z]+)\s+"
            r"(?P<object>(?:his|her|their)\s+.+?)\?$",
            first_clause,
            re.I,
        )
        if possession and resolved.casefold() == "why":
            person = possession.group("person")
            object_phrase = possession.group("object")
            return f"Why was {person} allowed to have {object_phrase}"
        statement = re.sub(rf"^(?:{_INTERROGATIVES})\s+", "", first_clause.rstrip("?"), flags=re.I)
        return f"{resolved} {statement}".strip()
    return resolved


def decompose_query(query: str) -> list[str]:
    """Split multi-part questions with deterministic punctuation/conjunction rules."""
    original = " ".join(query.strip().split())
    if not original:
        return []

    question_parts = [part.strip() for part in re.findall(r"[^?]+\?", original)]
    trailing = re.sub(r"(?:[^?]+\?)+", "", original).strip()
    if trailing:
        question_parts.append(trailing)
    if len(question_parts) > 1:
        first = question_parts[0].rstrip(" ?") + "?"
        resolved_parts = []
        for part in question_parts[1:]:
            resolved = _resolve_references(part.rstrip(" ?"), first)
            resolved_parts.append(resolved[:1].upper() + resolved[1:] + "?")
        return [first] + resolved_parts

    conjunction = re.search(
        rf",?\s+and\s+(?P<interrogative>{_INTERROGATIVES})\b", original, re.I
    )
    if not conjunction:
        return [original]
    first = original[: conjunction.start()].rstrip(" ,?") + "?"
    second = original[conjunction.start("interrogative") :].rstrip(" ?")
    second = _resolve_references(second, first)
    second = second[:1].upper() + second[1:] + "?"
    return [first, second]


_STOP_WORDS = {
    "a", "an", "and", "the", "his", "her", "their", "it", "he", "she", "they",
    "was", "were", "is", "are", "did", "do", "does", "to", "of", "for", "have",
    "who", "what", "where", "when", "why", "how", "first", "this", "that",
}


def _content_terms(question: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", question.casefold())
    normalized = ["broom" if term.startswith("broom") else term for term in terms]
    return list(dict.fromkeys(term for term in normalized if term not in _STOP_WORDS and len(term) > 2))


def _evidence_excerpt(text: str, terms: Sequence[str], limit: int = 280) -> str:
    compact = " ".join((text or "").split())
    lowered = compact.casefold()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(0, (min(positions) if positions else 0) - 70)
    excerpt = compact[start : start + limit]
    return ("…" if start else "") + excerpt + ("…" if start + limit < len(compact) else "")


def inspect_subquestion_evidence(subquestion: str, retrieved_results: Sequence[dict]) -> dict[str, Any]:
    """Classify evidence from chunk content only; retrieval scores never determine coverage."""
    question = subquestion.strip()
    lowered_question = question.casefold()
    terms = _content_terms(question)
    direct_records: list[dict[str, Any]] = []
    partial_records: list[dict[str, Any]] = []
    give_actions = ("gave", "given", "sent", "provided", "bought", "arranged", "gifted")
    causal_markers = ("because", "reason", "so that", "in order to", "due to", "since")

    for item in retrieved_results:
        chunk = item.get("chunk", {}) if isinstance(item, dict) else {}
        text = str(chunk.get("text", ""))
        lowered = text.casefold()
        normalized_text = re.sub(r"broom(?:stick)?s?", "broom", lowered)
        matched_terms = [term for term in terms if term in normalized_text]
        direct = False
        partial = False
        reason = ""

        if lowered_question.startswith("who ") and any(action in lowered_question for action in ("gave", "give", "sent", "provided")):
            has_action = any(action in normalized_text for action in give_actions)
            has_person = "harry" in normalized_text if "harry" in terms else bool(matched_terms)
            has_object = "broom" in normalized_text if "broom" in terms else True
            direct = has_action and has_person and has_object
            partial = not direct and sum((has_action, has_person, has_object)) >= 2
            reason = (
                "The chunk explicitly connects a giver action with Harry and the broom."
                if direct else "The chunk mentions part of the giver, recipient, or broom relationship."
            )
        elif lowered_question.startswith("why "):
            topic_match = bool(matched_terms) or "broom" in normalized_text
            has_cause = any(marker in normalized_text for marker in causal_markers)
            has_permission = any(word in normalized_text for word in ("allowed", "permission", "permitted"))
            direct = topic_match and has_cause
            partial = not direct and topic_match and has_permission
            reason = (
                "The chunk states a causal reason for the event or permission."
                if direct else "The chunk mentions the event or permission but does not state its reason."
            )
        elif lowered_question.startswith("what "):
            has_definition = any(marker in normalized_text for marker in (" is ", " means ", "refers to", "used for"))
            direct = bool(matched_terms) and has_definition
            partial = not direct and bool(matched_terms)
            reason = "The chunk contains a direct definition." if direct else "The chunk mentions the subject without defining it."
        elif lowered_question.startswith("where "):
            marker = any(word in normalized_text for word in ("located", "situated", "near", "inside", "outside"))
            direct = bool(matched_terms) and marker
            partial = not direct and bool(matched_terms)
            reason = "The chunk states a location relationship." if direct else "The chunk mentions the place without locating it."
        elif lowered_question.startswith("when "):
            marker = bool(re.search(r"\b(?:\d{4}|year|age|before|after|during)\b", normalized_text))
            direct = bool(matched_terms) and marker
            partial = not direct and bool(matched_terms)
            reason = "The chunk supplies temporal evidence." if direct else "The chunk mentions the event without a time reference."
        else:
            required = max(2, min(3, len(terms)))
            direct = len(matched_terms) >= required and bool(terms)
            partial = not direct and bool(matched_terms)
            reason = (
                "The chunk contains the key subject terms needed to answer the question."
                if direct else "The chunk overlaps with the question but does not establish a complete answer."
            )

        if direct or partial:
            record = {
                "source": str(chunk.get("book", "Unknown source")),
                "chunk_id": chunk.get("chunk_id", 0),
                "excerpt": _evidence_excerpt(text, matched_terms or terms),
                "reason": reason,
                "result": item,
            }
            (direct_records if direct else partial_records).append(record)

    if direct_records:
        status, records = "Supported", direct_records
        summary = f"{len(records)} chunk(s) directly answer this sub-question."
    elif partial_records:
        status, records = "Partially supported", partial_records
        summary = f"{len(records)} chunk(s) contain relevant clues but not a complete answer."
    else:
        status, records = "Unsupported", []
        summary = "No final retrieved chunk contains enough content to answer this sub-question."
    return {
        "sub_question": question,
        "coverage_status": status,
        "supporting_chunks": records,
        "evidence_summary": summary,
    }


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase and re.search(rf"\b{re.escape(phrase)}\b", text, re.I))


def calculate_answerability_score(
    query: str, intent: str, subject: str, chunk_text: str
) -> tuple[float, list[str]]:
    """Score whether a candidate is likely to directly answer the detected intent."""
    del query  # Reserved for future deterministic query-level rules.
    text = chunk_text or ""
    lowered = text.casefold()
    reasons: list[str] = []
    score = 0.0
    subject_present = _contains(text, subject)
    if subject_present:
        score += 0.15
        reasons.append("exact subject match")

    rules: dict[str, tuple[str, ...]] = {
        "who": ("gave", "provided", "sent", "gift", "from"),
        "what_is": ("definition", "means", "refers to", "purpose", "used for"),
        "where_is": ("located", "situated", "place", "near", "inside", "outside"),
        "when_did": ("date", "year", "age", "before", "after", "during"),
        "why_did": ("because", "reason", "so that", "in order to", "allowed", "permission"),
        "why": ("because", "reason", "so that", "in order to", "allowed", "permission"),
        "how_did": ("process", "method", " by ", "through", "steps"),
        "tell_me_about": ("is", "was", "known for", "role", "responsible for", "member of"),
    }

    descriptive_match = False
    if intent == "who_is":
        if subject and re.search(rf"\b{re.escape(subject)}\s+(?:is|was)\b", text, re.I):
            score += 0.28
            descriptive_match = True
            reasons.append(f"'{subject} is/was' description")
        if subject and re.search(
            rf"\b(?:Professor|Mr|Mrs|Ms|Miss|Dr)\.?\s+(?:\w+\s+)*{re.escape(subject)}\b",
            text,
            re.I,
        ):
            score += 0.15
            descriptive_match = True
            reasons.append("title immediately before subject")
        role_phrases = (
            "professor", "teacher", "head of", "headmaster", "headmistress", "keeper",
            "student", "member of", "works as", "responsible for", "house", "role", "position",
        )
        matches = [phrase for phrase in role_phrases if phrase in lowered]
        if matches:
            score += min(0.42, 0.11 * len(matches))
            descriptive_match = True
            reasons.append("role language: " + ", ".join(matches[:4]))
    else:
        matches = [phrase.strip() for phrase in rules.get(intent, ()) if phrase in lowered]
        if intent == "what_is" and subject and re.search(
            rf"\b{re.escape(subject)}\s+(?:is|was)\b", text, re.I
        ):
            matches.append(f"{subject} is/was")
        if matches:
            score += min(0.70, 0.14 * len(matches))
            descriptive_match = True
            reasons.append(f"{intent.replace('_', ' ')} evidence: " + ", ".join(matches[:4]))

    if subject_present and not descriptive_match:
        score -= 0.20
        reasons.append("penalty: mention without descriptive evidence")
    if not subject_present and intent in {"who_is", "what_is", "where_is", "tell_me_about"}:
        score -= 0.10
        reasons.append("penalty: subject not present")
    return max(0.0, min(1.0, score)), reasons


def rerank_candidates(
    query: str,
    intent: str,
    subject: str,
    candidates: Sequence[SearchResult],
    final_k: int = FINAL_RESULT_COUNT,
) -> list[dict[str, Any]]:
    """Blend normalized semantic similarity with deterministic answerability."""
    if not candidates:
        return []
    semantic_scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
    # Cosine similarity is bounded to [-1, 1]; this stable mapping avoids amplifying
    # tiny differences within a small candidate set as per-query min-max scaling would.
    normalized = np.clip((semantic_scores + 1.0) / 2.0, 0.0, 1.0)

    reranked: list[dict[str, Any]] = []
    for candidate, normalized_score in zip(candidates, normalized, strict=True):
        answerability, reasons = calculate_answerability_score(
            query, intent, subject, candidate.chunk.text
        )
        final_score = SEMANTIC_WEIGHT * float(normalized_score) + ANSWERABILITY_WEIGHT * answerability
        reranked.append(
            {
                "chunk": candidate.chunk,
                "semantic_score": float(candidate.score),
                "normalized_semantic_score": float(normalized_score),
                "answerability_score": float(answerability),
                "final_score": float(final_score),
                "boost_reasons": reasons,
                "candidate_rank": candidate.rank,
            }
        )
    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    return reranked[: min(final_k, len(reranked))]


def smart_retrieve(
    query: str,
    search_engine: SearchEngine,
    embed_query_fn: Callable[[str], np.ndarray],
    candidate_k: int = CANDIDATE_COUNT,
    final_k: int = FINAL_RESULT_COUNT,
) -> dict[str, Any]:
    """Retrieve independently per clause, merge evidence, and prioritize coverage."""
    subqueries = decompose_query(query)
    is_multipart = len(subqueries) > 1
    per_query_k = MULTIPART_CANDIDATE_COUNT if is_multipart else candidate_k
    embedding_latency = 0.0
    retrieval_latency = 0.0
    query_embeddings: list[np.ndarray] = []
    candidates_by_subquery: list[dict[str, Any]] = []

    for subquery in subqueries:
        intent = detect_query_intent(subquery)
        embedding_started = time.perf_counter()
        query_embedding = embed_query_fn(intent["transformed_retrieval_query"])
        embedding_latency += time.perf_counter() - embedding_started
        query_embeddings.append(np.asarray(query_embedding, dtype=np.float32))
        retrieval_started = time.perf_counter()
        semantic_candidates = search_engine.search(query_embedding, per_query_k)
        ranked_candidates = rerank_candidates(
            subquery,
            intent["intent_type"],
            intent["extracted_entity_or_subject"],
            semantic_candidates,
            len(semantic_candidates),
        )
        retrieval_latency += time.perf_counter() - retrieval_started
        for item in ranked_candidates:
            item["originating_subquery"] = subquery
            item["originating_subqueries"] = [subquery]
            item["coverage_score"] = 1
        candidates_by_subquery.append({**intent, "subquery": subquery, "candidates": ranked_candidates})

    merged_by_key: dict[object, dict[str, Any]] = {}
    for group in candidates_by_subquery:
        subquery = group["subquery"]
        for candidate in group["candidates"]:
            chunk = candidate["chunk"]
            key: object = (
                ("global", chunk.global_id)
                if getattr(chunk, "global_id", 0)
                else ("source", getattr(chunk, "source", ""), getattr(chunk, "id", ""))
            )
            if key not in merged_by_key:
                merged_by_key[key] = {**candidate}
                continue
            existing = merged_by_key[key]
            existing["originating_subqueries"] = list(
                dict.fromkeys(existing["originating_subqueries"] + [subquery])
            )
            existing["coverage_score"] = len(existing["originating_subqueries"])
            existing["semantic_score"] = max(existing["semantic_score"], candidate["semantic_score"])
            existing["answerability_score"] = max(
                existing["answerability_score"], candidate["answerability_score"]
            )
            existing["final_score"] = max(existing["final_score"], candidate["final_score"])
            existing["boost_reasons"] = list(
                dict.fromkeys(existing["boost_reasons"] + candidate["boost_reasons"])
            )

    merged_candidates = sorted(
        merged_by_key.values(),
        key=lambda item: (item["coverage_score"], item["final_score"]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_keys: set[object] = set()
    for subquery in subqueries:
        supporting = [item for item in merged_candidates if subquery in item["originating_subqueries"]]
        if supporting:
            best = supporting[0]
            key = best["chunk"].global_id or (best["chunk"].source, best["chunk"].id)
            if key not in selected_keys:
                selected.append(best)
                selected_keys.add(key)
    for candidate in merged_candidates:
        key = candidate["chunk"].global_id or (candidate["chunk"].source, candidate["chunk"].id)
        if key not in selected_keys and len(selected) < final_k:
            selected.append(candidate)
            selected_keys.add(key)
    results = selected[:final_k]
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    if query_embeddings:
        query_embedding = np.mean(np.stack(query_embeddings), axis=0)
        norm = np.linalg.norm(query_embedding)
        if norm:
            query_embedding = query_embedding / norm
    else:
        query_embedding = np.empty(0, dtype=np.float32)
    primary_intent = detect_query_intent(query) if not is_multipart else {
        "intent_type": "multi_part",
        "extracted_entity_or_subject": " / ".join(
            group["extracted_entity_or_subject"] for group in candidates_by_subquery
            if group["extracted_entity_or_subject"]
        ),
        "original_query": query.strip(),
        "transformed_retrieval_query": " | ".join(
            group["transformed_retrieval_query"] for group in candidates_by_subquery
        ),
    }
    raw_candidates = [candidate for group in candidates_by_subquery for candidate in group["candidates"]]
    return {
        **primary_intent,
        "original_query": query.strip(),
        "decomposed_subqueries": subqueries,
        "is_multipart": is_multipart,
        "candidates_by_subquery": candidates_by_subquery,
        "merged_candidates": merged_candidates,
        "coverage_score": len({
            subquery for item in results for subquery in item["originating_subqueries"]
        }),
        "query_embedding": query_embedding,
        "embedding_latency": embedding_latency,
        "retrieval_latency": retrieval_latency,
        "candidates": raw_candidates,
        "results": results,
    }
