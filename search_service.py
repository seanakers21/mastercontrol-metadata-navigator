from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from langchain_core.documents import Document

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
STOP_WORDS = {
    "a", "an", "and", "any", "are", "for", "had", "has", "have", "in",
    "is", "it", "of", "on", "or", "past", "the", "to", "what", "with",
}
SYNONYMS = {
    "dry": {"dry-run", "dry-running", "starved", "starvation", "no-flow"},
    "pump": {"pumps", "centrifugal", "transfer-pump"},
    "broke": {"break", "breaking", "broken", "failed", "failure", "damage"},
    "issues": {"issue", "event", "deviation", "variance", "incident"},
    "leak": {"leaking", "seal-leak", "seal-failure"},
}


def tokenize(value: str) -> List[str]:
    return [token for token in TOKEN_PATTERN.findall(value.lower()) if token not in STOP_WORDS]


def expanded_terms(query: str) -> set[str]:
    terms = set(tokenize(query))
    for term in list(terms):
        terms.update(SYNONYMS.get(term, set()))
    return terms


def lexical_score(query: str, text: str) -> float:
    terms = expanded_terms(query)
    if not terms:
        return 0.0
    counts = Counter(tokenize(text))
    matched = sum(1.0 + min(counts[t], 3) * 0.2 for t in terms if counts[t])
    phrase_bonus = 1.5 if query.strip().lower() in text.lower() else 0.0
    return (matched + phrase_bonus) / len(terms)


def event_searchable_text(item: Dict[str, Any]) -> str:
    keys = [
        "record_id", "event_type", "title", "site", "area", "department",
        "equipment_id", "equipment_type", "manufacturer", "problem_statement",
        "failure_mode", "immediate_action", "root_cause_category",
        "confirmed_root_cause", "corrective_action", "preventive_action",
        "recurrence_classification", "keywords", "related_records",
    ]
    values = []
    for key in keys:
        value = item.get(key, "")
        values.append(", ".join(value) if isinstance(value, list) else str(value or ""))
    return "\n".join(values)


def apply_event_filters(items: Sequence[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    def allowed(item: Dict[str, Any]) -> bool:
        for key in ("site", "event_type", "department", "status"):
            selected = filters.get(key)
            if selected and item.get(key) not in selected:
                return False
        year_range = filters.get("year_range")
        if year_range:
            year = int(str(item.get("event_date", "0000"))[:4])
            if not year_range[0] <= year <= year_range[1]:
                return False
        return True
    return [item for item in items if allowed(item)]


def rank_events(query: str, items: Sequence[Dict[str, Any]], limit: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    ranked = [(item, lexical_score(query, event_searchable_text(item))) for item in items]
    ranked = [row for row in ranked if row[1] > 0]
    return sorted(ranked, key=lambda row: (row[1], row[0].get("event_date", "")), reverse=True)[:limit]


def combine_document_results(query: str, semantic_results: Sequence[Tuple[Document, float, float]]) -> List[Tuple[Document, float]]:
    combined = [
        (doc, float(rerank_score) + lexical_score(query, doc.page_content) * 2.0)
        for doc, _, rerank_score in semantic_results
    ]
    return sorted(combined, key=lambda row: row[1], reverse=True)
