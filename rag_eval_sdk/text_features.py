"""Transparent text features and deterministic evidence interventions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:[.,:/-]\d+)*(?:%|[A-Za-z]+)?")
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d[\d,]*)(?:\.\d+)?%?(?!\w)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r";\s+|,\s+(?:but|whereas|while)\s+", re.IGNORECASE)
_VERB_HINT_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|"
    r"will|would|shall|should|may|might|must|[A-Za-z]+(?:ed|ing))\b",
    re.IGNORECASE,
)
_NEGATIONS = {
    "no", "not", "never", "neither", "nor", "without", "cannot", "can't",
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "hasn't", "haven't",
}
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his", "i",
    "in", "is", "it", "its", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "these", "they", "this", "those", "to",
    "was", "we", "were", "will", "with", "you", "your",
}


@dataclass(frozen=True)
class TextSpan:
    """A normalized verification unit mapped to its original text offsets."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("TextSpan offsets must satisfy 0 <= start < end")


def _claim_texts(text: str, *, granularity: str) -> list[str]:
    """Return claim texts while preserving the public v2 splitting policy."""

    sentences = _SENTENCE_BOUNDARY_RE.split((text or "").strip())
    candidates: list[str] = []
    for sentence in sentences:
        pieces = [piece.strip() for piece in _CLAUSE_BOUNDARY_RE.split(sentence)]
        can_split = (
            granularity == "clause"
            and len(pieces) > 1
            and all(len(tokens(piece)) >= 3 and _VERB_HINT_RE.search(piece) for piece in pieces)
        )
        candidates.extend(pieces if can_split else [sentence])
    claims: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split()).strip(" -•\t")
        if len(cleaned) >= 8 and _TOKEN_RE.search(cleaned):
            claims.append(cleaned)
    return claims


def _normalized_text_with_raw_offsets(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and retain a raw offset for every output character."""

    normalized: list[str] = []
    raw_offsets: list[int] = []
    whitespace_open = False
    whitespace_offset = 0
    for raw_index, character in enumerate(text):
        if character.isspace():
            if normalized and not whitespace_open:
                whitespace_open = True
                whitespace_offset = raw_index
            continue
        if whitespace_open:
            normalized.append(" ")
            raw_offsets.append(whitespace_offset)
            whitespace_open = False
        normalized.append(character)
        raw_offsets.append(raw_index)
    return "".join(normalized), raw_offsets


def split_claim_spans(
    text: str,
    max_claims: int | None = 24,
    *,
    granularity: str = "sentence",
) -> list[TextSpan]:
    """Split claims and map every unit back to exact offsets in ``text``.

    The verifier can normalize whitespace for stable scoring while consumers
    still receive faithful character offsets for highlighting and span-level
    evaluation. Mapping is deterministic and label-free.
    """

    if granularity not in {"sentence", "clause"}:
        raise ValueError("granularity must be 'sentence' or 'clause'")
    if max_claims is not None and max_claims < 0:
        raise ValueError("max_claims must be non-negative or None")

    raw_text = text or ""
    normalized, raw_offsets = _normalized_text_with_raw_offsets(raw_text)
    cursor = 0
    spans: list[TextSpan] = []
    for claim in _claim_texts(raw_text, granularity=granularity):
        normalized_start = normalized.find(claim, cursor)
        if normalized_start < 0:
            raise RuntimeError(
                "Claim splitter produced text that could not be mapped to the original response"
            )
        normalized_end = normalized_start + len(claim)
        raw_start = raw_offsets[normalized_start]
        raw_end = raw_offsets[normalized_end - 1] + 1
        spans.append(TextSpan(text=claim, start=raw_start, end=raw_end))
        cursor = normalized_end
        if max_claims is not None and len(spans) >= max_claims:
            break
    return spans


def split_claims(
    text: str,
    max_claims: int | None = 24,
    *,
    granularity: str = "sentence",
) -> list[str]:
    """Create deterministic claim units without an LLM call.

    These are sentence-level verification units, not guaranteed logical atoms.
    The distinction is kept explicit in reports.
    """

    return [
        span.text
        for span in split_claim_spans(
            text,
            max_claims=max_claims,
            granularity=granularity,
        )
    ]


def tokens(text: str, *, content_only: bool = False) -> list[str]:
    result = [match.group(0).casefold() for match in _TOKEN_RE.finditer(text or "")]
    if content_only:
        result = [token for token in result if token not in _STOPWORDS and len(token) > 2]
    return result


def lexical_coverage(claim: str, evidence: str) -> float:
    claim_tokens = set(tokens(claim, content_only=True))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(tokens(evidence, content_only=True))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def numbers(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).replace(",", "").casefold() for match in _NUMBER_RE.finditer(text or ""))


def named_tokens(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        value = match.group(0)
        if value[:1].isupper() and value.casefold() not in _STOPWORDS and len(value) > 2:
            values.append(value.casefold())
    return tuple(dict.fromkeys(values))


def critical_tokens(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*numbers(text), *named_tokens(text)]))


def critical_coverage(claim: str, evidence: str) -> float:
    critical = set(critical_tokens(claim))
    if not critical:
        return 1.0
    evidence_folded = evidence.casefold()
    return sum(1 for value in critical if value in evidence_folded) / len(critical)


def numeric_mismatch(claim: str, evidence: str) -> float:
    claim_numbers = set(numbers(claim))
    if not claim_numbers:
        return 0.0
    evidence_numbers = set(numbers(evidence))
    if not evidence_numbers or claim_numbers & evidence_numbers:
        return 0.0
    return 1.0


def negation_mismatch(claim: str, evidence: str) -> float:
    if lexical_coverage(claim, evidence) < 0.45:
        return 0.0
    claim_negated = any(token in _NEGATIONS for token in tokens(claim))
    evidence_negated = any(token in _NEGATIONS for token in tokens(evidence))
    return 1.0 if claim_negated != evidence_negated else 0.0


def infer_severity(text: str) -> str:
    """A transparent triage signal; applications may override it."""

    folded = text.casefold()
    critical_terms = (
        "dose", "dosage", "diagnosis", "legal", "deadline", "allergy", "fatal",
        "emergency", "contraindicated", "password", "credential", "wire transfer",
    )
    high_terms = (
        "must", "required", "prohibited", "guarantee", "percent", "million",
        "billion", "contract", "policy", "compliance", "tax",
    )
    if numbers(text) and any(term in folded for term in critical_terms):
        return "critical"
    if any(term in folded for term in critical_terms):
        return "high"
    if numbers(text) or any(term in folded for term in high_terms):
        return "medium"
    return "low"


@dataclass(frozen=True)
class Mutation:
    text: str
    changed: bool
    kind: str


def mutate_critical_fact(evidence: str, claim: str) -> Mutation:
    """Create a local counterfactual probe without using it as labelled data."""

    claim_numbers = set(numbers(claim))
    shared_number_matches = [
        match
        for match in _NUMBER_RE.finditer(evidence)
        if match.group(0).replace(",", "").casefold() in claim_numbers
    ]
    if shared_number_matches:
        match = shared_number_matches[0]
        original = match.group(0)
        try:
            numeric = float(original.replace(",", "").rstrip("%"))
            replacement = str(int(numeric + 1)) if numeric.is_integer() else str(numeric + 1.0)
            if original.endswith("%"):
                replacement += "%"
            changed = evidence[: match.start()] + replacement + evidence[match.end() :]
            return Mutation(changed, changed != evidence, "number_shift")
        except ValueError:
            pass

    shared_names = [value for value in named_tokens(claim) if value in evidence.casefold()]
    if shared_names:
        target = shared_names[0]
        changed = re.sub(rf"\b{re.escape(target)}\b", "[COUNTERFACTUAL_ENTITY]", evidence, count=1, flags=re.IGNORECASE)
        return Mutation(changed, changed != evidence, "entity_mask")

    evidence_tokens = tokens(evidence)
    has_negation = any(token in _NEGATIONS for token in evidence_tokens)
    if has_negation:
        changed = re.sub(r"\b(?:not|never|no)\b\s*", "", evidence, count=1, flags=re.IGNORECASE)
        return Mutation(changed, changed != evidence, "negation_flip")

    first_verb = re.search(r"\b(is|are|was|were|has|have|can|will|does|do)\b", evidence, flags=re.IGNORECASE)
    if first_verb:
        end = first_verb.end()
        changed = evidence[:end] + " not" + evidence[end:]
        return Mutation(changed, True, "negation_flip")

    return Mutation(evidence, False, "none")


def pack_evidence(texts: Sequence[str], max_characters: int) -> str:
    selected: list[str] = []
    remaining = max_characters
    for text in texts:
        if remaining <= 0:
            break
        cleaned = " ".join(str(text).split())
        if not cleaned:
            continue
        clipped = cleaned[:remaining]
        selected.append(clipped)
        remaining -= len(clipped) + 2
    return "\n\n".join(selected)
