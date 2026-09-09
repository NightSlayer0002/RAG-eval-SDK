"""
conflict_detector.py
--------------------
NOVEL FEATURE #3: Inter-Chunk Conflict Detector (v2 — Negation-Aware)

Scans retrieved chunks for contradictions BEFORE the LLM generates a response.
No popular RAG evaluation framework (RAGAS, TruLens, DeepEval) does this.

Key Insight (3-signal approach):
  Signal 1: HIGH keyword overlap + LOW semantic similarity = different content
  Signal 2: HIGH keyword overlap + NEGATION pattern = opposing claims
  Either signal independently triggers a conflict flag.

Thresholds:
  KEYWORD_OVERLAP_THRESHOLD = 0.30 (Jaccard overlap)
  SEMANTIC_CONFLICT_THRESHOLD = 0.72 (anything below = possible conflict)
"""

from .embeddings import cosine_similarity, embed_text
import re

KEYWORD_OVERLAP_THRESHOLD = 0.30
SEMANTIC_CONFLICT_THRESHOLD = 0.72
MIN_CHUNK_LENGTH = 20
MIN_CONFLICT_SCORE = 0.05


def _extract_keywords(text: str) -> set:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "up", "it", "its", "this", "that",
        "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
        "him", "his", "she", "her", "they", "them", "their", "what", "which",
        "who", "whom"
    }
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return {w for w in words if w not in stopwords and len(w) > 2}


# --- Negation patterns for conflict detection ---
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "without", "hardly", "barely", "rarely", "seldom"
}
NEGATION_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't",
    "hadn't", "can't", "cannot", "mustn't", "needn't"
}


def _detect_negation_conflict(text_a: str, text_b: str, keywords_a: set, keywords_b: set) -> dict:
    """Detects if two chunks express opposing claims via negation."""
    shared_keywords = keywords_a & keywords_b
    if len(shared_keywords) < 2:
        return {"has_negation_conflict": False, "negated_terms": [], "confidence": 0.0}

    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    def find_negation_zones(text):
        words = re.findall(r"[a-zA-Z']+", text)
        negated_words = set()
        for i, word in enumerate(words):
            if word in NEGATION_WORDS or word in NEGATION_CONTRACTIONS:
                for j in range(i + 1, min(i + 4, len(words))):
                    negated_words.add(words[j].lower())
        return negated_words

    negated_in_a = find_negation_zones(text_a_lower)
    negated_in_b = find_negation_zones(text_b_lower)

    conflicting_terms = []
    for keyword in shared_keywords:
        if (keyword in negated_in_a) != (keyword in negated_in_b):
            conflicting_terms.append(keyword)

    if conflicting_terms:
        confidence = min(1.0, len(conflicting_terms) / 3)
        return {
            "has_negation_conflict": True,
            "negated_terms": conflicting_terms,
            "confidence": round(confidence, 2)
        }

    return {"has_negation_conflict": False, "negated_terms": [], "confidence": 0.0}


def _keyword_overlap(keywords_a: set, keywords_b: set) -> float:
    if not keywords_a or not keywords_b:
        return 0.0
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union)


def detect_conflicts(context_chunks: list) -> dict:
    """
    Scans all retrieved context chunks for potential contradictions.

    Returns:
        conflict_pairs, conflict_count, conflict_severity, conflict_summary
    """
    valid_chunks = []
    for chunk in context_chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "")
        if len(text.strip()) < MIN_CHUNK_LENGTH:
            continue
        valid_chunks.append({
            "id": chunk.get("id", -1),
            "text": text,
            "keywords": _extract_keywords(text),
            "embedding": embed_text(text),
            "preview": text[:100].replace("\n", " ").strip() + "..."
        })

    conflict_pairs = []
    seen_pairs = set()

    for i in range(len(valid_chunks)):
        for j in range(i + 1, len(valid_chunks)):
            chunk_a = valid_chunks[i]
            chunk_b = valid_chunks[j]
            pair_key = (chunk_a["id"], chunk_b["id"])
            if pair_key in seen_pairs:
                continue

            sem_sim = cosine_similarity(
                chunk_a["embedding"], chunk_b["embedding"]
            )
            kw_overlap = _keyword_overlap(chunk_a["keywords"], chunk_b["keywords"])

            # 2c: Check negation patterns (NEW)
            negation_result = _detect_negation_conflict(
                chunk_a["text"], chunk_b["text"],
                chunk_a["keywords"], chunk_b["keywords"]
            )

            classic_conflict = sem_sim < SEMANTIC_CONFLICT_THRESHOLD
            negation_conflict = negation_result["has_negation_conflict"]

            if not classic_conflict and not negation_conflict:
                continue

            # Determine detection method
            if negation_conflict and not classic_conflict:
                detection_method = "negation"
                conflict_score = round(kw_overlap * negation_result["confidence"], 3)
                explanation = (
                    f"Chunks {chunk_a['id']} and {chunk_b['id']} share "
                    f"{round(kw_overlap * 100)}% keyword overlap and have "
                    f"{round(sem_sim * 100)}% semantic similarity (appears safe), "
                    f"BUT negation detected on: {', '.join(negation_result['negated_terms'])}."
                )
            elif classic_conflict and negation_conflict:
                detection_method = "both"
                conflict_score = round(kw_overlap - sem_sim + negation_result["confidence"] * 0.2, 3)
                explanation = (
                    f"Chunks {chunk_a['id']} and {chunk_b['id']}: same topic, "
                    f"low similarity ({round(sem_sim * 100)}%), AND negation on: "
                    f"{', '.join(negation_result['negated_terms'])}. STRONG conflict."
                )
            else:
                detection_method = "keyword+cosine"
                conflict_score = round(kw_overlap - sem_sim, 3)
                explanation = (
                    f"Chunks {chunk_a['id']} and {chunk_b['id']} share "
                    f"{round(kw_overlap * 100)}% keyword overlap but only "
                    f"{round(sem_sim * 100)}% semantic similarity -- conflict."
                )

            if conflict_score < MIN_CONFLICT_SCORE:
                continue

            seen_pairs.add(pair_key)
            conflict_pairs.append({
                "chunk_a_id": chunk_a["id"],
                "chunk_b_id": chunk_b["id"],
                "keyword_overlap": round(kw_overlap, 3),
                "semantic_similarity": round(sem_sim, 3),
                "conflict_score": conflict_score,
                "detection_method": detection_method,
                "negation_terms": negation_result.get("negated_terms", []),
                "chunk_a_preview": chunk_a["preview"],
                "chunk_b_preview": chunk_b["preview"],
                "explanation": explanation
            })

    conflict_pairs.sort(key=lambda x: x["conflict_score"], reverse=True)

    count = len(conflict_pairs)
    if count == 0:
        severity = "NONE"
    elif count <= 2:
        severity = "LOW"
    elif count <= 5:
        severity = "MODERATE"
    else:
        severity = "HIGH"

    if count == 0:
        summary = (
            "No inter-chunk conflicts detected. All retrieved chunks are "
            "consistent with each other."
        )
    else:
        worst = conflict_pairs[0]
        summary = (
            f"{count} potential conflict(s) detected between retrieved chunks. "
            f"Severity: {severity}. "
            f"Worst conflict: Chunk {worst['chunk_a_id']} vs Chunk "
            f"{worst['chunk_b_id']} (keyword overlap: {worst['keyword_overlap']}, "
            f"semantic similarity: {worst['semantic_similarity']})."
        )

    return {
        "conflict_pairs": conflict_pairs,
        "conflict_count": count,
        "conflict_severity": severity,
        "conflict_summary": summary
    }
