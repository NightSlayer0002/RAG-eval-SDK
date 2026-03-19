"""
conflict_detector.py
--------------------
NOVEL FEATURE #3: Inter-Chunk Conflict Detector (Preventive)

Scans retrieved chunks for contradictions BEFORE the LLM generates a response.
No popular RAG evaluation framework (RAGAS, TruLens, DeepEval) does this.

Key Insight:
  Two chunks with HIGH keyword overlap but LOW semantic similarity are talking
  about the SAME topic but saying DIFFERENT things -> conflict!

Thresholds:
  KEYWORD_OVERLAP_THRESHOLD = 0.30 (Jaccard overlap)
  SEMANTIC_CONFLICT_THRESHOLD = 0.72 (anything below = possible conflict)
"""

from sentence_transformers import util
from .embeddings import embed_text
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

            sem_sim = util.cos_sim(
                chunk_a["embedding"], chunk_b["embedding"]
            ).item()
            kw_overlap = _keyword_overlap(chunk_a["keywords"], chunk_b["keywords"])

            # Single-signal detection (keyword overlap + semantic divergence):
            # Two chunks that talk about the SAME topic (high keyword overlap)
            # but say DIFFERENT things (low semantic similarity) = conflict.
            is_conflict = (kw_overlap >= KEYWORD_OVERLAP_THRESHOLD
                           and sem_sim < SEMANTIC_CONFLICT_THRESHOLD)

            if not is_conflict:
                continue

            seen_pairs.add(pair_key)
            conflict_score = round(kw_overlap * (1.0 - sem_sim), 3)

            # Filter out borderline micro-conflicts
            if conflict_score < MIN_CONFLICT_SCORE:
                continue

            conflict_pairs.append({
                "chunk_a_id": chunk_a["id"],
                "chunk_b_id": chunk_b["id"],
                "keyword_overlap": round(kw_overlap, 3),
                "semantic_similarity": round(sem_sim, 3),
                "conflict_score": conflict_score,
                "chunk_a_preview": chunk_a["preview"],
                "chunk_b_preview": chunk_b["preview"],
                "explanation": (
                    f"Chunks {chunk_a['id']} and {chunk_b['id']} share "
                    f"{round(kw_overlap * 100)}% keyword overlap (same topic) "
                    f"but only {round(sem_sim * 100)}% semantic similarity -- "
                    f"they may provide contradictory information."
                )
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
