"""
conflict_detector.py
--------------------
NOVEL FEATURE #3: Inter-Chunk Conflict Detector

Most RAG evaluation tools only check if the LLM's response is grounded in the
retrieved context. But what if the retrieved chunks THEMSELVES contradict each
other? No popular open-source framework (RAGAS, TruLens, DeepEval) detects this.

The Problem:
  A vector DB might retrieve Chunk A saying "CEO is Alice" and Chunk B saying
  "CEO is Bob". The LLM is forced to pick one — or worse, it merges them into
  nonsense. If you only evaluate the RESPONSE, you'll blame the LLM for
  hallucinating. But the real fault was the retriever fetching contradictory info.

How this module works:
  1. Takes all retrieved context chunks.
  2. Computes pairwise semantic similarity between every pair of chunks.
  3. Identifies "suspiciously similar" pairs — chunks that talk about the SAME
     TOPIC (high keyword overlap) but say DIFFERENT things (lower semantic sim).
  4. For each suspicious pair, extracts a brief explanation of the conflict.

Key Insight:
  Two chunks with HIGH keyword overlap but MODERATE/LOW semantic similarity
  are contradiction candidates. They discuss the same subject but express
  different (possibly conflicting) information.

  Conversely, two chunks with LOW keyword overlap and LOW semantic similarity
  are just about different topics entirely — not a conflict.

Output:
  - conflict_pairs: list of chunk pairs flagged as potential contradictions
  - conflict_count: total number of conflicts detected
  - conflict_severity: NONE / LOW / MODERATE / HIGH
  - conflict_summary: human-readable summary
"""

from sentence_transformers import util
from embeddings import embed_text
import re
from collections import Counter


# --- Thresholds ---
# Keyword overlap threshold: if two chunks share >= 40% of significant words,
# they're about the same topic
KEYWORD_OVERLAP_THRESHOLD = 0.40

# If two chunks are about the same topic (high keyword overlap) but have
# semantic similarity BELOW this, they may be saying different things
SEMANTIC_CONFLICT_THRESHOLD = 0.75

# Minimum chunk text length to consider (skip near-empty chunks)
MIN_CHUNK_LENGTH = 20


def _extract_keywords(text: str) -> set:
    """
    Extracts significant words from text (lowercased, stopwords removed).
    Returns a set of keywords for overlap comparison.
    """
    # Common English stopwords to ignore
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

    # Tokenize: split on non-alphanumeric, lowercase, filter
    words = re.findall(r'[a-zA-Z]+', text.lower())
    keywords = {w for w in words if w not in stopwords and len(w) > 2}
    return keywords


def _keyword_overlap(keywords_a: set, keywords_b: set) -> float:
    """
    Computes Jaccard-like overlap between two keyword sets.
    Returns a float between 0 (no overlap) and 1 (identical keywords).
    """
    if not keywords_a or not keywords_b:
        return 0.0
    intersection = keywords_a & keywords_b
    union = keywords_a | keywords_b
    return len(intersection) / len(union)


def detect_conflicts(context_chunks: list) -> dict:
    """
    Scans all retrieved context chunks for potential contradictions.

    Args:
        context_chunks: List of chunk dicts from context.json (vector_data).
                        Each chunk should have 'id' and 'text' keys.

    Returns:
        A dict containing:
          - conflict_pairs: list of dicts, each describing a conflicting pair
          - conflict_count: number of conflicts found
          - conflict_severity: NONE / LOW / MODERATE / HIGH
          - conflict_summary: human-readable summary string
    """
    # --- Step 1: Filter valid chunks ---
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

    # --- Step 2: Pairwise comparison ---
    conflict_pairs = []

    for i in range(len(valid_chunks)):
        for j in range(i + 1, len(valid_chunks)):
            chunk_a = valid_chunks[i]
            chunk_b = valid_chunks[j]

            # 2a: Check keyword overlap (are they about the same topic?)
            kw_overlap = _keyword_overlap(chunk_a["keywords"], chunk_b["keywords"])

            if kw_overlap < KEYWORD_OVERLAP_THRESHOLD:
                # Different topics entirely — not a conflict, skip
                continue

            # 2b: Check semantic similarity (do they say the same thing?)
            sem_sim = util.cos_sim(
                chunk_a["embedding"], chunk_b["embedding"]
            ).item()

            if sem_sim >= SEMANTIC_CONFLICT_THRESHOLD:
                # High keyword overlap AND high semantic similarity
                # → They agree. Not a conflict.
                continue

            # 2c: If we reach here: same topic, different content → conflict!
            conflict_pairs.append({
                "chunk_a_id": chunk_a["id"],
                "chunk_b_id": chunk_b["id"],
                "keyword_overlap": round(kw_overlap, 3),
                "semantic_similarity": round(sem_sim, 3),
                "conflict_score": round(kw_overlap - sem_sim, 3),
                "chunk_a_preview": chunk_a["preview"],
                "chunk_b_preview": chunk_b["preview"],
                "explanation": (
                    f"Chunks {chunk_a['id']} and {chunk_b['id']} share "
                    f"{round(kw_overlap * 100)}% keyword overlap (same topic) "
                    f"but only {round(sem_sim * 100)}% semantic similarity — "
                    f"they may provide contradictory information."
                )
            })

    # --- Step 3: Sort by conflict_score (highest = most suspicious) ---
    conflict_pairs.sort(key=lambda x: x["conflict_score"], reverse=True)

    # --- Step 4: Determine severity ---
    count = len(conflict_pairs)
    if count == 0:
        severity = "NONE"
    elif count <= 2:
        severity = "LOW"
    elif count <= 5:
        severity = "MODERATE"
    else:
        severity = "HIGH"

    # --- Step 5: Build summary ---
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
