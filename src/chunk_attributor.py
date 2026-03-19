"""
chunk_attributor.py
-------------------
NOVEL FEATURE #1: Chunk Attribution Map

Most RAG evaluation tools give you one aggregate score for the whole response.
This module goes deeper — it scores the LLM's response against EACH retrieved
context chunk individually, showing you exactly which chunks the response is
grounded in and which were silently ignored.

Key concepts:
  - Attribution Threshold: a similarity score >= 0.35 means the response
    is "grounded in" that chunk.
  - Phantom Chunk: a chunk that was retrieved by the vector DB but contributed
    nothing to the response (similarity < threshold).
  - Coverage Gap: 1 - best_chunk_similarity. Measures how "ungrounded" the
    overall response is. High gap = likely hallucination.
"""

from sentence_transformers import util
from embeddings import embed_text

# If a chunk's similarity to the response is above this, it's "attributed"
ATTRIBUTION_THRESHOLD = 0.35


def score_attribution_map(response: str, context_chunks: list) -> dict:
    """
    Scores the response against each retrieved chunk individually.

    Args:
        response: The LLM's generated answer.
        context_chunks: List of chunk dicts from context.json (vector_data).

    Returns:
        A dict containing:
          - attributed_chunks: chunks the response is grounded in (ranked)
          - phantom_chunk_ids: chunk IDs that were retrieved but not used
          - coverage_gap: how ungrounded the response is (0=grounded, 1=not at all)
          - attribution_summary: a human-readable summary string
    """
    response_embedding = embed_text(response)

    chunk_scores = []
    for chunk in context_chunks:
        if not isinstance(chunk, dict):
            continue
        text = chunk.get("text", "")
        if not text:
            continue

        chunk_emb = embed_text(text)
        sim = util.cos_sim(response_embedding, chunk_emb).item()

        chunk_scores.append({
            "chunk_id": chunk.get("id", -1),
            "similarity": round(sim, 3),
            # Show only the first 100 chars as a preview
            "preview": text[:100].replace("\n", " ").strip() + "..."
        })

    # Rank chunks from most to least attributed
    chunk_scores.sort(key=lambda x: x["similarity"], reverse=True)

    attributed = []
    phantom_ids = []

    for rank, cs in enumerate(chunk_scores, start=1):
        is_attributed = cs["similarity"] >= ATTRIBUTION_THRESHOLD
        if is_attributed:
            attributed.append({
                "chunk_id": cs["chunk_id"],
                "similarity": cs["similarity"],
                "rank": rank,
                "preview": cs["preview"]
            })
        else:
            phantom_ids.append(cs["chunk_id"])

    # Coverage gap: how far the BEST chunk is from fully grounding the response
    best_similarity = chunk_scores[0]["similarity"] if chunk_scores else 0.0
    coverage_gap = round(1.0 - best_similarity, 3)

    total = len(chunk_scores)
    attr_count = len(attributed)
    phantom_count = len(phantom_ids)

    if coverage_gap > 0.5:
        gap_label = "HIGH — response is likely hallucinated"
    elif coverage_gap > 0.3:
        gap_label = "MODERATE — response is partially grounded"
    else:
        gap_label = "LOW — response is well grounded"

    summary = (
        f"Response grounded in {attr_count}/{total} chunks. "
        f"{phantom_count} phantom chunks retrieved but unused. "
        f"Coverage gap: {coverage_gap} ({gap_label})."
    )

    return {
        "attributed_chunks": attributed,
        "phantom_chunk_ids": phantom_ids,
        "total_chunks_retrieved": total,
        "attributed_count": attr_count,
        "phantom_count": phantom_count,
        "coverage_gap": coverage_gap,
        "attribution_summary": summary
    }
