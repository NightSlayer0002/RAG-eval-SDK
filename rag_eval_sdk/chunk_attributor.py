"""
chunk_attributor.py
-------------------
NOVEL FEATURE #1: Chunk Attribution Map

Scores the LLM's response against EACH retrieved chunk individually,
showing exactly which chunks grounded the response and which were phantom.

- Attribution Threshold: similarity >= 0.35 means the chunk was used.
- Phantom Chunk: retrieved but contributed nothing to the response.
- Coverage Gap: 1 - best_chunk_similarity (how ungrounded the response is).
"""

from .embeddings import cosine_similarity, embed_text

ATTRIBUTION_THRESHOLD = 0.35


def score_attribution_map(response: str, context_chunks: list) -> dict:
    """
    Scores the response against each retrieved chunk individually.

    Returns:
        attributed_chunks, phantom_chunk_ids, coverage_gap, attribution_summary
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
        sim = cosine_similarity(response_embedding, chunk_emb)

        chunk_scores.append({
            "chunk_id": chunk.get("id", -1),
            "similarity": round(sim, 3),
            "preview": text[:100].replace("\n", " ").strip() + "..."
        })

    chunk_scores.sort(key=lambda x: x["similarity"], reverse=True)

    attributed = []
    phantom_ids = []

    for rank, cs in enumerate(chunk_scores, start=1):
        if cs["similarity"] >= ATTRIBUTION_THRESHOLD:
            attributed.append({
                "chunk_id": cs["chunk_id"],
                "similarity": cs["similarity"],
                "rank": rank,
                "preview": cs["preview"]
            })
        else:
            phantom_ids.append(cs["chunk_id"])

    best_similarity = chunk_scores[0]["similarity"] if chunk_scores else 0.0
    coverage_gap = round(1.0 - best_similarity, 3)

    total = len(chunk_scores)
    attr_count = len(attributed)
    phantom_count = len(phantom_ids)

    if coverage_gap > 0.5:
        gap_label = "HIGH -- response is likely hallucinated"
    elif coverage_gap > 0.3:
        gap_label = "MODERATE -- response is partially grounded"
    else:
        gap_label = "LOW -- response is well grounded"

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
