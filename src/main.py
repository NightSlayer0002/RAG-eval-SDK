"""
main.py
-------
Entry point for the LLM RAG Evaluation Pipeline.

Run this file to execute the full evaluation:
  cd src
  python main.py

What this pipeline does, step by step:
  1. Load the user's chat message (data/chat.json)
  2. Load the retrieved context chunks (data/context.json)
     - These came from a real vector database (BeyondChats RAG system)
  3. Scan retrieved chunks for inter-chunk conflicts (Novel Feature #3)
  4. Call Google Gemini Flash with the query + context → get a real LLM response
  5. Evaluate the response with 6 components:
       a) Relevance Score              — how aligned is the response with the context?
       b) Hallucination Score          — how much did the LLM stray from the context?
       c) Chunk Attribution Map        — which specific chunks did the response use?
       d) Position Bias Detection      — did the LLM ignore chunks due to position? (Novel Feature #4)
       e) Failure Mode Analysis        — was the failure the retriever's or LLM's fault?
  6. Save a full JSON report to output/evaluation_report.json
  7. Print a summary to the terminal
"""

from utils import load_json, save_json
from llm import generate_llm_response
from evaluators import score_relevance_completeness, score_hallucination
from chunk_attributor import score_attribution_map
from failure_classifier import classify_failure_mode
from conflict_detector import detect_conflicts
from position_bias_detector import detect_position_bias


CHAT_PATH = "../data/chat.json"
CONTEXT_PATH = "../data/context.json"
OUTPUT_PATH = "../output/evaluation_report.json"


def run_evaluation_pipeline():
    # ------------------------------------------------------------------ #
    # STEP 1: Load inputs
    # ------------------------------------------------------------------ #
    print("\n[1/7] Loading inputs...")
    chat_data = load_json(CHAT_PATH)
    context_data = load_json(CONTEXT_PATH)

    user_query = chat_data["messages"][-1]["content"]
    context_chunks = context_data["data"]["vector_data"]

    # Extract retriever's own confidence scores (from the vector DB search)
    retriever_scores = [
        v["score"]
        for v in context_data["data"]["sources"]["vectors_info"]
    ]

    print(f"     Query: \"{user_query}\"")
    print(f"     Context chunks loaded: {len(context_chunks)}")

    # ------------------------------------------------------------------ #
    # STEP 2: Novel Feature #3 — Inter-Chunk Conflict Detector
    # ------------------------------------------------------------------ #
    print("\n[2/7] Scanning for inter-chunk conflicts (Novel Feature #3)...")
    conflict_analysis = detect_conflicts(context_chunks)
    print(f"     {conflict_analysis['conflict_summary']}")

    # ------------------------------------------------------------------ #
    # STEP 3: Generate real LLM response via Gemini Flash
    # ------------------------------------------------------------------ #
    print("\n[3/7] Calling Gemini 2.0 Flash...")
    llm_output = generate_llm_response(user_query, context_chunks)
    print(f"     Response received in {llm_output['latency_seconds']}s")
    print(f"     Response preview: \"{llm_output['response'][:120]}...\"")

    # ------------------------------------------------------------------ #
    # STEP 4: Standard evaluation metrics
    # ------------------------------------------------------------------ #
    print("\n[4/7] Computing standard evaluation metrics...")
    relevance_score = score_relevance_completeness(
        llm_output["response"], context_chunks
    )
    hallucination_score = score_hallucination(
        llm_output["response"], context_chunks
    )
    print(f"     Relevance Score:     {relevance_score}")
    print(f"     Hallucination Score: {hallucination_score}")

    # ------------------------------------------------------------------ #
    # STEP 5: Novel Feature #1 — Chunk Attribution Map
    # ------------------------------------------------------------------ #
    print("\n[5/7] Running Chunk Attribution Map (Novel Feature #1)...")
    attribution_map = score_attribution_map(llm_output["response"], context_chunks)
    print(f"     {attribution_map['attribution_summary']}")

    # ------------------------------------------------------------------ #
    # STEP 6: Novel Feature #4 — Position Bias Detector
    # ------------------------------------------------------------------ #
    print("\n[6/7] Running Position Bias Detector (Novel Feature #4)...")
    position_bias = detect_position_bias(
        llm_output["response"], context_chunks, retriever_scores
    )
    print(f"     {position_bias['bias_summary']}")

    # ------------------------------------------------------------------ #
    # STEP 7: Novel Feature #2 — Failure Mode Classifier
    # ------------------------------------------------------------------ #
    print("\n[7/7] Running Failure Mode Classifier (Novel Feature #2)...")
    failure_analysis = classify_failure_mode(
        relevance_score=relevance_score,
        hallucination_score=hallucination_score,
        retriever_scores=retriever_scores,
        attribution_map=attribution_map,
        conflict_analysis=conflict_analysis,
        position_bias=position_bias
    )
    print(f"     Failure Mode:  {failure_analysis['failure_mode']}")
    print(f"     Confidence:    {failure_analysis['confidence']}")
    print(f"     Diagnosis:     {failure_analysis['diagnosis']}")

    # ------------------------------------------------------------------ #
    # Build and save the full evaluation report
    # ------------------------------------------------------------------ #
    report = {
        "meta": {
            "model": llm_output["model"],
            "query": user_query,
            "chunks_retrieved": len(context_chunks)
        },
        "llm_response": llm_output["response"],
        "standard_evaluation": {
            "relevance_score": relevance_score,
            "hallucination_score": hallucination_score,
            "interpretation": {
                "relevance": (
                    "GOOD (>0.6)" if relevance_score > 0.6 else
                    "MODERATE (0.4-0.6)" if relevance_score > 0.4 else
                    "POOR (<0.4)"
                ),
                "hallucination": (
                    "LOW — well grounded (<0.4)" if hallucination_score < 0.4 else
                    "MODERATE (0.4-0.6)" if hallucination_score < 0.6 else
                    "HIGH — likely hallucinated (>0.6)"
                )
            }
        },
        "chunk_attribution_map": attribution_map,
        "conflict_detection": conflict_analysis,
        "position_bias_analysis": position_bias,
        "failure_mode_analysis": failure_analysis,
        "performance": {
            "latency_seconds": llm_output["latency_seconds"],
            "estimated_cost_usd": llm_output["estimated_cost_usd"],
            "approx_input_tokens": llm_output["approx_input_tokens"],
            "approx_output_tokens": llm_output["approx_output_tokens"]
        }
    }

    save_json(report, OUTPUT_PATH)

    # ------------------------------------------------------------------ #
    # Final summary printout
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("              EVALUATION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:             {llm_output['model']}")
    print(f"  Relevance:         {relevance_score}")
    print(f"  Hallucination:     {hallucination_score}")
    print(f"  Attribution:       {attribution_map['attributed_count']}/{attribution_map['total_chunks_retrieved']} chunks used")
    print(f"  Coverage Gap:      {attribution_map['coverage_gap']}")
    print(f"  Conflicts:         {conflict_analysis['conflict_count']} detected  [{conflict_analysis['conflict_severity']}]")
    print(f"  Position Bias:     {'YES' if position_bias['bias_detected'] else 'NO'}  [{position_bias['bias_type']}]")
    print(f"  Failure Mode:      {failure_analysis['failure_mode']}  [{failure_analysis['confidence']} confidence]")
    print(f"  Latency:           {llm_output['latency_seconds']}s")
    print(f"  Est. Cost:         ${llm_output['estimated_cost_usd']}")
    print("=" * 60)
    print(f"\n  Full report saved → {OUTPUT_PATH}\n")


if __name__ == "__main__":
    run_evaluation_pipeline()
