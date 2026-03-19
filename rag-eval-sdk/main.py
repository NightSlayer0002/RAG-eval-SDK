"""
main.py — Entry point for the full RAG Evaluation Pipeline.

Run via CLI: rag-eval run
Or directly: python rag_eval_sdk/main.py

7-step pipeline:
  1. Load chat.json + context.json
  2. Scan for inter-chunk conflicts (Novel Feature #3)
  3. Call Gemini Flash for response
  4. Standard metrics (relevance, hallucination)
  5. Chunk Attribution Map (Novel Feature #1)
  6. Position Bias Detection (Novel Feature #4)
  7. Failure Mode Classification (Novel Feature #2) + save report
"""

import os

from .utils import load_json, save_json
from .llm import generate_llm_response
from .evaluators import score_relevance_completeness, score_hallucination
from .chunk_attributor import score_attribution_map
from .failure_classifier import classify_failure_mode
from .conflict_detector import detect_conflicts
from .position_bias_detector import detect_position_bias

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_evaluation_pipeline(
    chat_path: str = None,
    context_path: str = None,
    output_path: str = None
):
    if chat_path is None:
        chat_path = os.path.join(_PROJECT_ROOT, "data", "chat.json")
    if context_path is None:
        context_path = os.path.join(_PROJECT_ROOT, "data", "context.json")
    if output_path is None:
        output_path = os.path.join(_PROJECT_ROOT, "output", "evaluation_report.json")

    print("\n[1/7] Loading inputs...")
    chat_data = load_json(chat_path)
    context_data = load_json(context_path)

    user_query = chat_data["messages"][-1]["content"]
    context_chunks = context_data["data"]["vector_data"]
    retriever_scores = [
        v["score"]
        for v in context_data["data"]["sources"]["vectors_info"]
    ]

    print(f"     Query: \"{user_query}\"")
    print(f"     Context chunks loaded: {len(context_chunks)}")

    print("\n[2/7] Scanning for inter-chunk conflicts (Novel Feature #3)...")
    conflict_analysis = detect_conflicts(context_chunks)
    print(f"     {conflict_analysis['conflict_summary']}")

    print("\n[3/7] Calling Gemini 2.0 Flash...")
    llm_output = generate_llm_response(user_query, context_chunks)
    print(f"     Response received in {llm_output['latency_seconds']}s")
    print(f"     Response preview: \"{llm_output['response'][:120]}...\"")

    print("\n[4/7] Computing standard evaluation metrics...")
    relevance_score = score_relevance_completeness(llm_output["response"], context_chunks)
    hallucination_score = score_hallucination(llm_output["response"], context_chunks)
    print(f"     Relevance Score:     {relevance_score}")
    print(f"     Hallucination Score: {hallucination_score}")

    print("\n[5/7] Running Chunk Attribution Map (Novel Feature #1)...")
    attribution_map = score_attribution_map(llm_output["response"], context_chunks)
    print(f"     {attribution_map['attribution_summary']}")

    print("\n[6/7] Running Position Bias Detector (Novel Feature #4)...")
    position_bias = detect_position_bias(llm_output["response"], context_chunks, retriever_scores)
    print(f"     {position_bias['bias_summary']}")

    print("\n[7/7] Running Failure Mode Classifier (Novel Feature #2)...")
    failure_analysis = classify_failure_mode(
        relevance_score=relevance_score,
        hallucination_score=hallucination_score,
        retriever_scores=retriever_scores,
        attribution_map=attribution_map,
        conflict_analysis=conflict_analysis,
        position_bias=position_bias
    )
    print(f"     Failure Mode: {failure_analysis['failure_mode']}")
    print(f"     Confidence:   {failure_analysis['confidence']}")
    print(f"     Diagnosis:    {failure_analysis['diagnosis']}")

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
        },
        "chunk_attribution_map": attribution_map,
        "conflict_detection": conflict_analysis,
        "position_bias_analysis": position_bias,
        "failure_mode_analysis": failure_analysis,
        "performance": {
            "latency_seconds": llm_output["latency_seconds"],
            "estimated_cost_usd": llm_output["estimated_cost_usd"],
        }
    }

    save_json(report, output_path)

    print("\n" + "=" * 60)
    print("              EVALUATION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:         {llm_output['model']}")
    print(f"  Relevance:     {relevance_score}")
    print(f"  Hallucination: {hallucination_score}")
    print(f"  Conflicts:     {conflict_analysis['conflict_count']} [{conflict_analysis['conflict_severity']}]")
    print(f"  Position Bias: {'YES' if position_bias['bias_detected'] else 'NO'} [{position_bias['bias_type']}]")
    print(f"  Failure Mode:  {failure_analysis['failure_mode']} [{failure_analysis['confidence']} confidence]")
    print(f"  Latency:       {llm_output['latency_seconds']}s")
    print(f"  Report saved:  {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_evaluation_pipeline()
