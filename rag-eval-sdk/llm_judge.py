"""
llm_judge.py — LLM-as-Judge Hallucination Verifier (Novel Feature #5)

Uses Gemini Flash to verify whether each sentence in the LLM response is
supported by the retrieved context chunks. This cross-validates the
embedding-based hallucination score with an independent LLM opinion.

Why this matters (2026 market):
  - Embedding similarity catches surface-level divergence but misses
    subtle factual errors (e.g., wrong numbers, swapped names).
  - LLM-as-judge can reason about entailment and contradiction at the
    semantic level, catching things cosine similarity cannot.
  - The combination of both signals is the gold standard in 2026 eval.

Output:
  - Per-sentence verdicts: SUPPORTED / NOT_SUPPORTED / PARTIAL
  - Overall faithfulness_score (0.0 to 1.0)
  - Detailed explanations for each verdict
"""

import os
import re
import json
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_api_key = os.environ.get("GEMINI_API_KEY")
if _api_key:
    _client = genai.Client(api_key=_api_key)
else:
    _client = None

MODEL_NAME = "gemini-2.5-flash"

_JUDGE_PROMPT = """You are an expert fact-checker evaluating whether an AI assistant's response is faithful to the provided context.

=== CONTEXT (retrieved chunks) ===
{context}

=== AI RESPONSE ===
{response}

=== TASK ===
Break the AI response into individual claims/sentences. For EACH claim, determine:
- SUPPORTED: The claim is directly supported by information in the context.
- NOT_SUPPORTED: The claim contains information NOT found anywhere in the context (hallucinated).
- PARTIAL: Part of the claim is supported but part goes beyond the context.

Respond in this EXACT JSON format (no markdown, no code fences):
{{
  "sentences": [
    {{
      "text": "the sentence",
      "verdict": "SUPPORTED or NOT_SUPPORTED or PARTIAL",
      "explanation": "brief reason"
    }}
  ],
  "overall_faithfulness": 0.85,
  "overall_verdict": "MOSTLY_FAITHFUL or PARTIALLY_FAITHFUL or UNFAITHFUL",
  "reasoning": "brief overall assessment"
}}

Be strict. If a claim includes any specific detail (names, numbers, dates) not in the context, mark it NOT_SUPPORTED."""


def _split_sentences(text: str) -> list:
    """Split text into sentences for analysis."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_json(raw_text: str) -> dict:
    """
    Robustly extract JSON from LLM output that may contain thinking blocks,
    markdown fences, or other wrapper text.
    """
    text = raw_text.strip()

    # Strip thinking blocks (<think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex fallback: find first { ... } block (greedy)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def judge_faithfulness(response: str, context_chunks: list) -> dict:
    """
    Use Gemini as an independent judge to verify response faithfulness.

    Args:
        response:       The LLM-generated answer to evaluate.
        context_chunks: List of chunk dicts with 'text' field.

    Returns:
        Dict with per-sentence verdicts, overall faithfulness score,
        and detailed reasoning from the LLM judge.
    """
    if not _client:
        return {
            "error": "GEMINI_API_KEY not set — LLM judge unavailable",
            "sentences": [],
            "overall_faithfulness": None,
            "overall_verdict": "UNAVAILABLE",
            "reasoning": "API key not configured.",
            "latency_seconds": 0
        }

    # Build context string from chunks
    valid_chunks = [
        c for c in context_chunks
        if isinstance(c, dict) and c.get("text", "").strip()
    ]
    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {chunk['text'][:500]}"
        for i, chunk in enumerate(valid_chunks[:15])  # limit to 15 chunks
    )

    prompt = _JUDGE_PROMPT.format(
        context=context_text,
        response=response
    )

    start_time = time.time()
    try:
        # Retry logic for rate limits
        raw_text = None
        for attempt in range(3):
            try:
                result = _client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1024,
                        response_mime_type="application/json"
                    )
                )
                raw_text = result.text.strip()
                break
            except Exception as retry_err:
                if "429" in str(retry_err) or "RESOURCE_EXHAUSTED" in str(retry_err):
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise

        if raw_text is None:
            raise Exception("All retries exhausted for LLM judge")
        latency = round(time.time() - start_time, 3)

        parsed = _extract_json(raw_text)

        sentences = parsed.get("sentences", [])
        overall_faithfulness = parsed.get("overall_faithfulness", 0.0)
        overall_verdict = parsed.get("overall_verdict", "UNKNOWN")
        reasoning = parsed.get("reasoning", "")

        # Compute our own faithfulness score as a cross-check
        if sentences:
            supported_count = sum(
                1 for s in sentences
                if s.get("verdict") == "SUPPORTED"
            )
            partial_count = sum(
                1 for s in sentences
                if s.get("verdict") == "PARTIAL"
            )
            computed_score = round(
                (supported_count + 0.5 * partial_count) / len(sentences), 3
            )
        else:
            computed_score = overall_faithfulness

        return {
            "sentences": sentences,
            "sentence_count": len(sentences),
            "supported_count": sum(
                1 for s in sentences if s.get("verdict") == "SUPPORTED"
            ),
            "not_supported_count": sum(
                1 for s in sentences if s.get("verdict") == "NOT_SUPPORTED"
            ),
            "partial_count": sum(
                1 for s in sentences if s.get("verdict") == "PARTIAL"
            ),
            "overall_faithfulness": computed_score,
            "llm_reported_faithfulness": overall_faithfulness,
            "overall_verdict": overall_verdict,
            "reasoning": reasoning,
            "latency_seconds": latency,
            "judge_model": MODEL_NAME
        }

    except json.JSONDecodeError as e:
        latency = round(time.time() - start_time, 3)
        return {
            "error": f"Failed to parse judge response: {e}",
            "raw_response": raw_text[:500] if 'raw_text' in dir() else "",
            "sentences": [],
            "overall_faithfulness": None,
            "overall_verdict": "PARSE_ERROR",
            "reasoning": "LLM judge returned unparseable output.",
            "latency_seconds": latency,
            "judge_model": MODEL_NAME
        }
    except Exception as e:
        latency = round(time.time() - start_time, 3)
        return {
            "error": str(e),
            "sentences": [],
            "overall_faithfulness": None,
            "overall_verdict": "ERROR",
            "reasoning": f"LLM judge failed: {e}",
            "latency_seconds": latency,
            "judge_model": MODEL_NAME
        }
