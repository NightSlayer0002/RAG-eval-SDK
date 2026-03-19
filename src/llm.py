"""
llm.py
------
Handles all communication with Google Gemini Flash (the real LLM).

What is an LLM?
  A Large Language Model is an AI trained on massive amounts of text.
  It can read a prompt and generate a relevant, human-like response.
  We use it like a smart assistant: we give it a question + reference text,
  and it writes an answer.

What is Gemini Flash?
  Google's fast, cost-efficient LLM. "Flash" = speed-optimised variant.
  We use the "gemini-2.5-flash" model via Google's official Python SDK
  (google-genai — the latest recommended SDK as of 2025).

How to get your API key (FREE):
  1. Go to https://aistudio.google.com
  2. Sign in with your Google account
  3. Click "Get API Key" -> "Create API Key"
  4. Copy it into your .env file as: GEMINI_API_KEY=your_key_here

What does this file do?
  - Reads the API key from the environment / .env file
  - Builds a prompt that includes the retrieved context chunks
  - Calls Gemini and gets back a text response
  - Measures how long it took (latency) and estimates the cost
"""

import time
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load variables from your .env file (e.g., GEMINI_API_KEY=...)
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "\n[ERROR] GEMINI_API_KEY not found!\n"
        "Please create a .env file in the project root with:\n"
        "  GEMINI_API_KEY=your_key_here\n"
        "Get a FREE key at: https://aistudio.google.com"
    )

# Initialise the Gemini client once — reused for every call
_client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"


def generate_llm_response(user_query: str, context_chunks: list) -> dict:
    """
    Sends the user query + top context chunks to Gemini Flash.
    Instructs the model to answer ONLY from the provided context.

    Args:
        user_query:     The question asked by the user.
        context_chunks: Retrieved chunks from the vector database.

    Returns:
        A dict with:
          - response (str): The LLM's generated answer.
          - latency_seconds (float): Time taken for the API call.
          - estimated_cost_usd (float): Rough cost estimate in USD.
          - model (str): Model name used.
          - approx_input/output_tokens (int): Rough token counts.
    """
    start_time = time.time()

    # Take the top 5 most relevant chunks (already ranked in context.json)
    valid_chunks = [
        c for c in context_chunks
        if isinstance(c, dict) and c.get("text", "").strip()
    ][:5]

    # Build the context block for the prompt
    context_text = "\n\n".join(
        f"[Context {i + 1}]:\n{chunk['text']}"
        for i, chunk in enumerate(valid_chunks)
    )

    # System prompt: forces the model to stay grounded in the provided context
    prompt = f"""You are a helpful, accurate assistant. Answer the question below \
using ONLY the information provided in the context sections.
If the context does not contain enough information to answer fully, \
say so clearly rather than making things up.

=== CONTEXT ===
{context_text}

=== QUESTION ===
{user_query}

=== ANSWER ==="""

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,   # Low temperature = more factual, less creative
            max_output_tokens=512
        )
    )
    response_text = response.text.strip()

    latency = round(time.time() - start_time, 3)

    # --- Cost Estimation ---
    # Gemini 2.0 Flash pricing (as of early 2025):
    #   Input:  $0.075 per 1M tokens
    #   Output: $0.30  per 1M tokens
    # We approximate tokens as words (rough but good enough for a demo)
    input_tokens = len(prompt.split())
    output_tokens = len(response_text.split())
    cost_usd = round(
        (input_tokens / 1_000_000) * 0.075
        + (output_tokens / 1_000_000) * 0.30,
        8
    )

    return {
        "response": response_text,
        "latency_seconds": latency,
        "estimated_cost_usd": cost_usd,
        "model": MODEL_NAME,
        "approx_input_tokens": input_tokens,
        "approx_output_tokens": output_tokens
    }
