"""
embeddings.py
-------------
Loads the sentence-transformer model ONCE and exposes an embed_text() function.

Why do this in a separate file?
  - Both evaluators.py and chunk_attributor.py need to embed text.
  - If each file loaded its own model, we'd waste memory loading the same
    ~90MB model twice.
  - By importing from here, the model is loaded only once for the whole run.

Model: all-MiniLM-L6-v2
  - Small, fast, and accurate enough for semantic similarity tasks.
  - Converts a sentence into a 384-dimension vector (a list of 384 numbers)
    that captures its meaning.
"""

from sentence_transformers import SentenceTransformer

# Load model at import time — happens once when the program starts
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Converts a string into a semantic embedding tensor.
    convert_to_tensor=True makes it compatible with util.cos_sim().
    """
    return model.encode(text, convert_to_tensor=True)
