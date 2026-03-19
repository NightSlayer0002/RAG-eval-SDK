from sentence_transformers import SentenceTransformer

# Load model at import time — happens once when the program starts
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Converts a string into a semantic embedding tensor.
    convert_to_tensor=True makes it compatible with util.cos_sim().
    """
    return model.encode(text, convert_to_tensor=True)
