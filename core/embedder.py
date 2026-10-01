"""Embedding generator using sentence-transformers.

Automatically loads the fine-tuned model from models/finetuned-embedding/
if it exists, otherwise falls back to the default all-MiniLM-L6-v2.
"""

import os
from sentence_transformers import SentenceTransformer

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        # Prefer fine-tuned model if available
        finetuned = config.FINETUNED_EMBEDDING_MODEL
        if os.path.isdir(finetuned):
            _model = SentenceTransformer(finetuned)
        else:
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_embedding(text):
    """Generate embedding for a single text string."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_embeddings_batch(texts, batch_size=64):
    """Generate embeddings for a batch of texts."""
    model = _get_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
    return all_embeddings


def get_code_embedding(code, name="", chunk_type=""):
    """Generate embedding for code with contextual prefix."""
    prefix = f"{chunk_type}: {name}\n" if name else ""
    text = prefix + code[:2000]
    return get_embedding(text)


def get_query_embedding(query):
    """Generate embedding for a search query."""
    return get_embedding(f"search query: {query}")
