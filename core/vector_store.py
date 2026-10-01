"""Vector store abstraction — ChromaDB (default) or numpy fallback.

Selectable via config.VECTOR_STORE_BACKEND ("chromadb" | "numpy").
Both backends expose the same public API.
"""

import json
import os
import uuid
import numpy as np
from threading import Lock

import config

_store = None
_lock = Lock()


# ═══════════════════════════════════════════════════════════════════════════
# CHROMADB BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class ChromaDBStore:
    """ChromaDB-backed vector store with cosine similarity search."""

    def __init__(self, persist_dir):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="codesage_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, metadatas, documents):
        # ChromaDB requires string metadata values
        clean_metas = []
        for m in metadatas:
            clean_metas.append({k: str(v) for k, v in m.items()})

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=clean_metas,
            documents=documents,
        )

    def search(self, query_embedding, top_k=20, where=None):
        # ChromaDB where clause requires string values
        chroma_where = None
        if where:
            chroma_where = {k: str(v) for k, v in where.items()}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                where=chroma_where if chroma_where else None,
            )
        except Exception:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        return results

    def delete_by_filter(self, where):
        if not where:
            return
        chroma_where = {k: str(v) for k, v in where.items()}
        try:
            self._collection.delete(where=chroma_where)
        except Exception:
            pass

    def count(self):
        return self._collection.count()


# ═══════════════════════════════════════════════════════════════════════════
# NUMPY FALLBACK BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class NumpyVectorStore:
    """File-backed numpy vector store (fallback when ChromaDB unavailable)."""

    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._ids = []
        self._embeddings = None
        self._metadatas = []
        self._documents = []
        self._load()

    def _index_path(self):
        return os.path.join(self.storage_dir, "index.json")

    def _embeddings_path(self):
        return os.path.join(self.storage_dir, "embeddings.npy")

    def _load(self):
        idx_path = self._index_path()
        emb_path = self._embeddings_path()
        if os.path.exists(idx_path) and os.path.exists(emb_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids = data.get("ids", [])
            self._metadatas = data.get("metadatas", [])
            self._documents = data.get("documents", [])
            self._embeddings = np.load(emb_path)
        else:
            self._ids = []
            self._metadatas = []
            self._documents = []
            self._embeddings = None

    def _save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump({
                "ids": self._ids,
                "metadatas": self._metadatas,
                "documents": self._documents,
            }, f)
        if self._embeddings is not None:
            np.save(self._embeddings_path(), self._embeddings)

    def add(self, ids, embeddings, metadatas, documents):
        new_emb = np.array(embeddings, dtype=np.float32)
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        else:
            self._embeddings = new_emb
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        self._documents.extend(documents)
        self._save()

    def search(self, query_embedding, top_k=20, where=None):
        if self._embeddings is None or len(self._embeddings) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms
        similarities = normed @ query_norm

        mask = np.ones(len(self._ids), dtype=bool)
        if where:
            for key, val in where.items():
                for i, meta in enumerate(self._metadatas):
                    if str(meta.get(key, "")) != str(val):
                        mask[i] = False

        similarities[~mask] = -1
        k = min(top_k, int(mask.sum()))
        if k == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        top_indices = np.argsort(similarities)[::-1][:k]
        result_ids = [self._ids[i] for i in top_indices]
        result_docs = [self._documents[i] for i in top_indices]
        result_metas = [self._metadatas[i] for i in top_indices]
        result_dists = [float(1 - similarities[i]) for i in top_indices]

        return {
            "ids": [result_ids],
            "documents": [result_docs],
            "metadatas": [result_metas],
            "distances": [result_dists],
        }

    def delete_by_filter(self, where):
        if not where:
            return
        keep = []
        for i, meta in enumerate(self._metadatas):
            match = all(str(meta.get(k, "")) == str(v) for k, v in where.items())
            if not match:
                keep.append(i)
        if len(keep) == len(self._ids):
            return
        self._ids = [self._ids[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        if self._embeddings is not None and len(keep) > 0:
            self._embeddings = self._embeddings[keep]
        else:
            self._embeddings = None
        self._save()

    def count(self):
        return len(self._ids)


# ═══════════════════════════════════════════════════════════════════════════
# STORE FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def _get_store():
    """Instantiate the configured vector store backend (singleton)."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                if config.VECTOR_STORE_BACKEND == "chromadb":
                    try:
                        _store = ChromaDBStore(config.CHROMADB_PATH)
                    except Exception:
                        # ChromaDB import/init failed — fall back to numpy
                        _store = NumpyVectorStore(config.CHROMADB_PATH)
                else:
                    _store = NumpyVectorStore(config.CHROMADB_PATH)
    return _store


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API (unchanged interface)
# ═══════════════════════════════════════════════════════════════════════════

def add_chunks(project_id, chunks, embeddings):
    """Add parsed chunks with embeddings to the vector store."""
    store = _get_store()

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        cid = f"p{project_id}_{uuid.uuid4().hex[:12]}"
        ids.append(cid)
        documents.append(chunk["content"][:10000])
        metadatas.append({
            "project_id": str(project_id),
            "file_path": chunk.get("file_path", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "name": chunk.get("name", ""),
            "start_line": chunk.get("start_line", 0),
            "end_line": chunk.get("end_line", 0),
            "language": chunk.get("language", ""),
        })

    store.add(ids, embeddings, metadatas, documents)
    return ids


def search(query_embedding, project_id, top_k=20):
    """Perform vector similarity search within a project."""
    store = _get_store()
    results = store.search(
        query_embedding, top_k=top_k,
        where={"project_id": str(project_id)},
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for i, cid in enumerate(results["ids"][0]):
            chunks.append({
                "chromadb_id": cid,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })

    return chunks


def delete_project(project_id):
    """Delete all vectors for a project."""
    store = _get_store()
    store.delete_by_filter({"project_id": str(project_id)})


def get_collection_count():
    """Get total number of vectors in the store."""
    store = _get_store()
    return store.count()
