"""Embeddings multilingues + index vectoriel FAISS + recherche (retrieval)."""
import faiss
from sentence_transformers import SentenceTransformer

from .chunking import create_chunks

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None


def get_embedding_model():
    """Charge le modèle d'embeddings une seule fois."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed(texts):
    """Encode des textes en vecteurs normalisés (float32)."""
    vectors = get_embedding_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )
    return vectors.astype("float32")


def build_vector_store(documents, chunk_size=80, overlap=15):
    """Crée les chunks puis un index FAISS (produit scalaire = similarité cosinus)."""
    chunks = create_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    vectors = embed([chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return {"index": index, "chunks": chunks}


def retrieve(query, store, top_k=3):
    """Retourne les top_k chunks les plus proches de la question, avec leur score."""
    index, chunks = store["index"], store["chunks"]
    if index.ntotal == 0:
        return []

    k = min(top_k, index.ntotal)
    scores, indices = index.search(embed([query]), k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": float(score),
        })
    return results
