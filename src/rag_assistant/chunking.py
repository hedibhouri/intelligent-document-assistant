"""Découpage des documents en chunks avec chevauchement (overlap)."""


def split_into_chunks(text, chunk_size=80, overlap=15):
    """Découpe un texte en morceaux de chunk_size mots avec overlap mots communs."""
    if overlap >= chunk_size:
        raise ValueError("Overlap doit être inférieur au chunk_size.")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_text = " ".join(words[start:start + chunk_size])
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


def create_chunks(documents, chunk_size=80, overlap=15):
    """Transforme une liste de documents en chunks avec leurs métadonnées."""
    all_chunks = []
    for document in documents:
        for i, chunk in enumerate(split_into_chunks(document["text"], chunk_size, overlap)):
            all_chunks.append({"text": chunk, "source": document["source"], "chunk_id": i})
    return all_chunks
