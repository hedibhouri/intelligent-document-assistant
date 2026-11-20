import pytest

from src.rag_assistant.chunking import create_chunks, split_into_chunks


def test_overlap_is_kept():
    words = " ".join(f"w{i}" for i in range(10))
    chunks = split_into_chunks(words, chunk_size=4, overlap=2)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1] == "w2 w3 w4 w5"


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_into_chunks("a b c", chunk_size=2, overlap=2)


def test_chunks_keep_source_metadata():
    docs = [{"text": "un deux trois quatre cinq", "source": "doc.txt"}]
    chunks = create_chunks(docs, chunk_size=3, overlap=1)
    assert all(c["source"] == "doc.txt" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
