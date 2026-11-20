"""Compare plusieurs configurations RAG (chunk_size, overlap, top_k) avec Hit@K."""
from src.rag_assistant.evaluation import TEST_QUESTIONS, evaluate_retrieval
from src.rag_assistant.loaders import load_folder
from src.rag_assistant.vector_store import build_vector_store

CONFIGURATIONS = [
    {"name": "Configuration A", "chunk_size": 50, "overlap": 10, "top_k": 2},
    {"name": "Configuration B", "chunk_size": 80, "overlap": 15, "top_k": 3},
    {"name": "Configuration C", "chunk_size": 100, "overlap": 20, "top_k": 3},
]


def main():
    documents = load_folder("data/montelune")
    for config in CONFIGURATIONS:
        store = build_vector_store(documents, config["chunk_size"], config["overlap"])
        score, details = evaluate_retrieval(TEST_QUESTIONS, store, top_k=config["top_k"])
        print(
            f"{config['name']} | chunk_size={config['chunk_size']} "
            f"overlap={config['overlap']} top_k={config['top_k']} "
            f"-> Hit@{config['top_k']} = {score * 100:.1f} %"
        )
        for item in details:
            if not item["hit"]:
                print("   FAIL:", item["question"], "->", item["retrieved"])


if __name__ == "__main__":
    main()
