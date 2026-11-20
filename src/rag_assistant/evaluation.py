"""Évaluation du retrieval avec la métrique Hit@K."""
from .vector_store import retrieve

TEST_QUESTIONS = [
    {"question": "Quels sont les trois sites touristiques principaux ?", "expected_source": "destination.txt"},
    {"question": "À quelle altitude se trouve le village principal ?", "expected_source": "destination.txt"},
    {"question": "Combien de temps faut-il pour rejoindre le lac Azur à pied ?", "expected_source": "destination.txt"},
    {"question": "Quelle est la durée du parcours familial ?", "expected_source": "destination.txt"},
    {"question": "Quels sont les horaires de la navette touristique ?", "expected_source": "transport.md"},
    {"question": "Combien coûte la location d'un vélo classique ?", "expected_source": "transport.md"},
    {"question": "Combien coûte la location d'un vélo électrique ?", "expected_source": "transport.md"},
    {"question": "Où peut-on stationner sa voiture ?", "expected_source": "transport.md"},
    {"question": "Est-il autorisé de cueillir les plantes protégées ?", "expected_source": "eco_rules.docx"},
    {"question": "Quelles sont les règles concernant les chiens ?", "expected_source": "eco_rules.docx"},
]


def evaluate_retrieval(test_questions, store, top_k=3):
    """Hit@K = part des questions dont le document attendu figure dans les top_k résultats."""
    hits = 0
    details = []
    for test in test_questions:
        retrieved_sources = [r["source"] for r in retrieve(test["question"], store, top_k=top_k)]
        hit = test["expected_source"] in retrieved_sources
        hits += hit
        details.append({
            "question": test["question"],
            "expected": test["expected_source"],
            "retrieved": retrieved_sources,
            "hit": hit,
        })
    return hits / len(test_questions), details
