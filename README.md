# Intelligent Document Assistant — RAG multilingue

Assistant IA qui répond aux questions **uniquement à partir de vos documents** (TXT, Markdown, Word, PDF) et **cite ses sources**. S'il ne trouve pas la réponse dans les documents, il le dit au lieu d'inventer.

> Projet personnel réalisé dans le cadre de ma formation en AI Engineering : pipeline RAG complet, de la lecture des documents jusqu'à l'évaluation de la recherche.

## Démo

```
Question : Quels transports peut-on utiliser à Montelune ?

Réponse : 1. Navette touristique : elle circule entre le village principal, la gare et le
centre d'information touristique, de 8h00 à 19h00, toutes les 30 minutes...
2. Location de vélos : 12 euros par jour, casque inclus...

Sources utilisées : transport.md, destination.txt
```

```
Question : Quel est le nombre d'habitants de Montelune ?
Réponse : Je ne trouve pas cette information dans les documents fournis.
```

## Architecture

```
Documents (TXT / MD / DOCX / PDF)
        │  loaders.py
        ▼
Chunking (fenêtre de mots + overlap)          chunking.py
        │
        ▼
Embeddings multilingues                        vector_store.py
(paraphrase-multilingual-MiniLM-L12-v2, vecteurs normalisés)
        │
        ▼
Index FAISS (IndexFlatIP = similarité cosinus)
        │   question → top-k chunks + score
        ▼
Seuil de pertinence (min_score) ── trop faible ──► « Je ne trouve pas… »
        │
        ▼
Prompt avec contexte annoté par source         rag.py
        │
        ▼
LLM Qwen2.5-1.5B-Instruct (génération déterministe)
        │
        ▼
Réponse + liste des sources ──► Interface Gradio (app.py)
```

## Ce que le projet montre

- **Ingestion multi-format** : TXT, Markdown, DOCX (python-docx) et PDF (pypdf).
- **Chunking avec chevauchement** pour ne pas couper une information entre deux morceaux.
- **Recherche vectorielle** avec FAISS sur des embeddings multilingues (questions et documents en français).
- **Anti-hallucination** : seuil de score minimal + prompt système strict (« n'invente aucune information ») + réponse de refus standard.
- **Réponses sourcées** : chaque réponse liste les documents utilisés.
- **Évaluation** : métrique **Hit@K** sur un jeu de 10 questions annotées, et comparaison de plusieurs configurations (taille de chunk, overlap, top-k).

## Résultats de l'évaluation (Hit@K)

| Configuration | chunk_size | overlap | top_k | Hit@K |
|---|---|---|---|---|
| A | 50 | 10 | 2 | 100 % |
| Défaut | 80 | 15 | 3 | 100 % |
| B | 100 | 20 | 3 | 100 % |

Le corpus d'exemple est volontairement petit (3 documents fictifs sur la destination « Montelune », 10 questions). Les prochaines étapes visent un corpus plus large, où les configurations se différencient vraiment.

## Lancer le projet

```bash
pip install -r requirements.txt
python scripts/create_eco_rules_docx.py   # génère le document Word d'exemple
python evaluate.py                         # évaluation Hit@K des configurations
python app.py                              # interface chatbot Gradio
pytest                                     # tests unitaires du chunking
```

Le modèle Qwen2.5-1.5B tourne sur CPU mais beaucoup plus vite sur GPU. Le notebook d'origine (`notebooks/rag_assistant_colab.ipynb`) s'exécute directement sur Google Colab.

## Structure

```
├── app.py                      # interface Gradio
├── evaluate.py                 # comparaison des configurations (Hit@K)
├── data/montelune/             # corpus d'exemple
├── notebooks/                  # notebook Colab d'origine, étape par étape
├── scripts/                    # génération du document DOCX d'exemple
├── src/rag_assistant/
│   ├── loaders.py              # lecture TXT / MD / DOCX / PDF
│   ├── chunking.py             # découpage avec overlap
│   ├── vector_store.py         # embeddings + FAISS + retrieval
│   ├── rag.py                  # contexte, prompt, génération Qwen
│   └── evaluation.py           # métrique Hit@K
└── tests/
```

## Prochaines étapes

- API REST avec **FastAPI** et conteneurisation **Docker**.
- Upload de documents depuis l'interface.
- Corpus plus large et métriques supplémentaires (MRR, qualité des réponses).

## Stack

Python · Sentence-Transformers · FAISS · Hugging Face Transformers · Qwen 2.5 · Gradio · python-docx · pypdf · pytest

---
Auteur : **Mohamed Hedi Bhouri** — [LinkedIn](https://www.linkedin.com/in/hedibhouri/) · [GitHub](https://github.com/hedibhouri)
