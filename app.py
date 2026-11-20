"""Interface chatbot Gradio de l'assistant documentaire."""
import gradio as gr

from src.rag_assistant.loaders import load_folder
from src.rag_assistant.rag import rag_answer
from src.rag_assistant.vector_store import build_vector_store

DATA_FOLDER = "data/montelune"

store = build_vector_store(load_folder(DATA_FOLDER), chunk_size=80, overlap=15)


def chatbot_interface(message, history):
    response = rag_answer(message, store, top_k=3)
    answer = response["answer"]
    if response["sources"]:
        answer += "\n\n**Sources utilisées :**"
        for source in response["sources"]:
            answer += f"\n- {source}"
    return answer


demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="Intelligent Document Assistant (RAG)",
    description="Posez vos questions sur les documents du corpus. Les réponses citent leurs sources.",
    examples=[
        "Quels sont les sites touristiques de Montelune ?",
        "Quels sont les horaires de la navette ?",
        "Combien coûte la location d'un vélo ?",
        "Quelles sont les règles écologiques ?",
    ],
)

if __name__ == "__main__":
    demo.launch()
