"""Pipeline RAG : retrieval + contexte + génération avec Qwen 2.5 (réponses sourcées)."""
from .vector_store import retrieve

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NOT_FOUND = "Je ne trouve pas cette information dans les documents fournis."

SYSTEM_PROMPT = f"""
Tu es un assistant documentaire.
Tu dois répondre UNIQUEMENT avec les informations du contexte.

Règles obligatoires :
- N'invente aucune information.
- N'utilise pas tes connaissances externes.
- Ne crée pas de prix, dates ou horaires.
- Si le contexte ne contient pas la réponse, réponds exactement :
  {NOT_FOUND}
- Indique les noms des documents utilisés.
- Réponds en français simplement.
"""

_llm = None


def get_llm():
    """Charge le tokenizer et le modèle Qwen une seule fois."""
    global _llm
    if _llm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        model.eval()
        _llm = (tokenizer, model)
    return _llm


def build_context(results):
    """Assemble les chunks retrouvés en un contexte annoté par source."""
    context = ""
    for i, result in enumerate(results):
        context += f"\n[SOURCE {i + 1}: {result['source']}]\n{result['text']}\n"
    return context


def generate(question, context, max_new_tokens=250):
    """Génère une réponse avec Qwen à partir du contexte documentaire."""
    import torch

    tokenizer, model = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXTE DOCUMENTAIRE :\n{context}\n\nQUESTION :\n{question}\n\nRÉPONSE :"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def rag_answer(question, store, top_k=3, min_score=0.25):
    """Répond à une question. Refuse si aucun chunk n'est assez pertinent (anti-hallucination)."""
    retrieved = retrieve(question, store, top_k=top_k)
    if not retrieved or retrieved[0]["score"] < min_score:
        return {"answer": NOT_FOUND, "sources": []}

    answer = generate(question, build_context(retrieved))
    sources = sorted({chunk["source"] for chunk in retrieved})
    return {"answer": answer, "sources": sources}
