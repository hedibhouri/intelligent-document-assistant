"""Chargement de documents TXT, Markdown, DOCX et PDF."""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def load_document(file_path):
    """Retourne {"text": ..., "source": nom_du_fichier} pour un document."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
    elif extension == ".docx":
        from docx import Document

        doc = Document(path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif extension == ".pdf":
        from pypdf import PdfReader

        pdf = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        raise ValueError(f"Format non supporté : {extension}")

    return {"text": text, "source": path.name}


def load_folder(folder):
    """Charge tous les documents supportés et non vides d'un dossier."""
    documents = []
    for file_path in sorted(Path(folder).iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        document = load_document(file_path)
        if document["text"].strip():
            documents.append(document)
    return documents
