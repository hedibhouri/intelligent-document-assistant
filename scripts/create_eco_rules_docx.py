"""Génère le document Word d'exemple (eco_rules.docx) utilisé par le corpus Montelune."""
from pathlib import Path

from docx import Document

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "montelune" / "eco_rules.docx"


def main() -> None:
    doc = Document()
    doc.add_heading("REGLES ECOLOGIQUES DE MONTELUNE", level=1)

    doc.add_heading("Protection de la nature", level=2)
    doc.add_paragraph(
        "Les visiteurs doivent respecter la faune et la flore. "
        "Il est interdit de cueillir les plantes protégées."
    )

    doc.add_heading("Gestion des déchets", level=2)
    doc.add_paragraph(
        "Les déchets doivent être déposés dans les poubelles "
        "prévues à cet effet. Les visiteurs sont encouragés "
        "à utiliser une gourde réutilisable."
    )

    doc.add_heading("Sentiers de randonnée", level=2)
    doc.add_paragraph(
        "Les visiteurs doivent rester sur les sentiers balisés. "
        "Les feux sont interdits dans les zones forestières."
    )

    doc.add_heading("Animaux", level=2)
    doc.add_paragraph(
        "Les chiens doivent être tenus en laisse dans les "
        "zones naturelles. Il est interdit de nourrir les "
        "animaux sauvages."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print("Document DOCX créé :", OUTPUT)


if __name__ == "__main__":
    main()
