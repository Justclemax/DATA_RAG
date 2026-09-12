"""
Configuration du pipeline Docling : OCR multilingue via Nemotron,
description des images via Granite Vision (Ollama), extraction des
tableaux et enrichissement des formules.

Le pipeline utilise :
- Nemotron-OCR pour l'extraction de texte multilingue ;
- Granite Vision via Ollama pour la description des images ;
- Docling pour la structure des tableaux ;
- Mellea/Mathstral en aval pour le nettoyage des formules.
"""

from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
)

from .config import OLLAMA_URL, VLM_MODEL
from .prompts import VLM_PROMPT


def create_picture_description_options() -> PictureDescriptionApiOptions:
    base_url = str(OLLAMA_URL or "").strip().rstrip("/")

    if not base_url or not base_url.startswith(("http://", "https://")):
        raise ValueError(
            "OLLAMA_URL is not configured. Set it in .env or use the default "
            "http://localhost:11434 before running the PDF pipeline."
        )

    model_name = str(VLM_MODEL or "").strip()

    if not model_name:
        raise ValueError(
            "VLM_MODEL is not configured. Set it in .env or use the default "
            "granite3.2-vision:latest."
        )

    return PictureDescriptionApiOptions(
        url=f"{base_url}/v1/chat/completions",
        params={
            "model": model_name,
            "think": False,
            "seed": 42,
            "max_completion_tokens": 256,
        },
        prompt=VLM_PROMPT,
        timeout=90,
    )


def create_pdf_pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        enable_remote_services=True,

        # ========================================================
        # OCR - Nemotron multilingue
        # ========================================================

        do_ocr=True,

        ocr_options=NemotronOcrOptions(
            lang=["multilingual"],
        ),

        # ========================================================
        # Tables
        # ========================================================

        do_table_structure=True,

        # ========================================================
        # Images / VLM - Granite Vision via Ollama
        # ========================================================

        generate_picture_images=True,
        do_picture_description=True,

        picture_description_options=create_picture_description_options(),

        # ========================================================
        # Formules
        # ========================================================

        do_formula_enrichment=True,
    )