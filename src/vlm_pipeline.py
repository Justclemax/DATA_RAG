"""
Configuration du pipeline Docling : description des images via Granite
Vision (Ollama), extraction des tableaux et enrichissement des formules.

Ce module ne change pas par rapport au script d'origine : Docling
appelle lui-même l'endpoint OpenAI-compatible d'Ollama pour la
description d'images (PictureDescriptionApiOptions), ce qui est
indépendant de Mellea. Seul le nettoyage des formules Mathstral, en
aval, passe par Mellea (voir `latex_cleaning.py`).
"""

from docling.datamodel.pipeline_options import (
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
        do_ocr=False,

        # Tables
        do_table_structure=True,

        # Images
        generate_picture_images=True,
        do_picture_description=True,

        # Formules
        do_formula_enrichment=True,

        picture_description_options=create_picture_description_options(),
    )
