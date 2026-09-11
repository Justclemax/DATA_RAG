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
    return PictureDescriptionApiOptions(
        url=f"{OLLAMA_URL}/v1/chat/completions",
        params={
            "model": VLM_MODEL,
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
