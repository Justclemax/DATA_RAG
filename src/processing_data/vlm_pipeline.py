"""
Configuration du pipeline Docling : OCR multiplateforme,
description des images via Granite Vision (Ollama), extraction des
tableaux et enrichissement des formules.

Le backend OCR est choisi selon la plateforme pour éviter les erreurs
liées au support Linux-only de Nemotron sur macOS. Sur macOS, on
utilise RapidOCR (cross-platform), qui est installé dans l'environnement
actuel. Sur Linux, on peut conserver Nemotron si le package est
présent ; sinon le fallback est RapidOCR.
"""

import sys

from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
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
    if sys.platform == "linux":
        try:
            import nemotron_ocr.inference.pipeline_v2  # noqa: F401

            ocr_options = NemotronOcrOptions(lang=["multilingual"])
        except ImportError:
            ocr_options = RapidOcrOptions(
                backend="onnxruntime",
                lang=["chinese", "english"],
            )
    else:
        ocr_options = RapidOcrOptions(
            backend="onnxruntime",
            lang=["chinese", "english"],
        )

    return PdfPipelineOptions(
        enable_remote_services=True,

        # ========================================================
        # OCR - platform-aware backend
        # ========================================================

        do_ocr=True,
        ocr_options=ocr_options,

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