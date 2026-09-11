"""
Orchestration du traitement d'un PDF :

  1. Conversion PDF -> Markdown via Docling (backend PyPDFium).
  2. Marquage des descriptions d'images générées par le VLM.
  3. Nettoyage des formules mathématiques via Mellea/Mathstral.
"""

from pathlib import Path
from typing import Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from .config import (
    IMAGE_DESCRIPTION_END,
    IMAGE_DESCRIPTION_START,
    PAGE_BREAK_PLACEHOLDER,
)
from .latex_cleaning import process_formulas
from .vlm_pipeline import create_pdf_pipeline_options


def process_document(pdf_path: Path, use_parallel: bool = True, max_workers: int | None = None, chunksize: int = 1, show_progress: bool = False) -> Any:

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=create_pdf_pipeline_options(),
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    # Conversion PDF avec Docling
    result = converter.convert(pdf_path)

    doc = result.document

    # Export Markdown
    content = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        include_annotations=True,
        mark_meta=True,
        mark_annotations=True,
        page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
        image_placeholder="",
    )

    # ========================================================
    # Description des images
    # ========================================================

    content_start = content.replace(
        '<!--<annotation kind="description">-->',
        IMAGE_DESCRIPTION_START,
    )

    content_end = content_start.replace(
        "<!--<annotation/>-->",
        IMAGE_DESCRIPTION_END,
    )

    # ========================================================
    # Nettoyage des formules (Mellea/Mathstral)
    # ========================================================

    content_math = process_formulas(content_end, use_parallel=use_parallel, max_workers=max_workers, chunksize=chunksize, show_progress=show_progress)

    return content_math
