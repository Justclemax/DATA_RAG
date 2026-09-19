"""Prompts courts pour Mellea."""

VLM_PROMPT = (
"Describe the image briefly and faithfully. Focus on the main visual "
"content, structure, and scientific meaning. Do not invent details."
)

MATH_PROMPT = (
"Clean this LaTeX formula. Remove OCR artefacts and stray '&'. Return "
"valid LaTeX only. Keep the same mathematical meaning."
)

PDF_EXTRACTION_PROMPT = (
"Convert the PDF to clean Markdown. Preserve headings, equations, "
"tables, figures, and references. Do not hallucinate missing "
"information."
)

MELLEA_IBM_PROMPT = (
"Use the IBM Mellea pattern: generate, validate, repair. Return a "
"concise, accurate answer supported only by the source text."
)
