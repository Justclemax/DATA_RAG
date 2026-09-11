"""
Prompts utilisés par le pipeline.

Le prompt de nettoyage LaTeX (Mathstral) vit directement dans le
docstring de la fonction générative correspondante (voir
`latex_cleaning.py`) : Mellea utilise le docstring d'une fonction
`@generative` comme instruction envoyée au modèle, il ne peut donc pas
être un simple import de constante.
"""

VLM_PROMPT = (
    "You are an expert in remote sensing, satellite image analysis, "
    "and semantic segmentation. Analyze the image in the context of "
    "water-land segmentation using the U-Net model on optical remote "
    "sensing images. Describe in 1-2 sentences what the image shows, "
    "focusing on water bodies, land areas, segmentation masks, "
    "boundaries, spectral or visual patterns, and model predictions "
    "when visible. If the image is a chart, diagram, architecture, "
    "or experimental result, explain its purpose and the key "
    "information it conveys. Do not invent information that is "
    "not visible."
)
