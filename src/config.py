"""
Configuration centralisée du pipeline.

Toutes les variables d'environnement, noms de modèles Ollama et chemins
par défaut sont définis ici pour éviter toute duplication entre modules.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Ollama
# ============================================================

OLLAMA_URL = os.getenv("OLLAMA_URL",)

VLM_MODEL = os.getenv("VLM_MODEL",)

MATH_MODEL = os.getenv("MATH_MODEL")


# ============================================================
# Chemins par défaut (relatifs au dossier d'exécution, cf. main.py)
# ============================================================

DEFAULT_INPUT_PDF = Path("../data/0000721.pdf")
DEFAULT_OUTPUT_DIR = Path("../data/output")


# ============================================================
# Placeholders Markdown
# ============================================================

PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"

IMAGE_DESCRIPTION_START = "<image_description>"
IMAGE_DESCRIPTION_END = "</image_description>"
