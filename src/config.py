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
# Valeurs par défaut locales si l'environnement n'est pas encore défini
# (par exemple lors d'un premier lancement avec `uv run ...`).
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

VLM_MODEL = os.getenv("VLM_MODEL", "granite3.2-vision:latest")

MATH_MODEL = os.getenv("MATH_MODEL", "mathstral:latest")


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
