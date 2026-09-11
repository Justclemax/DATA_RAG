"""
Nettoyage des formules LaTeX extraites par Docling.

Deux niveaux de nettoyage :
  1. Un nettoyage regex local, rapide et déterministe (artefacts OCR,
     '&' parasites, espaces autour des indices/exposants...).
  2. Un nettoyage sémantique délégué à Mathstral, appelé via Mellea
     (@generative) plutôt qu'un appel HTTP brut : la sortie est
     contrainte par un schéma Pydantic (garantie de recevoir un champ
     `latex`), ce qui correspond au pattern Instruct-Validate-Repair de
     Mellea.

Si Ollama/Mathstral n'est pas disponible (serveur éteint, modèle
manquant...), le module se rabat automatiquement sur le nettoyage
regex seul, comme le faisait le script d'origine.
"""

import re

from pydantic import BaseModel, Field

from mellea import MelleaSession, generative
from mellea.backends.ollama import OllamaModelBackend

from .config import MATH_MODEL, OLLAMA_URL
from loguru import logger

# Parallel helper pour accélérer le nettoyage des formules
from .parallel_executor import parallel_map

FORMULA_PATTERN = r"\$\$(.*?)\$\$"


# ============================================================
# Nettoyage regex local
# ============================================================

def clean_latex(latex: str) -> str:
    """
    Nettoyage des erreurs LaTeX/OCR, avant et/ou après passage par Mathstral.
    """

    # Supprimer les blocs Markdown éventuels
    latex = latex.replace("```latex", "")
    latex = latex.replace("```", "")

    # Supprimer les erreurs OCR du type "Misplaced &"
    latex = re.sub(
        r"Misplaced\s*&",
        "",
        latex,
        flags=re.IGNORECASE,
    )

    # Supprimer les & parasites
    latex = latex.replace("&", "")

    # Nettoyer les espaces multiples
    latex = re.sub(r"\s+", " ", latex)

    # Nettoyer les espaces dans les indices
    latex = re.sub(
        r"_\s*\{\s*([^{}]+?)\s*\}",
        r"_{\1}",
        latex,
    )

    # Nettoyer les espaces dans les exposants
    latex = re.sub(
        r"\^\s*\{\s*([^{}]+?)\s*\}",
        r"^{\1}",
        latex,
    )

    # Nettoyer les espaces autour de \frac
    latex = re.sub(
        r"\\frac\s*\{\s*",
        r"\\frac{",
        latex,
    )

    return latex.strip()


# ============================================================
# Schéma de sortie structuré (garanti par Mellea/Ollama)
# ============================================================

class CleanedFormula(BaseModel):
    """Formule LaTeX nettoyée, renvoyée par Mathstral."""

    latex: str = Field(
        description="La formule LaTeX nettoyée, sans artefacts OCR ni '&' parasites."
    )


# ============================================================
# Appel générative Mellea -> Mathstral
# ============================================================
#
# Le docstring ci-dessous EST le prompt envoyé au modèle (c'est le
# fonctionnement du décorateur @generative de Mellea : docstring ->
# prompt, type hints -> schéma de sortie). Il reprend exactement les
# règles du MATH_PROMPT d'origine.

@generative
def clean_formula_with_mellea(formula: str) -> CleanedFormula:
    """You are an expert in mathematics and LaTeX.

    Clean the mathematical formula extracted from a scientific PDF.

    Rules:
    - Return valid LaTeX only.
    - Preserve the mathematical meaning exactly.
    - Do not invent missing symbols.
    - Do not change variables, indexes, coefficients, or equation numbers.
    - Remove OCR artifacts.
    - Remove misplaced '&' characters.
    - Fix malformed LaTeX commands.
    - Fix broken spaces around subscripts and superscripts.
    - Correct malformed \\frac, \\sum, \\epsilon, etc.
    - Keep the original mathematical structure.
    - Do not explain anything.
    - Return only the cleaned LaTeX formula, in the `latex` field.

    Example:

    Input:
    Misplaced &

    $$L _ { SD } = 1 - \\frac { \\sum _ { i = 1 } ^ { t }
    y _ { i } p _ { i } + \\epsilon }
    { \\sum _ { i = 1 } ^ { t } y _ { i } + p _ { i } + \\epsilon }
    \\quad ... (2) $$

    Output latex field:
    L_{SD} = 1 - \\frac{\\sum_{i=1}^{t} y_i p_i + \\epsilon}{\\sum_{i=1}^{t} y_i + p_i + \\epsilon} \\tag{2}
    """


# ============================================================
# Session Mellea dédiée à Mathstral
# ============================================================

def build_session(timeout: float = 120.0) -> MelleaSession:
    """
    Crée une session Mellea reliée à Mathstral via Ollama.

    Lève une exception (ConnectionError/OSError/ValueError) si le
    serveur Ollama n'est pas joignable à OLLAMA_URL ou si le modèle
    MATH_MODEL ne peut pas être chargé/pull.
    """

    backend = OllamaModelBackend(
        model_id=MATH_MODEL,
        base_url=OLLAMA_URL,
        model_options={
            "temperature": 0,
            "seed": 42,
        },
        timeout=timeout,
    )

    return MelleaSession(backend)


# ============================================================
# Traitement des formules d'un document
# ============================================================

def process_formulas(content: str, use_parallel: bool = True, max_workers: int | None = None, chunksize: int = 1, show_progress: bool = False) -> str:
    """
    Détecte les formules $$...$$ extraites par Docling et les nettoie.

    Cette version peut paralléliser le nettoyage des formules (chaque
    formule est traitée dans un process séparé). Le worker construit sa
    propre session Mellea si possible et retombe sur le nettoyage regex
    local en cas d'échec.

    Args:
        content: texte complet du document Markdown/texte.
        use_parallel: si True, utilise parallel_map pour traiter en parallèle.
        max_workers: nombre max de workers (None -> os.cpu_count()).
        chunksize: taille de lot pour l'envoi aux workers (voir parallel_map).
        show_progress: si True et si tqdm disponible, affiche la progression.

    Returns:
        Le contenu avec chaque formule nettoyée et replacée.
    """

    matches = list(re.finditer(FORMULA_PATTERN, content, flags=re.DOTALL))
    if not matches:
        return content

    originals = [m.group(1).strip() for m in matches]

    def _clean_formula_worker(formula: str) -> str:
        """Fonction exécutée dans le process worker."""
        try:
            # Tenter de construire une session Mellea locale au worker
            try:
                session = build_session()
            except Exception:
                session = None

            if session is not None:
                try:
                    res = clean_formula_with_mellea(session, formula=formula)
                    return clean_latex(res.latex)
                except Exception:
                    # si Mellea échoue pour cette formule, repli plus bas
                    pass

        except Exception:
            # Garantir que toute exception du worker ne casse pas le flow
            pass

        # Repli: nettoyage regex local
        return clean_latex(formula)

    # Exécuter en parallèle ou séquentiellement selon use_parallel
    if use_parallel and len(originals) > 1:
        cleaned_list = parallel_map(_clean_formula_worker, originals, max_workers=max_workers, chunksize=chunksize, show_progress=show_progress)
    else:
        cleaned_list = [_clean_formula_worker(f) for f in originals]

    # Normaliser les éventuelles exceptions retournées
    for i, item in enumerate(cleaned_list):
        if isinstance(item, Exception):
            cleaned_list[i] = clean_latex(originals[i])

    # Reconstruire le contenu en remplaçant chaque match par la formule nettoyée
    parts: list[str] = []
    last_end = 0
    for m, cleaned in zip(matches, cleaned_list):
        start, end = m.span()
        parts.append(content[last_end:start])
        parts.append(f"\n\n$$\n{cleaned}\n$$\n\n")
        last_end = end
    parts.append(content[last_end:])

    return "".join(parts)
