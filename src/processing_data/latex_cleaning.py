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
from .parallel_executor import parallel_map
from .prompts import MATH_PROMPT
from loguru import logger

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
# Mellea utilise le docstring comme instruction envoyée au modèle.
# On garde le prompt centralisé dans prompts.py pour éviter le doublon.

@generative
def clean_formula_with_mellea(formula: str) -> CleanedFormula:
    """Clean this LaTeX formula. Remove OCR artefacts and stray '&'. Return valid LaTeX only. Keep the same mathematical meaning."""
    pass


clean_formula_with_mellea.__doc__ = MATH_PROMPT


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

def _clean_formula_worker(formula: str) -> str:
    """Nettoie une formule isolée, avec repli local si Mellea/Ollama échoue."""
    try:
        try:
            session = build_session()
        except Exception as exc:
            logger.warning(f"[WARNING] Could not initialize Mellea/Ollama session: {exc}")
            logger.info("[INFO] Falling back to local regex-only LaTeX cleaning.")
            session = None

        if session is not None:
            try:
                result = clean_formula_with_mellea(session, formula=formula)
                cleaned_formula = clean_latex(result.latex)
                logger.info("[Mellea/Mathstral] Formula cleaned:")
                logger.info(cleaned_formula)
                return cleaned_formula
            except Exception as exc:
                logger.warning(f"[WARNING] Mellea/Mathstral error: {exc}")
    except Exception:
        pass

    fallback = clean_latex(formula)
    logger.info("[Local fallback] Formula cleaned:")
    logger.info(fallback)
    return fallback


def process_formulas(
    content: str,
    use_parallel: bool = True,
    max_workers: int | None = None,
    chunksize: int = 1,
    show_progress: bool = False,
) -> str:
    """
    Détecte les formules $$...$$ extraites par Docling et les nettoie
    via Mellea/Mathstral, avec repli sur le nettoyage regex local si le
    modèle est indisponible ou échoue sur une formule donnée.

    Args:
        content: document markdown complet.
        use_parallel: active la parallélisation via ProcessPoolExecutor.
        max_workers: nombre de workers, par défaut le CPU count.
        chunksize: taille de lot envoyée aux workers.
        show_progress: affiche une barre de progression si tqdm est présent.
    """
    matches = list(re.finditer(FORMULA_PATTERN, content, flags=re.DOTALL))
    if not matches:
        return content

    originals = [match.group(1).strip() for match in matches]

    if use_parallel and len(originals) > 1:
        cleaned_formulas = parallel_map(
            _clean_formula_worker,
            originals,
            max_workers=max_workers,
            chunksize=chunksize,
            show_progress=show_progress,
        )
    else:
        cleaned_formulas = [_clean_formula_worker(formula) for formula in originals]

    parts: list[str] = []
    last_end = 0
    for match, cleaned_formula in zip(matches, cleaned_formulas):
        start, end = match.span()
        parts.append(content[last_end:start])
        parts.append(f"\n\n$$\n{cleaned_formula}\n$$\n\n")
        last_end = end

    parts.append(content[last_end:])
    return "".join(parts)
