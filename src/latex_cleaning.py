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

from config import MATH_MODEL, OLLAMA_URL


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

def process_formulas(content: str) -> str:
    """
    Détecte les formules $$...$$ extraites par Docling et les nettoie
    via Mellea/Mathstral, avec repli sur le nettoyage regex local si le
    modèle est indisponible ou échoue sur une formule donnée.
    """

    session: MelleaSession | None = None

    try:
        session = build_session()
    except Exception as exc:
        print(f"[WARNING] Could not initialize Mellea/Ollama session: {exc}")
        print("[INFO] Falling back to local regex-only LaTeX cleaning for all formulas.")

    def replace_formula(match: re.Match) -> str:

        original_formula = match.group(1).strip()

        print("\n[Mellea/Mathstral]")
        print("Formula detected:")
        print(original_formula)

        if session is not None:
            try:
                result = clean_formula_with_mellea(session, formula=original_formula)
                cleaned_formula = clean_latex(result.latex)

                print("Formula cleaned:")
                print(cleaned_formula)

                return f"\n\n$$\n{cleaned_formula}\n$$\n\n"

            except Exception as exc:
                print(f"[WARNING] Mellea/Mathstral error: {exc}")

        # Si Mellea/Mathstral échoue (ou est indisponible),
        # on garde au minimum le nettoyage regex local.
        fallback = clean_latex(original_formula)

        return f"\n\n$$\n{fallback}\n$$\n\n"

    return re.sub(
        FORMULA_PATTERN,
        replace_formula,
        content,
        flags=re.DOTALL,
    )
