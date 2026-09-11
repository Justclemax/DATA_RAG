"""
Nettoyage des formules LaTeX extraites par Docling.

Deux niveaux de nettoyage :
  1. Un nettoyage regex local, rapide et déterministe (artefacts OCR,
     '&' parasites, espaces autour des indices/exposants, délimiteurs
     \\left/\\right mal échappés, tokens hallucinés...).
  2. Un nettoyage sémantique délégué à Mathstral, appelé via Mellea
     (@generative) plutôt qu'un appel HTTP brut : la sortie est
     contrainte par un schéma Pydantic (garantie de recevoir un champ
     `latex`) ET validée par deux `Requirement` Mellea avant d'être
     acceptée (pattern Instruct-Validate-Repair) :
       - pas de délimiteur \\left/\\right suivi d'une accolade nue
         (cause de l'erreur de rendu "Missing or unrecognized
         delimiter for \\left"),
       - pas de token isolé halluciné du type "\\ tual", "\\ ved".
     Si la validation échoue, Mellea relance automatiquement Mathstral
     (RejectionSamplingStrategy) en lui donnant la raison de l'échec
     avant de nous renvoyer un résultat.

Dans tous les cas, `clean_latex` ré-applique les mêmes corrections en
local (delimiters + tokens) en toute fin de chaîne : même si les
retries Mellea s'épuisent sans succès, ces deux classes d'erreurs ne
peuvent plus se retrouver dans la sortie finale.

Si Ollama/Mathstral n'est pas disponible (serveur éteint, modèle
manquant...), le module se rabat automatiquement sur le nettoyage
regex seul, comme le faisait le script d'origine.
"""

import json
import re

from pydantic import BaseModel, Field

from mellea import MelleaSession, generative
from mellea.backends.ollama import OllamaModelBackend
from mellea.stdlib.requirements.requirement import req, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

from .config import MATH_MODEL, OLLAMA_URL
from loguru import logger

# Parallel helper pour accélérer le nettoyage des formules
from .parallel_executor import parallel_map

FORMULA_PATTERN = r"\$\$(.*?)\$\$"

# Nombre de tentatives Mellea (génération + réparation) par formule.
MELLEA_LOOP_BUDGET = 3


# ============================================================
# Motifs de correction / détection partagés entre le nettoyage
# local (déterministe) et les Requirements Mellea (validation).
# ============================================================

# \left / \right suivis d'une accolade NON échappée : { ou } au lieu
# de \{ ou \} -> cause de "Missing or unrecognized delimiter for \left".
_BAD_DELIMITER_RE = re.compile(r"\\(left|right)([{}])")

# Backslash + espace + mot isolé en minuscules : jamais une vraie
# commande LaTeX (les commandes suivent directement le backslash),
# signe d'un token halluciné (ex: "\ tual", "\ ved").
_STRAY_TOKEN_RE = re.compile(r"\\ ([a-z]{2,})\b")


def _fix_left_right_delimiters(latex: str) -> str:
    """Échappe les accolades nues après \\left/\\right (\\left{ -> \\left\\{)."""
    return _BAD_DELIMITER_RE.sub(lambda m: f"\\{m.group(1)}\\{m.group(2)}", latex)


def _strip_stray_tokens(latex: str) -> str:
    """Retire les tokens hallucinés du type '\\ tual', '\\ ved'."""
    return _STRAY_TOKEN_RE.sub("", latex)


def _extract_latex(raw_output: str) -> str:
    """Récupère le champ `latex` d'une sortie JSON structurée, sinon le texte brut."""
    try:
        payload = json.loads(raw_output)
        return payload.get("latex", raw_output)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return raw_output


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

    # Corriger \left / \right suivis d'une accolade non échappée
    # (ex: \left{...\right} -> \left\{...\right\})
    latex = _fix_left_right_delimiters(latex)

    # Retirer les tokens hallucinés du type "\ tual", "\ ved"
    latex = _strip_stray_tokens(latex)

    # Corriger les environnements mathématiques non fermés
    # (ex: \begin{matrix} ... sans \end{matrix}) qui provoquent
    # des erreurs de parse MathJax du type "Missing \end{matrix}".
    envs = [
        "matrix",
        "pmatrix",
        "bmatrix",
        "Bmatrix",
        "vmatrix",
        "Vmatrix",
        "cases",
        "aligned",
        "align",
        "array",
        "equation",
        "eqnarray",
    ]
    for env in envs:
        begin_pattern = rf"\\begin\s*\{{\s*{env}\s*\}}"
        end_pattern = rf"\\end\s*\{{\s*{env}\s*\}}"
        begin_count = len(re.findall(begin_pattern, latex))
        end_count = len(re.findall(end_pattern, latex))
        if begin_count > end_count:
            latex = latex.rstrip()
            latex += " " + " ".join([f"\\end{{{env}}}" for _ in range(begin_count - end_count)])
        elif end_count > begin_count:
            latex = re.sub(end_pattern, "", latex, count=end_count - begin_count)

    # Recompacter les espaces laissés par les suppressions ci-dessus
    latex = re.sub(r"\s+", " ", latex)

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
# Requirements Mellea (Instruct-Validate-Repair)
# ============================================================

def _latex_delimiters_ok(raw_output: str) -> tuple[bool, str]:
    latex = _extract_latex(raw_output)
    if _BAD_DELIMITER_RE.search(latex):
        return False, (
            "\\left and \\right must always be followed by a valid, escaped "
            "delimiter (\\{, \\}, (, ), [, ], |, ., <, >), never a bare "
            "'{' or '}'. Fix every occurrence and return the full formula again."
        )
    return True, ""


def _no_stray_tokens_ok(raw_output: str) -> tuple[bool, str]:
    latex = _extract_latex(raw_output)
    match = _STRAY_TOKEN_RE.search(latex)
    if match:
        return False, (
            f"Found a suspicious token {match.group(0)!r}: a backslash "
            "followed by a space and a lowercase word is not a real LaTeX "
            "command. Do not invent words; remove it and return the full "
            "formula again."
        )
    return True, ""


LATEX_REQUIREMENTS = [
    req(
        "\\left and \\right must always be followed by a valid, escaped "
        "delimiter, never a bare '{' or '}'.",
        simple_validate(_latex_delimiters_ok),
    ),
    req(
        "The output must not contain invented words or stray tokens that "
        "are not real LaTeX commands.",
        simple_validate(_no_stray_tokens_ok),
    ),
]


# ============================================================
# Appel générative Mellea -> Mathstral
# ============================================================
#
# Le docstring ci-dessous EST le prompt envoyé au modèle (c'est le
# fonctionnement du décorateur @generative de Mellea : docstring ->
# prompt, type hints -> schéma de sortie). Il reprend les règles du
# MATH_PROMPT d'origine, plus les deux garde-fous validés ci-dessus.

@generative
def clean_formula_with_mellea(formula: str) -> CleanedFormula:
    """You are an expert in mathematics and LaTeX.

    Clean the mathematical formula extracted from a scientific PDF.

    Rules:
    - Return valid LaTeX only.
    - Preserve the mathematical meaning exactly.
    - Do not invent missing symbols or extra words.
    - Do not change variables, indexes, coefficients, or equation numbers.
    - Remove OCR artifacts.
    - Remove misplaced '&' characters.
    - Fix malformed LaTeX commands.
    - Fix broken spaces around subscripts and superscripts.
    - Correct malformed \\frac, \\sum, \\epsilon, etc.
    - \\left and \\right must always be followed by a valid, escaped
      delimiter (\\{, \\}, (, ), [, ], |, ., <, >). Never leave a bare
      '{' or '}' right after \\left or \\right.
    - Never insert a backslash followed by a space and then a plain
      word (e.g. "\\ tual", "\\ ved"); that is not valid LaTeX.
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

    Another example (unescaped \\left/\\right delimiters and a
    hallucinated word to remove):

    Input:
    version \\ ved \\ L_{Seg} = \\frac{1}{N}\\sum_{i=1}^{N}\\left{-\\log p_{l_i, i}\\right} \\ tual \\ (4)

    Output latex field:
    L_{Seg} = \\frac{1}{N}\\sum_{i=1}^{N}\\left\\{-\\log p_{l_i, i}\\right\\} \\tag{4}
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

        session = None
        try:
            session = build_session()
        except Exception as exc:
            logger.warning(f"Could not initialize Mellea/Ollama session: {exc}")

        if session is not None:
            try:
                res = clean_formula_with_mellea(
                    session,
                    formula=formula,
                    requirements=LATEX_REQUIREMENTS,
                    strategy=RejectionSamplingStrategy(loop_budget=MELLEA_LOOP_BUDGET),
                )
                raw_latex = res.latex

                if _BAD_DELIMITER_RE.search(raw_latex) or _STRAY_TOKEN_RE.search(raw_latex):
                    logger.warning(
                        "Mellea/Mathstral output still malformed after "
                        f"{MELLEA_LOOP_BUDGET} attempts for formula={formula!r}; "
                        "applying local regex repair."
                    )

                return clean_latex(raw_latex)

            except Exception as exc:
                logger.warning(f"Mellea/Mathstral error on formula={formula!r}: {exc}")

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