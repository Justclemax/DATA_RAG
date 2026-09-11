# PDF → Markdown pipeline (Docling + Mellea)

Refactorisation du script original en plusieurs fichiers, avec le
nettoyage des formules LaTeX (Mathstral) migré vers
[Mellea](https://mellea.ai/) au lieu d'un appel `requests.post` brut
vers l'API Ollama.

## Structure

| Fichier | Rôle |
|---|---|
| `config.py` | Variables d'environnement, noms de modèles, chemins par défaut |
| `prompts.py` | Prompt du VLM (Granite Vision) |
| `latex_cleaning.py` | Nettoyage regex local + nettoyage sémantique via Mellea/Mathstral |
| `vlm_pipeline.py` | Configuration Docling (OCR, tables, images, formules) |
| `document_processor.py` | Orchestration : PDF → Markdown → images → formules |
| `../main.py` | Point d'entrée CLI |

## Ce qui change par rapport au script d'origine

L'appel Mathstral (`call_mathstral`, via `requests.post` sur
`/api/chat`) est remplacé par une fonction `@generative` Mellea
(`clean_formula_with_mellea` dans `latex_cleaning.py`) :

- Le docstring de la fonction **est** le prompt (mêmes règles que le
  `MATH_PROMPT` d'origine).
- La sortie est contrainte par un schéma Pydantic (`CleanedFormula`),
  donc toujours structurée (`result.latex`) au lieu de texte brut à
  re-parser.
- Le repli sur le nettoyage regex local est conservé si Ollama/le
  modèle est indisponible ou échoue sur une formule.

La description d'images (Granite Vision) reste gérée nativement par
Docling via `PictureDescriptionApiOptions` (`vlm_pipeline.py`) — elle
n'est pas passée par Mellea, Docling appelant directement l'endpoint
OpenAI-compatible d'Ollama en interne.

## Installation

```bash
pip install -r requirements.txt
# ou : uv pip install -r requirements.txt
cp .env.example .env
```

Assure-toi qu'Ollama tourne et que les modèles sont disponibles :

```bash
ollama pull granite3.2-vision:latest
ollama pull mathstral:latest
```

## Exécution

```bash
python main.py --pdf ../data/0000721.pdf --output-dir ../data/output
```

## Parallélisation du nettoyage des formules

Le nettoyage des formules LaTeX ($$...$$) est parallélisé par défaut
pour accélérer le traitement des documents contenant de nombreuses
formules. Le module `src/latex_cleaning.py` utilise un ProcessPoolExecutor
via `src/parallel_executor.py` : chaque formule est traitée dans un
process séparé (tentative d'appel à Mathstral/Mellea puis repli sur
le nettoyage regex local en cas d'échec).

Options et configuration :

- L'implémentation utilise ProcessPoolExecutor (CPU-bound) et est
  activée par défaut.
- Pour afficher une barre de progression, installez `tqdm` (optionnel).
- Les paramètres (nombre de workers, chunksize, affichage) peuvent être
  ajustés en modifiant la fonction `process_formulas` dans
  `src/latex_cleaning.py` (arguments `max_workers`, `chunksize`,
  `show_progress`).

Exemple (si vous modifiez `main.py` ou appelez directement la fonction) :

```py
from src.latex_cleaning import process_formulas
cleaned = process_formulas(content, use_parallel=True, max_workers=4, show_progress=True)
```

