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

## Installation

Utilise `uv` pour créer un environnement propre et installer les dépendances rapidement :

```bash
# 1) installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) créer l'environnement virtuel
uv venv

# 3) activer l'environnement
source .venv/bin/activate

# 4) installer les dépendances du projet
uv pip install -r requirements.txt

# 5) copier le fichier d'environnement
cp .env.example .env
```

Si tu préfères utiliser directement le gestionnaire Python du projet sans créer d'environnement séparé :

```bash
uv pip install -r requirements.txt
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

