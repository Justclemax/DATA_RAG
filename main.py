"""
Point d'entrée du pipeline.

Usage:
    python main.py
    python main.py --pdf ../data/mon_papier.pdf --output-dir ../data/output
"""

import argparse
from pathlib import Path

from src.processing_data.config import DEFAULT_INPUT_PDF, DEFAULT_OUTPUT_DIR
from src.processing_data.document_processor import process_document
from loguru import logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convertit un PDF scientifique en Markdown (Docling + Mellea/Ollama)."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_INPUT_PDF,
        help=f"Chemin du PDF à traiter (défaut : {DEFAULT_INPUT_PDF})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie (défaut : {DEFAULT_OUTPUT_DIR})",
    )

    # Parallelization options for LaTeX cleaning
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        help="Forcer la parallélisation des nettoyages de formules (ProcessPoolExecutor).",
    )
    group.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Désactiver la parallélisation des nettoyages de formules (séquentiel).",
    )
    parser.set_defaults(parallel=True)

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de workers ProcessPoolExecutor (par défaut: nombre de CPUs).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Taille des lots envoyés aux workers (chunksize) pour améliorer la performance.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Afficher une barre de progression (si tqdm installé) lors du nettoyage des formules.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    file_path: Path = args.pdf
    output_dir: Path = args.output_dir

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    md_content = process_document(
        file_path,
        use_parallel=bool(getattr(args, "parallel", True)),
        max_workers=args.workers,
        chunksize=args.chunksize,
        show_progress=args.show_progress,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_path.stem}.md"

    output_file.write_text(md_content, encoding="utf-8")

    logger.info(f"\nDocument processed successfully: {output_file}")


if __name__ == "__main__":
    main()
