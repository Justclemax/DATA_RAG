"""
Point d'entrée du pipeline.

Usage:
    python main.py
    python main.py --pdf ../data/mon_papier.pdf --output-dir ../data/output
"""

import argparse
from pathlib import Path

from config import DEFAULT_INPUT_PDF, DEFAULT_OUTPUT_DIR
from document_processor import process_document


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
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    file_path: Path = args.pdf
    output_dir: Path = args.output_dir

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    md_content = process_document(file_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_path.stem}.md"

    output_file.write_text(md_content, encoding="utf-8")

    print(f"\nDocument processed successfully: {output_file}")


if __name__ == "__main__":
    main()
