#!/usr/bin/env python3
"""Extract the 2023 Tuite Svan grammar with table-preserving layout."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "http://www.mapageweb.umontreal.ca/tuitekj/publications/2023-Svan-grammar-Tuite.pdf"
DEFAULT_PDF_PATH = REPO_ROOT / "output" / "svan" / "sources" / "tuite-svan-grammar-2023.pdf"
DEFAULT_TEXT_PATH = REPO_ROOT / "output" / "svan" / "tmp" / "tuite-svan-grammar-2023-layout.txt"
DEFAULT_REPORT_PATH = REPO_ROOT / "output" / "svan" / "tmp" / "tuite-svan-grammar-2023-extraction.json"


def _extraction_metrics(text: str) -> dict[str, int]:
    metrics = {
        "characters": len(text),
        "pages": text.count("\f"),
        "legacy_pound_sign_count": text.count("£"),
        "replacement_character_count": text.count("\ufffd"),
    }
    if "The Svan language." not in text or "23 August 2023" not in text:
        raise ValueError("Extraction is not the 23 August 2023 Tuite Svan grammar.")
    if metrics["pages"] != 97:
        raise ValueError(f"Expected 97 extracted PDF pages, found {metrics['pages']}.")
    if metrics["legacy_pound_sign_count"] or metrics["replacement_character_count"]:
        raise ValueError("Extraction contains corrupted legacy or replacement characters.")
    return metrics


def extract_grammar(
    *,
    pdf_path: Path = DEFAULT_PDF_PATH,
    text_path: Path = DEFAULT_TEXT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = text_path.with_suffix(text_path.suffix + ".tmp")
    try:
        subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), str(temporary_path)],
            check=True,
        )
        text = temporary_path.read_text(encoding="utf-8")
        metrics = _extraction_metrics(text)
        temporary_path.replace(text_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report: dict[str, object] = {
        "source_url": SOURCE_URL,
        "pdf_path": str(pdf_path),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "text_path": str(text_path),
        "method": "pdftotext -layout -enc UTF-8",
        **metrics,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    print(json.dumps(extract_grammar(pdf_path=args.pdf, text_path=args.text_output, report_path=args.report_output), indent=2))


if __name__ == "__main__":
    main()
