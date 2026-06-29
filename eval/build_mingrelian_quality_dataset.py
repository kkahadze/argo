#!/usr/bin/env python3
"""Build one four-direction Promptfoo dataset from lesson-note triples."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
SOURCE = EVAL_DIR / "datasets" / "notion-mingrelian-lesson-notes-triples-product-script.csv"
DESTINATION = EVAL_DIR / "datasets" / "mingrelian-quality-four-way.csv"

DIRECTIONS = (
    ("english-to-mingrelian", "english", "mingrelian"),
    ("georgian-to-mingrelian", "georgian", "mingrelian"),
    ("mingrelian-to-english", "mingrelian", "english"),
    ("mingrelian-to-georgian", "mingrelian", "georgian"),
)

FIELDNAMES = (
    "case_id",
    "direction",
    "source_language",
    "target_language",
    "input_text",
    "reference_text",
    "row_type",
    "english",
    "georgian",
    "mingrelian",
    "acceptable_target_variants",
    "acceptable_mingrelian_variants",
    "source_page",
    "confidence",
    "notes",
    "mingrelian_source_transcription",
)


def _row_type(row: dict[str, str]) -> str:
    texts = (row["english"], row["georgian"], row["mingrelian"])
    return "lexical" if all(len(text.split()) == 1 for text in texts) else "sentence"


def build(source: Path = SOURCE, destination: Path = DESTINATION) -> int:
    with source.open(newline="", encoding="utf-8") as source_file:
        source_rows = list(csv.DictReader(source_file))

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(source_rows, start=1):
        row_type = _row_type(row)
        for direction, source_language, target_language in DIRECTIONS:
            output_rows.append(
                {
                    "case_id": f"lesson_notes_{index:02d}:{direction}",
                    "direction": direction,
                    "source_language": source_language,
                    "target_language": target_language,
                    "input_text": row[source_language],
                    "reference_text": row[target_language],
                    "row_type": row_type,
                    "english": row["english"],
                    "georgian": row["georgian"],
                    "mingrelian": row["mingrelian"],
                    "acceptable_target_variants": (
                        row.get("acceptable_mingrelian_variants", "")
                        if target_language == "mingrelian"
                        else ""
                    ),
                    "acceptable_mingrelian_variants": row.get(
                        "acceptable_mingrelian_variants",
                        "",
                    ),
                    "source_page": row.get("source_page", ""),
                    "confidence": row.get("confidence", ""),
                    "notes": row.get("notes", ""),
                    "mingrelian_source_transcription": row.get(
                        "mingrelian_source_transcription",
                        "",
                    ),
                }
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as destination_file:
        writer = csv.DictWriter(destination_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(
            f"private lesson-note dataset is missing: {args.source}. "
            "Run this in the live argo workspace or pass --source."
        )
    count = build(args.source, args.destination)
    print(f"Wrote {count} rows to {args.destination}")
