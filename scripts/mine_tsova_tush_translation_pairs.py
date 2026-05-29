#!/usr/bin/env python3
"""Mine high-confidence Tsova-Tush translation pairs from prepared assets."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READY_DIR = REPO_ROOT / "output" / "tsova-tush" / "ready"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "tsova-tush" / "pairs"
PAIR_FIELDS = (
    "source_id",
    "source_name",
    "pair_type",
    "batsbi_text",
    "batsbi_text_tokenized",
    "georgian_translation",
    "english_translation",
    "confidence",
    "notes",
)


@dataclass(frozen=True)
class PairMiningInputs:
    ready_dir: Path
    output_dir: Path


def _write_tsv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PAIR_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _dedupe_pairs(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            row.get("pair_type", ""),
            row.get("batsbi_text", "").strip(),
            (row.get("georgian_translation") or row.get("english_translation") or "").strip(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_titus_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            batsbi = (row.get("batsbi_text") or "").strip()
            georgian = (row.get("georgian_translation") or "").strip()
            if not batsbi or not georgian:
                continue
            rows.append(
                {
                    "source_id": f"titus:{row.get('lemma', '')}:{row.get('lemma_number', '')}:{row.get('example_number', '')}",
                    "source_name": "titus_examples",
                    "pair_type": "batsbi_georgian",
                    "batsbi_text": batsbi,
                    "batsbi_text_tokenized": (row.get("batsbi_text_tokenized") or batsbi).strip(),
                    "georgian_translation": georgian,
                    "english_translation": "",
                    "confidence": "high",
                    "notes": row.get("source_url", ""),
                }
            )
    return rows


def _load_optional_worker_pairs(inputs: PairMiningInputs) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    try:
        from src.tsova_tush.text_pair_extraction import iter_numbered_translation_pairs
    except ImportError:
        iter_numbered_translation_pairs = None

    try:
        from src.tsova_tush.grammar_pair_extraction import extract_grammar_translation_pairs
    except ImportError:
        extract_grammar_translation_pairs = None

    try:
        from src.tsova_tush.part4_story_pair_extraction import (
            extract_part4_story_translation_pairs,
        )
    except ImportError:
        extract_part4_story_translation_pairs = None

    if iter_numbered_translation_pairs is not None:
        for source_name, filename in (
            ("tsovatush_texts_part1_2009", "tsovatush_texts_part1_2009.txt"),
            ("tsovatush_texts_part4_2017", "tsovatush_texts_part4_2017.txt"),
        ):
            path = inputs.ready_dir / filename
            if not path.exists():
                continue
            for candidate in iter_numbered_translation_pairs(
                path.read_text(encoding="utf-8"),
                source_id=source_name,
            ):
                rows.append(
                    {
                        "source_id": candidate.source_id,
                        "source_name": source_name,
                        "pair_type": "batsbi_georgian",
                        "batsbi_text": candidate.batsbi_text,
                        "batsbi_text_tokenized": candidate.batsbi_text,
                        "georgian_translation": candidate.georgian_translation,
                        "english_translation": candidate.english_translation or "",
                        "confidence": "high" if candidate.confidence >= 0.95 else "review",
                        "notes": candidate.notes,
                    }
                )

    if extract_grammar_translation_pairs is not None:
        rows.extend(
            extract_grammar_translation_pairs(
                inputs.ready_dir / "holisky_gagua_1994.txt",
                source_name="holisky_gagua_1994",
            )
        )

    if extract_part4_story_translation_pairs is not None:
        rows.extend(
            extract_part4_story_translation_pairs(
                inputs.ready_dir / "tsovatush_texts_part4_2017.txt",
                source_name="tsovatush_texts_part4_2017",
            )
        )

    return rows


def mine_translation_pairs(inputs: PairMiningInputs) -> dict[str, object]:
    titus_pairs = _load_titus_pairs(inputs.ready_dir / "titus_examples.tsv")
    worker_pairs = _load_optional_worker_pairs(inputs)
    all_pairs = _dedupe_pairs([*titus_pairs, *worker_pairs])

    high_confidence = [row for row in all_pairs if row.get("confidence") == "high"]
    review = [row for row in all_pairs if row.get("confidence") != "high"]

    _write_tsv(inputs.output_dir / "pairs_high_confidence.tsv", high_confidence)
    _write_tsv(inputs.output_dir / "pairs_review.tsv", review)
    _write_tsv(inputs.output_dir / "pairs_all.tsv", all_pairs)

    manifest: dict[str, object] = {
        "counts": {
            "high_confidence_pairs": len(high_confidence),
            "review_pairs": len(review),
            "all_pairs": len(all_pairs),
        },
        "outputs": {
            "pairs_high_confidence_tsv": "pairs_high_confidence.tsv",
            "pairs_review_tsv": "pairs_review.tsv",
            "pairs_all_tsv": "pairs_all.tsv",
        },
    }
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    (inputs.output_dir / "pair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = mine_translation_pairs(
        PairMiningInputs(
            ready_dir=args.ready_dir,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
