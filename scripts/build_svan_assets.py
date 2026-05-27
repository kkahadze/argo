#!/usr/bin/env python3
"""Build first-pass translator-ready Svan ingestion assets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.svan.titus_parallel_extraction import (
    TitusAlignedPair,
    align_parallel_blocks,
    build_part_review_rows,
    parse_titus_link_blocks,
    parse_titus_text_lines,
    write_text_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "svan" / "ready"
DEFAULT_GRAMMAR_TEXT = REPO_ROOT / "output" / "svan" / "tmp" / "tuite-svan-grammar-2023-layout.txt"
DEFAULT_GRAMMAR_PDF = REPO_ROOT / "output" / "svan" / "sources" / "tuite-svan-grammar-2023.pdf"
DEFAULT_GRAMMAR_REPORT = REPO_ROOT / "output" / "svan" / "tmp" / "tuite-svan-grammar-2023-extraction.json"
DEFAULT_LIPARTELIANI_SOURCE = (
    REPO_ROOT / "output" / "svan" / "sources" / "liparteliani-svan-georgian-fulltext.txt"
)
DEFAULT_LIPARTELIANI_READY = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_ready.tsv"
DEFAULT_LIPARTELIANI_REVIEW = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_review.tsv"
DEFAULT_LIPARTELIANI_STATS = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_stats.tsv"
SVAN_BASE_URL = "https://titus.uni-frankfurt.de/texte/etca/cauc/svan/spto2"
GEORGIAN_BASE_URL = "https://titus.uni-frankfurt.de/texte/etca/cauc/svan/spto2g"


def _fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "margo-svan-ingestion/1.0"})
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def _write_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _aligned_rows(pairs: list[TitusAlignedPair]) -> list[dict[str, str]]:
    return [
        {
            "source_id": pair.source_id,
            "source_name": "titus_svan_prose_texts_ii",
            "pair_type": "svan_georgian_block",
            "svan_text": pair.svan_text,
            "georgian_translation": pair.georgian_translation,
            "english_translation": "",
            "confidence": pair.confidence,
            "notes": pair.notes,
            "svan_source_url": pair.svan_source_url,
            "georgian_source_url": pair.georgian_source_url,
        }
        for pair in pairs
    ]


def _read_liparteliani_counts(stats_path: Path) -> dict[str, int]:
    if not stats_path.exists():
        return {}

    with stats_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        raw_counts = {row["metric"]: row["value"] for row in reader if row.get("metric")}

    counts: dict[str, int] = {}
    for manifest_key, stats_key in (
        ("liparteliani_ready_rows", "ready_rows"),
        ("liparteliani_review_rows", "review_rows"),
    ):
        raw_value = raw_counts.get(stats_key)
        if raw_value is None:
            continue
        try:
            counts[manifest_key] = int(raw_value)
        except ValueError:
            continue
    return counts


def build_assets(*, output_dir: Path, grammar_text: Path, max_part: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_svan_lines = []
    all_georgian_lines = []
    all_svan_blocks = []
    all_georgian_blocks = []
    fetch_failures: list[dict[str, str]] = []

    for part in range(1, max_part + 1):
        svan_url = f"{SVAN_BASE_URL}/spto2{part:03d}.htm"
        georgian_url = f"{GEORGIAN_BASE_URL}/spto2{part:03d}.htm"

        try:
            svan_html = _fetch(svan_url)
            georgian_html = _fetch(georgian_url)
        except (HTTPError, URLError, TimeoutError) as exc:
            fetch_failures.append(
                {
                    "part": str(part),
                    "svan_source_url": svan_url,
                    "georgian_source_url": georgian_url,
                    "error": str(exc),
                }
            )
            continue

        all_svan_lines.extend(parse_titus_text_lines(svan_html, source_url=svan_url))
        all_georgian_lines.extend(parse_titus_text_lines(georgian_html, source_url=georgian_url))
        all_svan_blocks.extend(parse_titus_link_blocks(svan_html, source_url=svan_url))
        all_georgian_blocks.extend(parse_titus_link_blocks(georgian_html, source_url=georgian_url))

    aligned_pairs, line_review_rows = align_parallel_blocks(all_svan_blocks, all_georgian_blocks)
    part_review_rows = build_part_review_rows(all_svan_lines, all_georgian_lines)

    aligned_path = output_dir / "titus_svan_georgian_pairs_high_confidence.tsv"
    review_path = output_dir / "titus_svan_georgian_pairs_review.tsv"
    structure_review_path = output_dir / "titus_alignment_review.tsv"
    fetch_failures_path = output_dir / "titus_fetch_failures.tsv"
    svan_snapshot_path = output_dir / "titus_svan_prose_texts_ii.txt"
    georgian_snapshot_path = output_dir / "titus_svan_prose_texts_ii_georgian_translation.txt"
    grammar_output_path = output_dir / "tuite_svan_grammar_2023.txt"
    manifest_path = output_dir / "build_manifest.json"

    _write_tsv(
        aligned_path,
        (
            "source_id",
            "source_name",
            "pair_type",
            "svan_text",
            "georgian_translation",
            "english_translation",
            "confidence",
            "notes",
            "svan_source_url",
            "georgian_source_url",
        ),
        _aligned_rows(aligned_pairs),
    )
    _write_tsv(
        review_path,
        (
            "source_id",
            "pair_type",
            "svan_text",
            "georgian_translation",
            "confidence",
            "notes",
        ),
        part_review_rows,
    )
    _write_tsv(
        structure_review_path,
        ("part", "page", "issue", "svan_line_count", "georgian_line_count"),
        line_review_rows,
    )
    _write_tsv(
        fetch_failures_path,
        ("part", "svan_source_url", "georgian_source_url", "error"),
        fetch_failures,
    )

    write_text_snapshot(svan_snapshot_path, all_svan_lines)
    write_text_snapshot(georgian_snapshot_path, all_georgian_lines)

    if grammar_text.exists():
        grammar_output_path.write_text(grammar_text.read_text(encoding="utf-8"), encoding="utf-8")

    manifest_sources = {
        "titus_svan_prose_texts_ii": {
            "svan_base_url": SVAN_BASE_URL,
            "georgian_base_url": GEORGIAN_BASE_URL,
            "parts_requested": max_part,
        },
        "tuite_svan_grammar_2023": {
            "source_pdf_path": str(DEFAULT_GRAMMAR_PDF),
            "input_text_path": str(grammar_text),
            "output_text_path": str(grammar_output_path),
            "extraction_report_path": str(DEFAULT_GRAMMAR_REPORT),
        },
    }
    manifest_counts = {
        "svan_line_blocks": len(all_svan_lines),
        "georgian_line_blocks": len(all_georgian_lines),
        "svan_alignment_blocks": len(all_svan_blocks),
        "georgian_alignment_blocks": len(all_georgian_blocks),
        "high_confidence_pairs": len(aligned_pairs),
        "review_part_pairs": len(part_review_rows),
        "alignment_review_rows": len(line_review_rows),
        "fetch_failures": len(fetch_failures),
    }
    manifest_outputs = {
        "high_confidence_pairs": str(aligned_path),
        "review_pairs": str(review_path),
        "alignment_review": str(structure_review_path),
        "fetch_failures": str(fetch_failures_path),
        "svan_context": str(svan_snapshot_path),
        "georgian_context": str(georgian_snapshot_path),
        "grammar_context": str(grammar_output_path),
    }

    if (
        DEFAULT_LIPARTELIANI_SOURCE.exists()
        and DEFAULT_LIPARTELIANI_READY.exists()
        and DEFAULT_LIPARTELIANI_REVIEW.exists()
        and DEFAULT_LIPARTELIANI_STATS.exists()
    ):
        manifest_sources["liparteliani_svan_georgian_dictionary"] = {
            "input_text_path": str(DEFAULT_LIPARTELIANI_SOURCE),
            "ready_output_path": str(DEFAULT_LIPARTELIANI_READY),
            "review_output_path": str(DEFAULT_LIPARTELIANI_REVIEW),
            "stats_output_path": str(DEFAULT_LIPARTELIANI_STATS),
        }
        manifest_counts.update(_read_liparteliani_counts(DEFAULT_LIPARTELIANI_STATS))
        manifest_outputs.update(
            {
                "liparteliani_dictionary_ready": str(DEFAULT_LIPARTELIANI_READY),
                "liparteliani_dictionary_review": str(DEFAULT_LIPARTELIANI_REVIEW),
                "liparteliani_dictionary_stats": str(DEFAULT_LIPARTELIANI_STATS),
            }
        )

    manifest = {
        "sources": manifest_sources,
        "counts": manifest_counts,
        "outputs": manifest_outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grammar-text", type=Path, default=DEFAULT_GRAMMAR_TEXT)
    parser.add_argument("--max-part", type=int, default=36)
    args = parser.parse_args()

    manifest = build_assets(
        output_dir=args.output_dir,
        grammar_text=args.grammar_text,
        max_part=args.max_part,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
