#!/usr/bin/env python3
"""Build a balanced TITUS Svan-Georgian Promptfoo evaluation slice."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "output" / "svan" / "ready" / "titus_svan_georgian_pairs_high_confidence.tsv"
OUTPUT_PATH = Path(__file__).resolve().parent / "datasets" / "svan-titus-balanced-blocks.csv"
PILOT_OUTPUT_PATH = Path(__file__).resolve().parent / "datasets" / "svan-titus-balanced-blocks-pilot.csv"

TARGET_PER_BUCKET = 16
BUCKETS = (
    ("short", 0, 80),
    ("medium", 81, 220),
    ("long", 221, 10_000),
)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def bucket_for_length(length: int) -> str | None:
    for name, minimum, maximum in BUCKETS:
        if minimum <= length <= maximum:
            return name
    return None


def select_balanced_rows(rows: list[dict[str, str]], target_per_bucket: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []

    for bucket_name, _, _ in BUCKETS:
        candidates = [
            row
            for row in rows
            if bucket_for_length(len(normalize_whitespace(row["svan_text"]))) == bucket_name
        ]
        candidates.sort(
            key=lambda row: (
                len(normalize_whitespace(row["svan_text"])),
                row["source_id"],
            )
        )

        if len(candidates) < target_per_bucket:
            raise ValueError(f"Need {target_per_bucket} {bucket_name} rows, found {len(candidates)}")

        if target_per_bucket == 1:
            selected.extend(candidates[:1])
            continue

        stride = (len(candidates) - 1) / (target_per_bucket - 1)
        picked_indexes = []
        used = set()
        for position in range(target_per_bucket):
            index = round(position * stride)
            while index in used and index + 1 < len(candidates):
                index += 1
            if index in used:
                index = next(idx for idx in range(len(candidates)) if idx not in used)
            used.add(index)
            picked_indexes.append(index)

        selected.extend(candidates[index] for index in picked_indexes)

    return selected


def write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_id",
        "length_bucket",
        "svan",
        "georgian",
        "confidence",
        "notes",
        "svan_source_url",
        "georgian_source_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            svan = normalize_whitespace(row["svan_text"])
            georgian = normalize_whitespace(row["georgian_translation"])
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "length_bucket": bucket_for_length(len(svan)),
                    "svan": svan,
                    "georgian": georgian,
                    "confidence": row["confidence"],
                    "notes": row["notes"],
                    "svan_source_url": row["svan_source_url"],
                    "georgian_source_url": row["georgian_source_url"],
                }
            )


def build_dataset() -> tuple[int, int]:
    with SOURCE_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    selected = select_balanced_rows(rows, TARGET_PER_BUCKET)
    pilot_selected = select_balanced_rows(rows, 4)
    write_dataset(OUTPUT_PATH, selected)
    write_dataset(PILOT_OUTPUT_PATH, pilot_selected)

    return len(selected), len(pilot_selected)


if __name__ == "__main__":
    full_rows, pilot_rows = build_dataset()
    print(f"rows={full_rows}")
    print(f"output={OUTPUT_PATH}")
    print(f"pilot_rows={pilot_rows}")
    print(f"pilot_output={PILOT_OUTPUT_PATH}")
