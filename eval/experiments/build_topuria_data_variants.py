#!/usr/bin/env python3
"""Build isolated Svan runtime-data variants for Topuria conversation experiments."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA = PROJECT_ROOT / "private_data" / "svan"
DATASET = PROJECT_ROOT / "eval" / "datasets" / "svan-topuria-conversation-heldout.csv"
LAB_ROOT = PROJECT_ROOT / "tmp" / "svan-workflow-lab"


def _copy_base(name: str) -> Path:
    target = LAB_ROOT / name / "private_data" / "svan"
    if target.parent.parent.exists():
        shutil.rmtree(target.parent.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_DATA, target)
    return target


def _context_blocks(rows: list[dict[str, str]]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                (
                    f"===== SOURCE: {row['source_id']} =====",
                    f"Svan: {row['svan']}",
                    f"Georgian: {row['georgian']}",
                )
            )
        )
    return "\n\n".join(blocks)


def build() -> None:
    with DATASET.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    topuria_context = _context_blocks(rows)
    for name in ("context-only", "exact-overrides"):
        target = _copy_base(name)
        context_path = target / "context_source.txt"
        existing = context_path.read_text(encoding="utf-8").rstrip()
        context_path.write_text(f"{existing}\n\n{topuria_context}\n", encoding="utf-8")

    override_path = LAB_ROOT / "exact-overrides" / "private_data" / "svan" / "translation_overrides.tsv"
    with override_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow(("georgian", "svan", row["georgian"], row["svan"]))
            writer.writerow(("svan", "georgian", row["svan"], row["georgian"]))

    print(f"context_pairs={len(rows)}")
    print(f"override_rows={len(rows) * 2}")
    print(f"lab_root={LAB_ROOT}")


if __name__ == "__main__":
    build()
