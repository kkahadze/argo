#!/usr/bin/env python3
"""Build Argo-compatible runtime data files for the Svan translator pack."""
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READY_DIR = REPO_ROOT / "output" / "svan" / "ready"
DEFAULT_PRIVATE_DATA_DIR = REPO_ROOT / "argo" / "private_data" / "svan"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _write_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_master_lexicon(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("headword", "headword_raw", "translation"))
        writer.writeheader()


def _build_context_blocks(pair_rows: list[dict[str, str]]) -> list[str]:
    blocks: list[str] = []
    for row in pair_rows:
        svan = (row.get("svan_text") or "").strip()
        georgian = (row.get("georgian_translation") or "").strip()
        if not svan or not georgian:
            continue
        source_id = (row.get("source_id") or "titus").strip()
        blocks.append(
            "\n".join(
                (
                    f"===== SOURCE: {source_id} =====",
                    f"Svan: {svan}",
                    f"Georgian: {georgian}",
                )
            )
        )
    return blocks


def build_runtime_data(
    *,
    ready_dir: Path = DEFAULT_READY_DIR,
    private_data_dir: Path = DEFAULT_PRIVATE_DATA_DIR,
) -> dict[str, int]:
    private_data_dir.mkdir(parents=True, exist_ok=True)

    dictionary_rows = _read_tsv(ready_dir / "liparteliani_dictionary_ready.tsv")
    pair_rows = _read_tsv(ready_dir / "titus_svan_georgian_pairs_high_confidence.tsv")
    grammar_path = ready_dir / "tuite_svan_grammar_2023.txt"

    kk_rows = [
        {
            "word": (row.get("headword_svan") or "").strip(),
            "ipa": "",
            "russian_def": "",
            "georgian_def": (row.get("georgian_gloss") or "").strip(),
        }
        for row in dictionary_rows
        if (row.get("headword_svan") or "").strip()
        and (row.get("georgian_gloss") or "").strip()
    ]

    _write_master_lexicon(private_data_dir / "master-lexicon-mkhedruli.csv")
    _write_tsv(
        private_data_dir / "sentence_pairs.tsv",
        ("svan", "english"),
        [],
    )
    _write_tsv(
        private_data_dir / "gal.tsv",
        ("russian", "svan"),
        [],
    )
    _write_tsv(
        private_data_dir / "kk.tsv",
        ("word", "ipa", "russian_def", "georgian_def"),
        kk_rows,
    )
    _write_tsv(
        private_data_dir / "translation_overrides.tsv",
        ("source_language", "target_language", "source_text", "target_text"),
        [],
    )

    context_blocks = _build_context_blocks(pair_rows)
    (private_data_dir / "context_source.txt").write_text(
        "\n\n".join(context_blocks),
        encoding="utf-8",
    )

    grammar = grammar_path.read_text(encoding="utf-8") if grammar_path.exists() else ""
    (private_data_dir / "tuite.txt").write_text(grammar, encoding="utf-8")
    (private_data_dir / "tuite_compact.txt").write_text(grammar, encoding="utf-8")

    return {
        "master_lexicon_rows": 0,
        "kk_rows": len(kk_rows),
        "sentence_pair_rows": 0,
        "override_rows": 0,
        "context_blocks": len(context_blocks),
    }


def main() -> None:
    counts = build_runtime_data()
    for key, value in counts.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
