#!/usr/bin/env python3
"""Build a private Mingrelian master lexicon from a Kaikki JSONL extract.

The generated CSV is intended to live in private_data/ so it can augment the
runtime exact-lookup path without committing a large derived corpus.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_BASE = REPO_ROOT / "fastapi_app" / "data" / "master-lexicon-mkhedruli.csv"
DEFAULT_OUTPUT = REPO_ROOT / "private_data" / "master-lexicon-mkhedruli.csv"

DEFAULT_ALLOWED_POS = ("noun", "adj", "num", "adv", "intj", "verb")

BAD_GLOSS_PREFIXES = (
    "abbreviation of ",
    "accusative ",
    "added to ",
    "alternative form ",
    "alternative letter-case form ",
    "archaic form ",
    "comparative form ",
    "dative ",
    "definite ",
    "expresses ",
    "forms ",
    "genitive ",
    "inflection of ",
    "letter ",
    "locative ",
    "nominative ",
    "obsolete form ",
    "plural ",
    "suffixed to ",
    "superlative form ",
    "the first letter ",
    "used to ",
)

BAD_GLOSS_FRAGMENTS = (
    " entries with ",
    " language header",
    "written in the georgian script",
)

SPLIT_GLOSS_RE = re.compile(r"\s*(?:;|,|\bor\b)\s*", re.IGNORECASE)
PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")
WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text the same way the lookup path roughly does."""
    return WHITESPACE_RE.sub(" ", (text or "").strip()).casefold()


def compact(text: str) -> str:
    """Collapse whitespace while preserving case and script."""
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def read_master_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            headword = compact(row.get("headword") or "")
            headword_raw = compact(row.get("headword_raw") or "")
            translation = compact(row.get("translation") or "")
            if translation and (headword or headword_raw):
                rows.append(
                    {
                        "headword": headword,
                        "headword_raw": headword_raw,
                        "translation": translation,
                    }
                )
        return rows


def romanization(entry: dict) -> str:
    for form in entry.get("forms") or []:
        if "romanization" in (form.get("tags") or []) and form.get("form"):
            return compact(form["form"])
    return ""


def clean_gloss_candidates(gloss: str, *, max_chars: int) -> tuple[list[str], str | None]:
    raw = compact(gloss)
    if not raw:
        return [], "empty_gloss"

    lowered = normalize(raw)
    if len(raw) > max_chars:
        return [], "long_gloss"
    if lowered.startswith(BAD_GLOSS_PREFIXES):
        return [], "gloss_prefix"
    if any(fragment in lowered for fragment in BAD_GLOSS_FRAGMENTS):
        return [], "gloss_fragment"
    if ":" in raw:
        return [], "gloss_explanation"

    candidates: list[str] = []
    without_parens = compact(PARENTHETICAL_RE.sub("", raw))
    variants = [without_parens]

    if SPLIT_GLOSS_RE.search(without_parens):
        variants.extend(part for part in SPLIT_GLOSS_RE.split(without_parens) if part)

    seen: set[str] = set()
    for candidate in variants:
        candidate = compact(candidate.strip(" .;,:"))
        candidate_key = normalize(candidate)
        if not candidate_key or candidate_key in seen:
            continue
        if len(candidate) > max_chars:
            continue
        if len(candidate) == 1:
            continue
        if candidate_key.startswith(BAD_GLOSS_PREFIXES):
            continue
        candidates.append(candidate)
        seen.add(candidate_key)

    if not candidates:
        return [], "no_clean_gloss_candidate"
    return candidates, None


def iter_kaikki_rows(
    kaikki_path: Path,
    *,
    allowed_pos: set[str],
    max_gloss_chars: int,
) -> tuple[list[dict[str, str]], Counter]:
    rows: list[dict[str, str]] = []
    skipped: Counter = Counter()
    seen: set[tuple[str, str, str]] = set()

    with kaikki_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped["invalid_json"] += 1
                continue

            if entry.get("lang_code") != "xmf":
                skipped["non_mingrelian_lang_code"] += 1
                continue

            pos = entry.get("pos") or ""
            if pos not in allowed_pos:
                skipped[f"pos:{pos or '<none>'}"] += 1
                continue

            word = compact(entry.get("word") or "")
            word_key = normalize(word)
            if not word:
                skipped["empty_word"] += 1
                continue
            if word.startswith("-") or len(word_key) <= 1:
                skipped["affix_or_single_character"] += 1
                continue

            roman = romanization(entry)
            for sense in entry.get("senses") or []:
                glosses = sense.get("glosses") or []
                if not glosses:
                    skipped["sense_without_gloss"] += 1
                    continue

                for gloss in glosses:
                    translations, reason = clean_gloss_candidates(
                        gloss,
                        max_chars=max_gloss_chars,
                    )
                    if reason:
                        skipped[reason] += 1
                        continue

                    for translation in translations:
                        key = (normalize(word), normalize(roman), normalize(translation))
                        if key in seen:
                            skipped["duplicate_kaikki_row"] += 1
                            continue
                        seen.add(key)
                        rows.append(
                            {
                                "headword": word,
                                "headword_raw": roman,
                                "translation": translation,
                            }
                        )

    skipped["kaikki_rows_kept"] = len(rows)
    return rows, skipped


def dedupe_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        headword = compact(row.get("headword") or "")
        headword_raw = compact(row.get("headword_raw") or "")
        translation = compact(row.get("translation") or "")
        if not translation or not (headword or headword_raw):
            continue

        key = (normalize(headword or headword_raw), normalize(translation))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "headword": headword,
                "headword_raw": headword_raw,
                "translation": translation,
            }
        )

    return deduped


def write_master_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("headword", "headword_raw", "translation"))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append safe Kaikki/Wiktionary Mingrelian rows to a private master lexicon.",
    )
    parser.add_argument(
        "kaikki_jsonl",
        type=Path,
        help="Path to kaikki.org-dictionary-Mingrelian.jsonl",
    )
    parser.add_argument(
        "--base",
        type=Path,
        help=(
            "Base master lexicon to preserve before appending Kaikki rows. "
            "Defaults to the output file when it exists, otherwise the public data path."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allowed-pos",
        default=",".join(DEFAULT_ALLOWED_POS),
        help="Comma-separated POS list to import.",
    )
    parser.add_argument("--max-gloss-chars", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed_pos = {part.strip() for part in args.allowed_pos.split(",") if part.strip()}
    base_path = args.base or (args.output if args.output.exists() else DEFAULT_PUBLIC_BASE)

    base_rows = read_master_rows(base_path)
    kaikki_rows, skipped = iter_kaikki_rows(
        args.kaikki_jsonl,
        allowed_pos=allowed_pos,
        max_gloss_chars=args.max_gloss_chars,
    )
    combined_rows = dedupe_rows([*base_rows, *kaikki_rows])
    added_rows = len(combined_rows) - len(dedupe_rows(base_rows))

    if not args.dry_run:
        write_master_rows(args.output, combined_rows)

    print(
        json.dumps(
            {
                "base": str(base_path),
                "output": str(args.output),
                "dry_run": args.dry_run,
                "allowed_pos": sorted(allowed_pos),
                "base_rows": len(base_rows),
                "candidate_kaikki_rows": len(kaikki_rows),
                "added_rows_after_dedupe": added_rows,
                "combined_rows": len(combined_rows),
                "skipped": dict(skipped.most_common()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
