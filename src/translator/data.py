#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.translator.text_utils import _normalize_lookup_value

REPO_ROOT = Path(__file__).resolve().parents[2]


def _master_lexicon_enabled() -> bool:
    """Allow master lexicon ablations without editing the core pipeline."""
    value = os.getenv("ARGO_ENABLE_MASTER_LEXICON", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _get_data_path(filename: str) -> str:
    """Get the path to a data file, checking multiple possible locations."""
    configured_data_dir = os.getenv("ARGO_DATA_DIR")
    candidate_dirs = []
    if configured_data_dir:
        candidate_dirs.append(Path(configured_data_dir).expanduser())

    candidate_dirs.extend(
        [
            REPO_ROOT / "private_data",
            REPO_ROOT / "fastapi_app" / "data",
            REPO_ROOT / "data",
            REPO_ROOT / "notebooks",
            REPO_ROOT / "notebooks" / "dicts",
            REPO_ROOT / "eval" / "datasets",
        ]
    )

    for data_dir in candidate_dirs:
        candidate = data_dir / filename
        if candidate.exists():
            return str(candidate)

    return str(REPO_ROOT / "private_data" / filename)


def _data_file_cache_key(filename: str) -> tuple[str, Optional[int]]:
    """Build a cache key that invalidates when a data file changes on disk."""
    file_path = _get_data_path(filename)
    try:
        mtime_ns = Path(file_path).stat().st_mtime_ns
    except FileNotFoundError:
        return file_path, None
    return file_path, mtime_ns


def _is_header_row(parts: list[str], expected_header: tuple[str, ...]) -> bool:
    """Return True when a TSV row exactly matches the expected header cells."""
    if len(parts) < len(expected_header):
        return False
    normalized_parts = tuple(
        _normalize_lookup_value(part).lstrip("\ufeff")
        for part in parts[:len(expected_header)]
    )
    return normalized_parts == expected_header


@lru_cache(maxsize=4)
def _load_master_lexicon_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[tuple[str, str, str], ...]:
    """Load and cache the master lexicon while still reloading when the file changes."""
    if mtime_ns is None:
        return ()

    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            headword = (row.get("headword") or "").strip()
            headword_raw = (row.get("headword_raw") or "").strip()
            translation = (row.get("translation") or "").strip()
            if not translation or not (headword or headword_raw):
                continue
            rows.append((headword, headword_raw, translation))
    return tuple(rows)


def _load_master_lexicon_rows() -> tuple[tuple[str, str, str], ...]:
    """Load the master lexicon used for exact Mingrelian ↔ English matches."""
    if not _master_lexicon_enabled():
        return ()
    return _load_master_lexicon_rows_cached(*_data_file_cache_key("master-lexicon-mkhedruli.csv"))


@lru_cache(maxsize=4)
def _load_sentence_pairs_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[tuple[str, str], ...]:
    """Load and cache sentence_pairs.tsv while still picking up file edits."""
    if mtime_ns is None:
        return ()

    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if _is_header_row(parts, ("mingrelian", "english")):
                continue
            if len(parts) < 2:
                continue
            mingrelian = parts[0].strip()
            english = parts[1].strip()
            if mingrelian and english:
                rows.append((mingrelian, english))
    return tuple(rows)


def _load_sentence_pairs_rows() -> tuple[tuple[str, str], ...]:
    """Load sentence pairs for extractive Mingrelian ↔ English lookups."""
    return _load_sentence_pairs_rows_cached(*_data_file_cache_key("sentence_pairs.tsv"))


@lru_cache(maxsize=4)
def _load_gal_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[tuple[str, str], ...]:
    """Load and cache gal.tsv while still picking up file edits."""
    if mtime_ns is None:
        return ()

    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for parts in reader:
            if _is_header_row(parts, ("russian", "mingrelian")):
                continue
            if len(parts) < 2:
                continue
            russian = parts[0].strip()
            mingrelian = parts[1].strip()
            if russian and mingrelian:
                rows.append((russian, mingrelian))
    return tuple(rows)


def _load_gal_rows() -> tuple[tuple[str, str], ...]:
    """Load Russian ↔ Mingrelian dictionary rows."""
    return _load_gal_rows_cached(*_data_file_cache_key("gal.tsv"))


@lru_cache(maxsize=4)
def _load_kk_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[tuple[str, str, str, str], ...]:
    """Load and cache kk.tsv while still picking up file edits."""
    if mtime_ns is None:
        return ()

    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for parts in reader:
            if _is_header_row(parts, ("word", "ipa", "russian_def", "georgian_def")):
                continue
            if len(parts) < 4:
                continue
            mingrelian = parts[0].strip()
            ipa = parts[1].strip()
            russian = parts[2].strip()
            georgian = parts[3].strip()
            if mingrelian and russian and georgian:
                rows.append((mingrelian, ipa, russian, georgian))
    return tuple(rows)


def _load_kk_rows() -> tuple[tuple[str, str, str, str], ...]:
    """Load Mingrelian ↔ Russian/Georgian dictionary rows."""
    return _load_kk_rows_cached(*_data_file_cache_key("kk.tsv"))


@lru_cache(maxsize=4)
def _load_context_source_entries_cached(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[str, ...]:
    """Load and cache unstructured fallback context blocks."""
    if mtime_ns is None:
        return ()

    context_source_text = Path(file_path).read_text(encoding="utf-8")
    entries = re.split(r"\n\s*\n", context_source_text.strip())
    return tuple(entry.strip() for entry in entries if entry.strip())


def _load_context_source_entries() -> tuple[str, ...]:
    """Load fallback context blocks from the source-agnostic reference corpus."""
    return _load_context_source_entries_cached(*_data_file_cache_key("context_source.txt"))


@lru_cache(maxsize=4)
def _load_grammar_cached(path: str, mtime_ns: Optional[int]) -> str:
    """Load and cache grammar text, invalidating automatically when it changes."""
    if mtime_ns is None:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _load_grammar(path: Optional[str] = None) -> str:
    """Load the Mingrelian grammar file."""
    if path is None:
        path, mtime_ns = _data_file_cache_key("harris.txt")
    else:
        try:
            mtime_ns = Path(path).stat().st_mtime_ns
        except FileNotFoundError:
            return ""

    try:
        return _load_grammar_cached(path, mtime_ns)
    except FileNotFoundError:
        return ""


def _load_compact_grammar(path: Optional[str] = None) -> str:
    """Load the compact translator-oriented grammar reference."""
    if path is None:
        path, mtime_ns = _data_file_cache_key("harris_compact.txt")
    else:
        try:
            mtime_ns = Path(path).stat().st_mtime_ns
        except FileNotFoundError:
            return ""

    try:
        return _load_grammar_cached(path, mtime_ns)
    except FileNotFoundError:
        return ""
