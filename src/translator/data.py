#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.language_packs import get_language_pack
from src.translator.text_utils import _normalize_lookup_value

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_ID = "mingrelian"
PACK_ID_ALIASES = {
    "bats": "tsova_tush",
    "batsbi": "tsova_tush",
    "tsova-tush": "tsova_tush",
    "tsova tush": "tsova_tush",
    "swan": "svan",
}
LOW_RESOURCE_HEADER_ALIASES = {
    "mingrelian": ("mingrelian",),
    "tsova_tush": ("tsova_tush", "tsova-tush", "tsova tush", "bats", "batsbi"),
    "svan": ("svan", "swan"),
}


def _normalize_pack_id(pack_id: str = DEFAULT_PACK_ID) -> str:
    normalized = (pack_id or DEFAULT_PACK_ID).strip().casefold()
    normalized = PACK_ID_ALIASES.get(normalized, normalized)
    return normalized.replace("-", "_").replace(" ", "_")


def _normalize_header_cell(text: str) -> str:
    return _normalize_lookup_value(text).lstrip("\ufeff").replace("-", "_").replace(" ", "_")


def _low_resource_header_aliases(pack_id: str) -> tuple[str, ...]:
    normalized_pack_id = _normalize_pack_id(pack_id)
    aliases = LOW_RESOURCE_HEADER_ALIASES.get(normalized_pack_id, (normalized_pack_id,))
    return tuple(_normalize_header_cell(alias) for alias in aliases)


def _is_low_resource_header_cell(text: str, pack_id: str) -> bool:
    return _normalize_header_cell(text) in _low_resource_header_aliases(pack_id)


def _candidate_data_dirs(pack_id: str = DEFAULT_PACK_ID) -> list[Path]:
    normalized_pack_id = _normalize_pack_id(pack_id)
    configured_data_dir = os.getenv("ARGO_DATA_DIR")
    candidate_dirs = []
    if configured_data_dir:
        configured_path = Path(configured_data_dir).expanduser()
        candidate_dirs.append(configured_path / normalized_pack_id)
        candidate_dirs.append(configured_path)

    candidate_dirs.append(REPO_ROOT / "private_data" / normalized_pack_id)

    if normalized_pack_id == DEFAULT_PACK_ID:
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

    return candidate_dirs


def _master_lexicon_enabled() -> bool:
    """Allow master lexicon ablations without editing the core pipeline."""
    value = os.getenv("ARGO_ENABLE_MASTER_LEXICON", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _get_data_path(filename: str, pack_id: str = DEFAULT_PACK_ID) -> str:
    """Get the path to a data file, checking multiple possible locations."""
    for data_dir in _candidate_data_dirs(pack_id):
        candidate = data_dir / filename
        if candidate.exists():
            return str(candidate)

    normalized_pack_id = _normalize_pack_id(pack_id)
    if normalized_pack_id == DEFAULT_PACK_ID:
        return str(REPO_ROOT / "private_data" / filename)
    configured_data_dir = os.getenv("ARGO_DATA_DIR")
    if configured_data_dir:
        return str(Path(configured_data_dir).expanduser() / normalized_pack_id / filename)
    return str(REPO_ROOT / "private_data" / normalized_pack_id / filename)


def _data_file_cache_key(filename: str, pack_id: str = DEFAULT_PACK_ID) -> tuple[str, Optional[int]]:
    """Build a cache key that invalidates when a data file changes on disk."""
    if _normalize_pack_id(pack_id) == DEFAULT_PACK_ID:
        file_path = _get_data_path(filename)
    else:
        file_path = _get_data_path(filename, pack_id=pack_id)
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


def _is_low_resource_to_english_header(parts: list[str], pack_id: str) -> bool:
    if len(parts) < 2:
        return False
    return _is_low_resource_header_cell(parts[0], pack_id) and _normalize_header_cell(parts[1]) == "english"


def _is_russian_to_low_resource_header(parts: list[str], pack_id: str) -> bool:
    if len(parts) < 2:
        return False
    return _normalize_header_cell(parts[0]) == "russian" and _is_low_resource_header_cell(parts[1], pack_id)


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


def _load_master_lexicon_rows(pack_id: str = DEFAULT_PACK_ID) -> tuple[tuple[str, str, str], ...]:
    """Load the master lexicon used for exact low-resource ↔ English matches."""
    if not _master_lexicon_enabled():
        return ()
    return _load_master_lexicon_rows_cached(*_data_file_cache_key("master-lexicon-mkhedruli.csv", pack_id))


@lru_cache(maxsize=4)
def _load_sentence_pairs_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
    pack_id: str = DEFAULT_PACK_ID,
) -> tuple[tuple[str, str], ...]:
    """Load and cache sentence_pairs.tsv while still picking up file edits."""
    if mtime_ns is None:
        return ()

    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if _is_low_resource_to_english_header(parts, pack_id):
                continue
            if len(parts) < 2:
                continue
            mingrelian = parts[0].strip()
            english = parts[1].strip()
            if mingrelian and english:
                rows.append((mingrelian, english))
    return tuple(rows)


def _load_sentence_pairs_rows(pack_id: str = DEFAULT_PACK_ID) -> tuple[tuple[str, str], ...]:
    """Load sentence pairs for extractive low-resource ↔ English lookups."""
    return _load_sentence_pairs_rows_cached(*_data_file_cache_key("sentence_pairs.tsv", pack_id), pack_id)


@lru_cache(maxsize=4)
def _load_gal_rows_cached(
    file_path: str,
    mtime_ns: Optional[int],
    pack_id: str = DEFAULT_PACK_ID,
) -> tuple[tuple[str, str], ...]:
    """Load and cache gal.tsv while still picking up file edits."""
    if mtime_ns is None:
        return ()

    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for parts in reader:
            if _is_russian_to_low_resource_header(parts, pack_id):
                continue
            if len(parts) < 2:
                continue
            russian = parts[0].strip()
            mingrelian = parts[1].strip()
            if russian and mingrelian:
                rows.append((russian, mingrelian))
    return tuple(rows)


def _load_gal_rows(pack_id: str = DEFAULT_PACK_ID) -> tuple[tuple[str, str], ...]:
    """Load Russian ↔ low-resource dictionary rows."""
    return _load_gal_rows_cached(*_data_file_cache_key("gal.tsv", pack_id), pack_id)


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
            if mingrelian and (russian or georgian):
                rows.append((mingrelian, ipa, russian, georgian))
    return tuple(rows)


def _load_kk_rows(pack_id: str = DEFAULT_PACK_ID) -> tuple[tuple[str, str, str, str], ...]:
    """Load low-resource ↔ Russian/Georgian dictionary rows."""
    return _load_kk_rows_cached(*_data_file_cache_key("kk.tsv", pack_id))


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


def _load_context_source_entries(pack_id: str = DEFAULT_PACK_ID) -> tuple[str, ...]:
    """Load fallback context blocks from the source-agnostic reference corpus."""
    path, mtime_ns = _data_file_cache_key("context_source.txt", pack_id)
    if mtime_ns is None:
        path, mtime_ns = _data_file_cache_key("kajaia_cleaned.txt", pack_id)
    return _load_context_source_entries_cached(path, mtime_ns)


@lru_cache(maxsize=4)
def _load_grammar_cached(path: str, mtime_ns: Optional[int]) -> str:
    """Load and cache grammar text, invalidating automatically when it changes."""
    if mtime_ns is None:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _grammar_filename(pack_id: str, *, compact: bool = False) -> str:
    """Return the authored grammar asset name for a language pack."""
    pack = get_language_pack(_normalize_pack_id(pack_id))
    return pack.compact_grammar_filename if compact else pack.grammar_filename


def _load_grammar(path: Optional[str] = None, pack_id: str = DEFAULT_PACK_ID) -> str:
    """Load the low-resource grammar file."""
    if path is None:
        path, mtime_ns = _data_file_cache_key(_grammar_filename(pack_id), pack_id)
    else:
        try:
            mtime_ns = Path(path).stat().st_mtime_ns
        except FileNotFoundError:
            return ""

    try:
        return _load_grammar_cached(path, mtime_ns)
    except FileNotFoundError:
        return ""


def _load_compact_grammar(path: Optional[str] = None, pack_id: str = DEFAULT_PACK_ID) -> str:
    """Load the compact translator-oriented grammar reference."""
    if path is None:
        path, mtime_ns = _data_file_cache_key(_grammar_filename(pack_id, compact=True), pack_id)
    else:
        try:
            mtime_ns = Path(path).stat().st_mtime_ns
        except FileNotFoundError:
            return ""

    try:
        return _load_grammar_cached(path, mtime_ns)
    except FileNotFoundError:
        return ""
