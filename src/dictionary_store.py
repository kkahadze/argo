#!/usr/bin/env python3
"""
Indexed dictionary access for the translation pipeline.

This module keeps the old grep-style behavior available, but moves file IO and
row scanning behind a cacheable store. Common standalone/exact lookups use
indexes; broad substring lookups remain as a compatibility fallback.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Callable, Optional


LOOKUP_SEPARATOR = "========\n"
MAX_LOOKUP_OUTPUT_CHARS = 10000

FIGURATIVE_MARKERS = {
    "ru": ("переносное значение", "перен."),
    "ka": ("გადატანილი მნიშვნელობით", "გადატ."),
}


@dataclass(frozen=True)
class SentencePair:
    mingrelian: str
    english: str


@dataclass(frozen=True)
class GalEntry:
    russian: str
    mingrelian: str


@dataclass(frozen=True)
class KkEntry:
    mingrelian: str
    ipa: str
    russian: str
    georgian: str


@dataclass(frozen=True)
class SearchResult:
    output: str
    has_standalone_matches: bool


def get_data_path(filename: str) -> str:
    """Get the path to a data file, checking multiple possible locations."""
    project_root = Path(__file__).parent.parent
    configured_data_dir = os.getenv("ARGO_DATA_DIR")

    candidate_dirs = []
    if configured_data_dir:
        candidate_dirs.append(Path(configured_data_dir).expanduser())

    candidate_dirs.extend(
        [
            project_root / "private_data",
            project_root / "fastapi_app" / "data",
            project_root / "data",
            project_root / "notebooks",
            project_root / "notebooks" / "dicts",
        ]
    )

    for data_dir in candidate_dirs:
        candidate = data_dir / filename
        if candidate.exists():
            return str(candidate)

    return str(project_root / "private_data" / filename)


def normalize_lookup_value(text: str) -> str:
    """Normalize user input and dictionary text for exact-match comparisons."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def truncate_lookup_output(output: str) -> str:
    """Keep lookup payloads bounded before they enter the LLM prompt."""
    if len(output) > MAX_LOOKUP_OUTPUT_CHARS:
        return output[:MAX_LOOKUP_OUTPUT_CHARS]
    return output


def normalize_gloss_segment(text: str) -> str:
    """Strip numbering and separators that are not meaningful translation content."""
    normalized = re.sub(r"\s+", " ", text).strip(" ;,\n\t")
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
    normalized = re.sub(r"\s+\d+\.$", "", normalized)
    return normalized.strip(" ;,\n\t")


def split_figurative_gloss(gloss: str, lang_code: str) -> tuple[str, Optional[str]]:
    """Split a dictionary gloss into primary and figurative senses."""
    markers = FIGURATIVE_MARKERS.get(lang_code, ())
    if not gloss or not markers:
        normalized = normalize_gloss_segment(gloss)
        return normalized, None

    lowered = gloss.lower()
    marker_position = min(
        (idx for marker in markers if (idx := lowered.find(marker)) != -1),
        default=-1,
    )
    if marker_position == -1:
        normalized = normalize_gloss_segment(gloss)
        return normalized, None

    matched_marker = next(marker for marker in markers if lowered.find(marker) == marker_position)
    primary = normalize_gloss_segment(gloss[:marker_position])
    secondary = normalize_gloss_segment(gloss[marker_position + len(matched_marker):])
    return primary, secondary or None


def format_kk_entry(mingrelian: str, russian: str, georgian: str) -> str:
    """Format a kk.tsv entry with figurative senses clearly marked as secondary."""
    ru_primary, ru_figurative = split_figurative_gloss(russian, "ru")
    ka_primary, ka_figurative = split_figurative_gloss(georgian, "ka")

    lines = [f"Mingrelian: {mingrelian}"]
    if ru_primary:
        lines.append(f"Russian primary meaning: {ru_primary}")
    if ru_figurative:
        lines.append(f"Russian secondary figurative meaning: {ru_figurative}")
    if ka_primary:
        lines.append(f"Georgian primary meaning: {ka_primary}")
    if ka_figurative:
        lines.append(f"Georgian secondary figurative meaning: {ka_figurative}")
    return "\n".join(lines)


def choose_kk_bridge_gloss(
    russian: str,
    georgian: str,
    target_lang: Optional[str],
) -> tuple[str, str]:
    """
    Pick the safest gloss for direct bridging from kk.tsv.

    Russian glosses bridge to English more reliably than Georgian glosses in the
    current pipeline, so prefer Russian when producing English.
    """
    russian_primary, _ = split_figurative_gloss(russian, "ru")
    georgian_primary, _ = split_figurative_gloss(georgian, "ka")

    if target_lang == "english":
        if russian_primary:
            return russian_primary, "ru"
        if georgian_primary:
            return georgian_primary, "ka"
    elif target_lang == "georgian":
        if georgian_primary:
            return georgian_primary, "ka"
        if russian_primary:
            return russian_primary, "ru"

    if georgian_primary:
        return georgian_primary, "ka"
    if russian_primary:
        return russian_primary, "ru"
    return "", ""


def is_standalone_match(text: str, word: str) -> bool:
    """
    Check if word appears as a standalone word in text.

    Supports Mingrelian headword notations like `ნუმ(უ)` by allowing
    punctuation between letters while still enforcing standalone boundaries.
    """
    word = (word or "").strip()
    if not word:
        return False
    return _compiled_word_pattern(word, standalone=True).search(text) is not None


def is_substring_match(text: str, word: str) -> bool:
    """Like `word in text`, but also handles punctuation inside headword forms."""
    word = (word or "").strip()
    if not word:
        return False
    if word in text:
        return True
    return _compiled_word_pattern(word, standalone=False).search(text) is not None


@lru_cache(maxsize=4096)
def _compiled_word_pattern(word: str, standalone: bool) -> re.Pattern:
    escaped = re.escape(word)
    if len(word) <= 2:
        core = escaped
    else:
        sep = r"[^\w\s]*"
        core = sep.join(re.escape(ch) for ch in word)

    if standalone:
        pattern = r"(?<!\w)" + core + r"(?!\w)"
    else:
        pattern = core

    return re.compile(pattern, re.IGNORECASE)


def _data_file_cache_key(filename: str) -> tuple[str, Optional[int]]:
    """Build a cache key that invalidates when a data file changes on disk."""
    file_path = get_data_path(filename)
    try:
        mtime_ns = Path(file_path).stat().st_mtime_ns
    except FileNotFoundError:
        return file_path, None
    return file_path, mtime_ns


def _read_tsv_rows(file_path: str, expected_columns: int) -> list[list[str]]:
    try:
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    rows: list[list[str]] = []
    for line in lines:
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) >= expected_columns:
            rows.append(parts)
    return rows


def _load_sentence_pairs(file_path: str, mtime_ns: Optional[int]) -> tuple[SentencePair, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for parts in _read_tsv_rows(file_path, 2):
        mingrelian, english = parts[0], parts[1]
        if mingrelian and english:
            rows.append(SentencePair(mingrelian=mingrelian, english=english))
    return tuple(rows)


def _load_gal_entries(file_path: str, mtime_ns: Optional[int]) -> tuple[GalEntry, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for index, parts in enumerate(_read_tsv_rows(file_path, 2)):
        russian, mingrelian = parts[0], parts[1]
        if index == 0 and russian.casefold() == "russian" and mingrelian.casefold() == "mingrelian":
            continue
        if russian and mingrelian:
            rows.append(GalEntry(russian=russian, mingrelian=mingrelian))
    return tuple(rows)


def _load_kk_entries(file_path: str, mtime_ns: Optional[int]) -> tuple[KkEntry, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for index, parts in enumerate(_read_tsv_rows(file_path, 4)):
        mingrelian, ipa, russian, georgian = parts[0], parts[1], parts[2], parts[3]
        if (
            index == 0
            and mingrelian.casefold() == "word"
            and ipa.casefold() == "ipa"
            and russian.casefold() == "russian_def"
            and georgian.casefold() == "georgian_def"
        ):
            continue
        if mingrelian and russian and georgian:
            rows.append(
                KkEntry(
                    mingrelian=mingrelian,
                    ipa=ipa,
                    russian=russian,
                    georgian=georgian,
                )
            )
    return tuple(rows)


def _load_context_entries(file_path: str, mtime_ns: Optional[int]) -> tuple[str, ...]:
    if mtime_ns is None:
        return ()

    try:
        context_text = Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()

    entries = re.split(r"\n\s*\n", context_text.strip())
    return tuple(entry.strip() for entry in entries if entry.strip())


def _lookup_terms(*texts: str) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        normalized = normalize_lookup_value(text)
        if normalized:
            terms.add(normalized)
        for token in re.findall(r"\w+", text or "", flags=re.UNICODE):
            token_normalized = normalize_lookup_value(token)
            if token_normalized:
                terms.add(token_normalized)
    return terms


def _add_index_value(index: dict[str, list[int]], key: str, row_index: int) -> None:
    if not key:
        return
    bucket = index[key]
    if not bucket or bucket[-1] != row_index:
        bucket.append(row_index)


def _build_index(rows: tuple, fields: Callable[[object], tuple[str, ...]]) -> dict[str, tuple[int, ...]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for term in _lookup_terms(*fields(row)):
            _add_index_value(index, term, row_index)
    return {key: tuple(value) for key, value in index.items()}


def _build_exact_index(rows: tuple, fields: Callable[[object], tuple[str, ...]]) -> dict[str, tuple[int, ...]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for field in fields(row):
            _add_index_value(index, normalize_lookup_value(field), row_index)
    return {key: tuple(value) for key, value in index.items()}


class DictionaryStore:
    """Cached dictionary rows plus indexes for common exact/standalone lookups."""

    def __init__(
        self,
        *,
        sentence_pairs: tuple[SentencePair, ...],
        gal_entries: tuple[GalEntry, ...],
        kk_entries: tuple[KkEntry, ...],
        context_key: tuple[str, Optional[int]],
    ) -> None:
        self.sentence_pairs = sentence_pairs
        self.gal_entries = gal_entries
        self.kk_entries = kk_entries
        self._context_key = context_key
        self._context_entries: Optional[tuple[str, ...]] = None
        self._context_index: Optional[dict[str, tuple[int, ...]]] = None

        self._sentence_index = _build_index(
            sentence_pairs,
            lambda row: (row.mingrelian, row.english),
        )
        self._gal_index = _build_index(
            gal_entries,
            lambda row: (row.russian, row.mingrelian),
        )
        self._kk_index = _build_index(
            kk_entries,
            lambda row: (row.mingrelian, row.ipa, row.russian, row.georgian),
        )

        self._sentence_mingrelian_exact = _build_exact_index(sentence_pairs, lambda row: (row.mingrelian,))
        self._sentence_english_exact = _build_exact_index(sentence_pairs, lambda row: (row.english,))
        self._gal_russian_exact = _build_exact_index(gal_entries, lambda row: (row.russian,))
        self._gal_mingrelian_exact = _build_exact_index(gal_entries, lambda row: (row.mingrelian,))
        self._kk_mingrelian_exact = _build_exact_index(kk_entries, lambda row: (row.mingrelian,))
        self._kk_russian_exact = _build_exact_index(kk_entries, lambda row: (row.russian,))
        self._kk_georgian_exact = _build_exact_index(kk_entries, lambda row: (row.georgian,))

    @property
    def context_entries(self) -> tuple[str, ...]:
        self._ensure_context_loaded()
        return self._context_entries or ()

    def exact_sentence_mingrelian(self, text: str) -> tuple[SentencePair, ...]:
        return self._rows_for_exact(self.sentence_pairs, self._sentence_mingrelian_exact, text)

    def exact_sentence_english(self, text: str) -> tuple[SentencePair, ...]:
        return self._rows_for_exact(self.sentence_pairs, self._sentence_english_exact, text)

    def exact_gal_russian(self, text: str) -> tuple[GalEntry, ...]:
        return self._rows_for_exact(self.gal_entries, self._gal_russian_exact, text)

    def exact_gal_mingrelian(self, text: str) -> tuple[GalEntry, ...]:
        return self._rows_for_exact(self.gal_entries, self._gal_mingrelian_exact, text)

    def exact_kk_mingrelian(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_mingrelian_exact, text)

    def exact_kk_russian(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_russian_exact, text)

    def exact_kk_georgian(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_georgian_exact, text)

    @lru_cache(maxsize=4096)
    def search_sentence_pairs(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.sentence_pairs,
            index=self._sentence_index,
            standalone_text=lambda row: (row.mingrelian, row.english),
            substring_text=lambda row: f"{row.mingrelian}\t{row.english}",
            format_entry=lambda row: f"Mingrelian: {row.mingrelian}\nEnglish: {row.english}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_gal(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.gal_entries,
            index=self._gal_index,
            standalone_text=lambda row: (row.mingrelian, row.russian),
            substring_text=lambda row: f"{row.russian}\t{row.mingrelian}",
            format_entry=lambda row: f"Mingrelian: {row.mingrelian}\nRussian: {row.russian}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_kk(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.kk_entries,
            index=self._kk_index,
            standalone_text=lambda row: (row.mingrelian, row.russian, row.georgian),
            substring_text=lambda row: "\t".join((row.mingrelian, row.ipa, row.russian, row.georgian)),
            format_entry=lambda row: format_kk_entry(row.mingrelian, row.russian, row.georgian) + "\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_context(self, word: str, *, standalone_only: bool = False) -> str:
        self._ensure_context_loaded()
        result = self._search_rows(
            word=word,
            rows=self._context_entries or (),
            index=self._context_index or {},
            standalone_text=lambda entry: (entry,),
            substring_text=lambda entry: entry,
            format_entry=lambda entry: entry.strip() + "\n",
            standalone_only=standalone_only,
        )
        return result.output

    def _ensure_context_loaded(self) -> None:
        if self._context_entries is not None and self._context_index is not None:
            return

        context_entries = _load_context_entries(*self._context_key)
        self._context_entries = context_entries
        self._context_index = _build_index(
            context_entries,
            lambda entry: (entry,),
        )

    def _rows_for_exact(self, rows: tuple, index: dict[str, tuple[int, ...]], text: str) -> tuple:
        row_indexes = index.get(normalize_lookup_value(text), ())
        return tuple(rows[row_index] for row_index in row_indexes)

    def _candidate_indexes(self, index: dict[str, tuple[int, ...]], word: str) -> tuple[int, ...]:
        return index.get(normalize_lookup_value(word), ())

    def _search_rows(
        self,
        *,
        word: str,
        rows: tuple,
        index: dict[str, tuple[int, ...]],
        standalone_text: Callable[[object], tuple[str, ...]],
        substring_text: Callable[[object], str],
        format_entry: Callable[[object], str],
        standalone_only: bool,
    ) -> SearchResult:
        if not rows:
            return SearchResult("", False)

        candidate_indexes = self._candidate_indexes(index, word)
        standalone_indexes = candidate_indexes or range(len(rows))

        standalone_parts = [LOOKUP_SEPARATOR]
        for row_index in standalone_indexes:
            row = rows[row_index]
            if any(is_standalone_match(text, word) for text in standalone_text(row)):
                standalone_parts.append(format_entry(row))
                standalone_parts.append(LOOKUP_SEPARATOR)

        if len(standalone_parts) > 1:
            return SearchResult("".join(standalone_parts), True)

        if standalone_only:
            return SearchResult("", False)

        substring_parts = [LOOKUP_SEPARATOR]
        for row in rows:
            if is_substring_match(substring_text(row), word):
                substring_parts.append(format_entry(row))
                substring_parts.append(LOOKUP_SEPARATOR)

        if len(substring_parts) > 1:
            return SearchResult("".join(substring_parts), False)
        return SearchResult("", False)


def get_dictionary_store() -> DictionaryStore:
    """Return a cached store, invalidated by source file modification times."""
    return _get_dictionary_store_cached(
        _data_file_cache_key("sentence_pairs.tsv"),
        _data_file_cache_key("gal.tsv"),
        _data_file_cache_key("kk.tsv"),
        _data_file_cache_key("kajaia_cleaned.txt"),
    )


@lru_cache(maxsize=4)
def _get_dictionary_store_cached(
    sentence_pairs_key: tuple[str, Optional[int]],
    gal_key: tuple[str, Optional[int]],
    kk_key: tuple[str, Optional[int]],
    context_key: tuple[str, Optional[int]],
) -> DictionaryStore:
    return DictionaryStore(
        sentence_pairs=_load_sentence_pairs(*sentence_pairs_key),
        gal_entries=_load_gal_entries(*gal_key),
        kk_entries=_load_kk_entries(*kk_key),
        context_key=context_key,
    )
