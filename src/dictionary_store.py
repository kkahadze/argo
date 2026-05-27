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
from pathlib import Path
from typing import Callable, Optional


LOOKUP_SEPARATOR = "========\n"
MAX_LOOKUP_OUTPUT_CHARS = 10000

FIGURATIVE_MARKERS = {
    "ru": ("переносное значение", "перен."),
    "ka": ("გადატანილი მნიშვნელობით", "გადატ."),
}

DEFAULT_PACK_ID = "mingrelian"
PACK_ID_ALIASES = {
    "bats": "tsova_tush",
    "batsbi": "tsova_tush",
    "tsova-tush": "tsova_tush",
    "tsova tush": "tsova_tush",
    "swan": "svan",
}
LOW_RESOURCE_LABELS = {
    "mingrelian": "Mingrelian",
    "tsova_tush": "Bats",
    "svan": "Svan",
}
LOW_RESOURCE_HEADER_ALIASES = {
    "mingrelian": ("mingrelian",),
    "tsova_tush": ("tsova_tush", "tsova-tush", "tsova tush", "bats", "batsbi"),
    "svan": ("svan", "swan"),
}


def normalize_pack_id(pack_id: str = DEFAULT_PACK_ID) -> str:
    normalized = (pack_id or DEFAULT_PACK_ID).strip().casefold()
    normalized = PACK_ID_ALIASES.get(normalized, normalized)
    return normalized.replace("-", "_").replace(" ", "_")


def low_resource_label(pack_id: str) -> str:
    normalized_pack_id = normalize_pack_id(pack_id)
    return LOW_RESOURCE_LABELS.get(normalized_pack_id, normalized_pack_id.replace("_", " ").title())


@dataclass(frozen=True)
class SentencePair:
    low_resource: str
    english: str

    @property
    def mingrelian(self) -> str:
        return self.low_resource

    @property
    def tsova_tush(self) -> str:
        return self.low_resource

    @property
    def svan(self) -> str:
        return self.low_resource


@dataclass(frozen=True)
class GalEntry:
    russian: str
    low_resource: str

    @property
    def mingrelian(self) -> str:
        return self.low_resource

    @property
    def tsova_tush(self) -> str:
        return self.low_resource

    @property
    def svan(self) -> str:
        return self.low_resource


@dataclass(frozen=True)
class KkEntry:
    low_resource: str
    ipa: str
    russian: str
    georgian: str

    @property
    def mingrelian(self) -> str:
        return self.low_resource

    @property
    def tsova_tush(self) -> str:
        return self.low_resource

    @property
    def svan(self) -> str:
        return self.low_resource


@dataclass(frozen=True)
class TranslationOverride:
    source_language: str
    target_language: str
    source_text: str
    target_text: str


@dataclass(frozen=True)
class SearchResult:
    output: str
    has_standalone_matches: bool


def get_data_path(filename: str, pack_id: str = DEFAULT_PACK_ID) -> str:
    """Get a data path through the canonical translator data resolver."""
    from src.translator import data as translator_data

    if normalize_pack_id(pack_id) == DEFAULT_PACK_ID:
        return translator_data._get_data_path(filename)
    return translator_data._get_data_path(filename, pack_id=pack_id)


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


def format_kk_entry(
    low_resource: str,
    russian: str,
    georgian: str,
    *,
    label: str = "Mingrelian",
) -> str:
    """Format a kk.tsv entry with figurative senses clearly marked as secondary."""
    ru_primary, ru_figurative = split_figurative_gloss(russian, "ru")
    ka_primary, ka_figurative = split_figurative_gloss(georgian, "ka")

    lines = [f"{label}: {low_resource}"]
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


def _data_file_cache_key(filename: str, pack_id: str = DEFAULT_PACK_ID) -> tuple[str, Optional[int]]:
    """Build a cache key that invalidates when a data file changes on disk."""
    if normalize_pack_id(pack_id) == DEFAULT_PACK_ID:
        file_path = get_data_path(filename)
    else:
        file_path = get_data_path(filename, pack_id=pack_id)
    try:
        mtime_ns = Path(file_path).stat().st_mtime_ns
    except FileNotFoundError:
        return file_path, None
    return file_path, mtime_ns


def _normalize_header_cell(text: str) -> str:
    return normalize_lookup_value(text).lstrip("\ufeff").replace("-", "_").replace(" ", "_")


def _low_resource_header_aliases(pack_id: str) -> tuple[str, ...]:
    normalized_pack_id = normalize_pack_id(pack_id)
    aliases = LOW_RESOURCE_HEADER_ALIASES.get(normalized_pack_id, (normalized_pack_id,))
    return tuple(_normalize_header_cell(alias) for alias in aliases)


def _is_low_resource_header_cell(text: str, pack_id: str) -> bool:
    return _normalize_header_cell(text) in _low_resource_header_aliases(pack_id)


def _is_sentence_pairs_header(parts: list[str], pack_id: str) -> bool:
    if len(parts) < 2:
        return False
    return _is_low_resource_header_cell(parts[0], pack_id) and _normalize_header_cell(parts[1]) == "english"


def _is_gal_header(parts: list[str], pack_id: str) -> bool:
    if len(parts) < 2:
        return False
    return _normalize_header_cell(parts[0]) == "russian" and _is_low_resource_header_cell(parts[1], pack_id)


def _context_file_cache_key(pack_id: str) -> tuple[str, Optional[int]]:
    context_key = _data_file_cache_key("context_source.txt", pack_id)
    if context_key[1] is not None:
        return context_key
    return _data_file_cache_key("kajaia_cleaned.txt", pack_id)


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


def _load_sentence_pairs(
    file_path: str,
    mtime_ns: Optional[int],
    pack_id: str = DEFAULT_PACK_ID,
) -> tuple[SentencePair, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for parts in _read_tsv_rows(file_path, 2):
        low_resource, english = parts[0], parts[1]
        if _is_sentence_pairs_header(parts, pack_id):
            continue
        if low_resource and english:
            rows.append(SentencePair(low_resource=low_resource, english=english))
    return tuple(rows)


def _load_gal_entries(
    file_path: str,
    mtime_ns: Optional[int],
    pack_id: str = DEFAULT_PACK_ID,
) -> tuple[GalEntry, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for parts in _read_tsv_rows(file_path, 2):
        russian, low_resource = parts[0], parts[1]
        if _is_gal_header(parts, pack_id):
            continue
        if russian and low_resource:
            rows.append(GalEntry(russian=russian, low_resource=low_resource))
    return tuple(rows)


def _load_kk_entries(file_path: str, mtime_ns: Optional[int]) -> tuple[KkEntry, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for index, parts in enumerate(_read_tsv_rows(file_path, 4)):
        low_resource, ipa, russian, georgian = parts[0], parts[1], parts[2], parts[3]
        if (
            index == 0
            and low_resource.casefold() == "word"
            and ipa.casefold() == "ipa"
            and russian.casefold() == "russian_def"
            and georgian.casefold() == "georgian_def"
        ):
            continue
        if low_resource and (russian or georgian):
            rows.append(
                KkEntry(
                    low_resource=low_resource,
                    ipa=ipa,
                    russian=russian,
                    georgian=georgian,
                )
            )
    return tuple(rows)


def _load_translation_overrides(
    file_path: str,
    mtime_ns: Optional[int],
) -> tuple[TranslationOverride, ...]:
    if mtime_ns is None:
        return ()

    rows = []
    for parts in _read_tsv_rows(file_path, 4):
        source_language, target_language, source_text, target_text = parts[:4]
        if (
            normalize_lookup_value(source_language).lstrip("\ufeff") == "source_language"
            and normalize_lookup_value(target_language) == "target_language"
        ):
            continue
        if source_language and target_language and source_text and target_text:
            rows.append(
                TranslationOverride(
                    source_language=source_language,
                    target_language=target_language,
                    source_text=source_text,
                    target_text=target_text,
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


def _translation_override_key(source_language: str, target_language: str, source_text: str) -> str:
    return "\t".join(
        (
            normalize_lookup_value(source_language),
            normalize_lookup_value(target_language),
            normalize_lookup_value(source_text),
        )
    )


def _build_translation_override_index(rows: tuple[TranslationOverride, ...]) -> dict[str, tuple[int, ...]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        _add_index_value(
            index,
            _translation_override_key(row.source_language, row.target_language, row.source_text),
            row_index,
        )
    return {key: tuple(value) for key, value in index.items()}


class DictionaryStore:
    """Cached dictionary rows plus indexes for common exact/standalone lookups."""

    def __init__(
        self,
        *,
        pack_id: str,
        translation_overrides: tuple[TranslationOverride, ...],
        sentence_pairs: tuple[SentencePair, ...],
        gal_entries: tuple[GalEntry, ...],
        kk_entries: tuple[KkEntry, ...],
        context_key: tuple[str, Optional[int]],
    ) -> None:
        self.pack_id = normalize_pack_id(pack_id)
        self.low_resource_label = low_resource_label(self.pack_id)
        self.translation_overrides = translation_overrides
        self.sentence_pairs = sentence_pairs
        self.gal_entries = gal_entries
        self.kk_entries = kk_entries
        self._context_key = context_key
        self._context_entries: Optional[tuple[str, ...]] = None
        self._context_index: Optional[dict[str, tuple[int, ...]]] = None

        self._sentence_index = _build_index(
            sentence_pairs,
            lambda row: (row.low_resource, row.english),
        )
        self._sentence_low_resource_index = _build_index(
            sentence_pairs,
            lambda row: (row.low_resource,),
        )
        self._gal_index = _build_index(
            gal_entries,
            lambda row: (row.russian, row.low_resource),
        )
        self._gal_low_resource_index = _build_index(
            gal_entries,
            lambda row: (row.low_resource,),
        )
        self._kk_index = _build_index(
            kk_entries,
            lambda row: (row.low_resource, row.ipa, row.russian, row.georgian),
        )
        self._kk_low_resource_index = _build_index(
            kk_entries,
            lambda row: (row.low_resource,),
        )

        self._sentence_low_resource_exact = _build_exact_index(sentence_pairs, lambda row: (row.low_resource,))
        self._sentence_english_exact = _build_exact_index(sentence_pairs, lambda row: (row.english,))
        self._gal_russian_exact = _build_exact_index(gal_entries, lambda row: (row.russian,))
        self._gal_low_resource_exact = _build_exact_index(gal_entries, lambda row: (row.low_resource,))
        self._kk_low_resource_exact = _build_exact_index(kk_entries, lambda row: (row.low_resource,))
        self._kk_russian_exact = _build_exact_index(kk_entries, lambda row: (row.russian,))
        self._kk_georgian_exact = _build_exact_index(kk_entries, lambda row: (row.georgian,))
        self._translation_override_exact = _build_translation_override_index(translation_overrides)

    @property
    def context_entries(self) -> tuple[str, ...]:
        self._ensure_context_loaded()
        return self._context_entries or ()

    def exact_sentence_low_resource(self, text: str) -> tuple[SentencePair, ...]:
        return self._rows_for_exact(self.sentence_pairs, self._sentence_low_resource_exact, text)

    def exact_sentence_mingrelian(self, text: str) -> tuple[SentencePair, ...]:
        return self.exact_sentence_low_resource(text)

    def exact_sentence_tsova_tush(self, text: str) -> tuple[SentencePair, ...]:
        return self.exact_sentence_low_resource(text)

    def exact_sentence_svan(self, text: str) -> tuple[SentencePair, ...]:
        return self.exact_sentence_low_resource(text)

    def exact_sentence_english(self, text: str) -> tuple[SentencePair, ...]:
        return self._rows_for_exact(self.sentence_pairs, self._sentence_english_exact, text)

    def exact_gal_russian(self, text: str) -> tuple[GalEntry, ...]:
        return self._rows_for_exact(self.gal_entries, self._gal_russian_exact, text)

    def exact_gal_low_resource(self, text: str) -> tuple[GalEntry, ...]:
        return self._rows_for_exact(self.gal_entries, self._gal_low_resource_exact, text)

    def exact_gal_mingrelian(self, text: str) -> tuple[GalEntry, ...]:
        return self.exact_gal_low_resource(text)

    def exact_gal_tsova_tush(self, text: str) -> tuple[GalEntry, ...]:
        return self.exact_gal_low_resource(text)

    def exact_gal_svan(self, text: str) -> tuple[GalEntry, ...]:
        return self.exact_gal_low_resource(text)

    def exact_kk_low_resource(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_low_resource_exact, text)

    def exact_kk_mingrelian(self, text: str) -> tuple[KkEntry, ...]:
        return self.exact_kk_low_resource(text)

    def exact_kk_tsova_tush(self, text: str) -> tuple[KkEntry, ...]:
        return self.exact_kk_low_resource(text)

    def exact_kk_svan(self, text: str) -> tuple[KkEntry, ...]:
        return self.exact_kk_low_resource(text)

    def exact_kk_russian(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_russian_exact, text)

    def exact_kk_georgian(self, text: str) -> tuple[KkEntry, ...]:
        return self._rows_for_exact(self.kk_entries, self._kk_georgian_exact, text)

    def exact_translation_overrides(
        self,
        source_language: str,
        target_language: str,
        source_text: str,
    ) -> tuple[TranslationOverride, ...]:
        key = _translation_override_key(source_language, target_language, source_text)
        row_indexes = self._translation_override_exact.get(key, ())
        return tuple(self.translation_overrides[row_index] for row_index in row_indexes)

    @lru_cache(maxsize=4096)
    def search_sentence_pairs(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.sentence_pairs,
            index=self._sentence_index,
            standalone_text=lambda row: (row.low_resource, row.english),
            substring_text=lambda row: f"{row.low_resource}\t{row.english}",
            format_entry=lambda row: f"{self.low_resource_label}: {row.low_resource}\nEnglish: {row.english}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_sentence_low_resource(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.sentence_pairs,
            index=self._sentence_low_resource_index,
            standalone_text=lambda row: (row.low_resource,),
            substring_text=lambda row: row.low_resource,
            format_entry=lambda row: f"{self.low_resource_label}: {row.low_resource}\nEnglish: {row.english}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_gal(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.gal_entries,
            index=self._gal_index,
            standalone_text=lambda row: (row.low_resource, row.russian),
            substring_text=lambda row: f"{row.russian}\t{row.low_resource}",
            format_entry=lambda row: f"{self.low_resource_label}: {row.low_resource}\nRussian: {row.russian}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_gal_low_resource(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.gal_entries,
            index=self._gal_low_resource_index,
            standalone_text=lambda row: (row.low_resource,),
            substring_text=lambda row: row.low_resource,
            format_entry=lambda row: f"{self.low_resource_label}: {row.low_resource}\nRussian: {row.russian}\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_kk(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.kk_entries,
            index=self._kk_index,
            standalone_text=lambda row: (row.low_resource, row.russian, row.georgian),
            substring_text=lambda row: "\t".join((row.low_resource, row.ipa, row.russian, row.georgian)),
            format_entry=lambda row: format_kk_entry(
                row.low_resource,
                row.russian,
                row.georgian,
                label=self.low_resource_label,
            )
            + "\n",
            standalone_only=standalone_only,
        )

    @lru_cache(maxsize=4096)
    def search_kk_low_resource(self, word: str, *, standalone_only: bool = False) -> SearchResult:
        return self._search_rows(
            word=word,
            rows=self.kk_entries,
            index=self._kk_low_resource_index,
            standalone_text=lambda row: (row.low_resource,),
            substring_text=lambda row: row.low_resource,
            format_entry=lambda row: format_kk_entry(
                row.low_resource,
                row.russian,
                row.georgian,
                label=self.low_resource_label,
            )
            + "\n",
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


def get_dictionary_store(pack_id: str = DEFAULT_PACK_ID) -> DictionaryStore:
    """Return a cached store, invalidated by source file modification times."""
    normalized_pack_id = normalize_pack_id(pack_id)
    return _get_dictionary_store_cached(
        normalized_pack_id,
        _data_file_cache_key("translation_overrides.tsv", normalized_pack_id),
        _data_file_cache_key("sentence_pairs.tsv", normalized_pack_id),
        _data_file_cache_key("gal.tsv", normalized_pack_id),
        _data_file_cache_key("kk.tsv", normalized_pack_id),
        _context_file_cache_key(normalized_pack_id),
    )


@lru_cache(maxsize=4)
def _get_dictionary_store_cached(
    pack_id: str,
    translation_overrides_key: tuple[str, Optional[int]],
    sentence_pairs_key: tuple[str, Optional[int]],
    gal_key: tuple[str, Optional[int]],
    kk_key: tuple[str, Optional[int]],
    context_key: tuple[str, Optional[int]],
) -> DictionaryStore:
    return DictionaryStore(
        pack_id=pack_id,
        translation_overrides=_load_translation_overrides(*translation_overrides_key),
        sentence_pairs=_load_sentence_pairs(*sentence_pairs_key, pack_id),
        gal_entries=_load_gal_entries(*gal_key, pack_id),
        kk_entries=_load_kk_entries(*kk_key),
        context_key=context_key,
    )
