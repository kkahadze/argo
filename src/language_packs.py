#!/usr/bin/env python3
"""Language-pack metadata and low-resource translation hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

from src.tsova_tush.normalization import (
    detect_batsbi_scheme,
    format_batsbi_display_text,
    to_canonical_practical,
)


def _identity(text: str) -> str:
    return text


def _canonical_bats_target(text: str) -> str:
    scheme = detect_batsbi_scheme(text)
    if scheme in {"mkhedruli_batsbi", "latin_practical", "cyrillic_academic"}:
        return to_canonical_practical(text, scheme)
    return text


@dataclass(frozen=True)
class LanguagePack:
    code: str
    display_name: str
    data_dir_name: str
    grammar_heading: str
    dictionary_heading: str
    grammar_subject: str
    normalize_output: Callable[[str], str]
    canonicalize_lookup_target: Callable[[str], str]
    grammar_filename: str = "harris.txt"
    compact_grammar_filename: str = "harris_compact.txt"
    mkhedruli_grammar_filename: str | None = None
    ipa_grammar_filename: str | None = None


LANGUAGE_PACKS: Final[dict[str, LanguagePack]] = {
    "mingrelian": LanguagePack(
        code="mingrelian",
        display_name="Mingrelian",
        data_dir_name="mingrelian",
        grammar_heading="Here is the Mingrelian grammar information:",
        dictionary_heading="Mingrelian dictionaries",
        grammar_subject="Mingrelian",
        normalize_output=_identity,
        canonicalize_lookup_target=_identity,
    ),
    "tsova_tush": LanguagePack(
        code="tsova_tush",
        display_name="Bats",
        data_dir_name="tsova_tush",
        grammar_heading="Here is the Bats grammar information:",
        dictionary_heading="Bats dictionaries",
        grammar_subject="Bats",
        normalize_output=lambda text: format_batsbi_display_text(
            _canonical_bats_target(text)
        ),
        canonicalize_lookup_target=_canonical_bats_target,
    ),
    "svan": LanguagePack(
        code="svan",
        display_name="Svan",
        data_dir_name="svan",
        grammar_heading="Here is the Svan grammar information:",
        dictionary_heading="Svan dictionaries",
        grammar_subject="Svan",
        normalize_output=_identity,
        canonicalize_lookup_target=_identity,
        grammar_filename="tuite.txt",
        compact_grammar_filename="tuite_compact.txt",
        mkhedruli_grammar_filename="tuite_mkhedruli.txt",
        ipa_grammar_filename="tuite_ipa.txt",
    ),
}

SUPPORTED_TRANSLATION_PAIRS: Final[set[tuple[str, str]]] = {
    ("mingrelian", "english"),
    ("english", "mingrelian"),
    ("mingrelian", "georgian"),
    ("georgian", "mingrelian"),
    ("tsova_tush", "english"),
    ("english", "tsova_tush"),
    ("tsova_tush", "georgian"),
    ("georgian", "tsova_tush"),
    ("svan", "english"),
    ("english", "svan"),
    ("svan", "georgian"),
    ("georgian", "svan"),
    ("english", "georgian"),
    ("georgian", "english"),
}


def get_language_pack(language_code: str) -> LanguagePack:
    """Return a registered low-resource language pack."""
    try:
        return LANGUAGE_PACKS[language_code]
    except KeyError as exc:
        raise KeyError(f"Unknown language pack: {language_code}") from exc


def get_low_resource_pack_for_pair(source_lang: str, target_lang: str) -> LanguagePack | None:
    """Resolve the low-resource language pack for a supported translation pair."""
    source_pack = LANGUAGE_PACKS.get(source_lang)
    target_pack = LANGUAGE_PACKS.get(target_lang)
    if source_pack and target_pack and source_pack.code != target_pack.code:
        return None
    return source_pack or target_pack


def is_supported_translation_pair(source_lang: str, target_lang: str) -> bool:
    """Return True for translation directions intentionally supported by the shared engine."""
    return (source_lang, target_lang) in SUPPORTED_TRANSLATION_PAIRS
