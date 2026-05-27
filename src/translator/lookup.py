#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import os
import string
from typing import Callable, Optional

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

from src.logger import setup_logger
from src.dictionary_store import get_dictionary_store
from src.language_packs import get_low_resource_pack_for_pair

from src.translator.data import (
    _load_context_source_entries,
    _load_master_lexicon_rows,
)
from src.translator.text_utils import (
    LANG_LABEL,
    LOW_VALUE_LOOKUP_TERMS,
    LOOKUP_SEPARATOR,
    _choose_kk_bridge_gloss,
    _format_token_candidate_block,
    _is_low_value_lookup_term,
    _is_standalone_match,
    _is_substring_match,
    _lookup_variants_for_token,
    _normalize_lookup_value,
    _split_figurative_gloss,
    _truncate_lookup_output,
)

logger = setup_logger('translator')
MAX_RETRIEVAL_CONTEXT_CHARS = int(os.getenv("ARGO_MAX_RETRIEVAL_CONTEXT_CHARS", "8000"))


def _pack_for_pair(source_lang: str, target_lang: str):
    return get_low_resource_pack_for_pair(source_lang, target_lang)


def _pack_code_for_pair(source_lang: str, target_lang: str) -> str:
    pack = _pack_for_pair(source_lang, target_lang)
    return pack.code if pack else "mingrelian"


def _low_resource_label(source_lang: str, target_lang: str) -> str:
    pack = _pack_for_pair(source_lang, target_lang)
    return pack.display_name if pack else "Mingrelian"

def _collect_master_lexicon_exact_candidates(
    input_text: str,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    """Collect exact-match candidates from the master lexicon."""
    input_normalized = _normalize_lookup_value(input_text)
    candidates: list[dict[str, str]] = []

    pack_id = _pack_code_for_pair(source_lang, target_lang)
    for headword, headword_raw, translation in _load_master_lexicon_rows(pack_id=pack_id):

        matched_fields: list[str] = []
        if headword and _normalize_lookup_value(headword) == input_normalized:
            matched_fields.append("headword")
        if headword_raw and _normalize_lookup_value(headword_raw) == input_normalized:
            matched_fields.append("headword_raw")
        if translation and _normalize_lookup_value(translation) == input_normalized:
            matched_fields.append("translation")

        if not matched_fields:
            continue

        if source_lang == pack_id and target_lang == "english" and any(
            field in {"headword", "headword_raw"} for field in matched_fields
        ):
            candidates.append(
                {
                    "source_name": "master_lexicon",
                    "target_text": translation,
                    "headword": headword,
                    "headword_raw": headword_raw,
                    "translation": translation,
                    "matched_on": ", ".join(field for field in matched_fields if field in {"headword", "headword_raw"}),
                }
            )
        elif source_lang == "english" and target_lang == pack_id and "translation" in matched_fields:
            candidates.append(
                {
                    "source_name": "master_lexicon",
                    "target_text": headword,
                    "headword": headword,
                    "headword_raw": headword_raw,
                    "translation": translation,
                    "matched_on": "translation",
                }
            )

    return candidates


def _collect_simple_exact_match_candidates(
    input_text: str,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    """Collect exact-match candidates from the existing extractive dictionaries."""
    candidates: list[dict[str, str]] = []
    pack_id = _pack_code_for_pair(source_lang, target_lang)
    store = get_dictionary_store(pack_id)

    for row in store.exact_translation_overrides(source_lang, target_lang, input_text):
        candidates.append(
            {
                "source_name": "translation_overrides",
                "target_text": row.target_text,
                "headword": row.source_text,
                "translation": row.target_text,
                "matched_on": f"{row.source_language}->{row.target_language}",
            }
        )

    if candidates:
        return candidates

    # sentence_pairs.tsv (low-resource ↔ English)
    if (source_lang, target_lang) in [(pack_id, "english"), ("english", pack_id)]:
        if source_lang == pack_id:
            for row in store.exact_sentence_low_resource(input_text):
                candidates.append(
                    {
                        "source_name": "sentence_pairs",
                        "target_text": row.english,
                        "headword": row.low_resource,
                        "translation": row.english,
                        "matched_on": pack_id,
                    }
                )
        else:
            for row in store.exact_sentence_english(input_text):
                candidates.append(
                    {
                        "source_name": "sentence_pairs",
                        "target_text": row.low_resource,
                        "headword": row.low_resource,
                        "translation": row.english,
                        "matched_on": "english",
                    }
                )

    # kk.tsv (low-resource ↔ Georgian)
    if source_lang == pack_id and target_lang == "georgian":
        for row in store.exact_kk_low_resource(input_text):
            georgian_primary, _ = _split_figurative_gloss(row.georgian, "ka")
            candidates.append(
                {
                    "source_name": "kk.tsv",
                    "target_text": georgian_primary or row.georgian,
                    "headword": row.low_resource,
                    "translation": georgian_primary or row.georgian,
                    "matched_on": pack_id,
                }
            )
    elif source_lang == "georgian" and target_lang == pack_id:
        for row in store.exact_kk_georgian(input_text):
            candidates.append(
                {
                    "source_name": "kk.tsv",
                    "target_text": row.low_resource,
                    "headword": row.low_resource,
                    "translation": row.georgian,
                    "matched_on": "georgian",
                }
            )

    return candidates


def collect_exact_match_candidates(
    input_text: str,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    """Collect and deduplicate exact-match candidates across supported lexicon sources."""
    combined = _collect_simple_exact_match_candidates(input_text, source_lang, target_lang)
    combined.extend(_collect_master_lexicon_exact_candidates(input_text, source_lang, target_lang))

    deduped: list[dict[str, str]] = []
    index_by_target: dict[str, int] = {}

    for candidate in combined:
        target_text = candidate.get("target_text", "")
        pack = _pack_for_pair(source_lang, target_lang)
        target_key = _normalize_lookup_value(
            pack.canonicalize_lookup_target(target_text)
            if pack and target_lang == pack.code
            else target_text
        )
        if not target_key:
            continue

        existing_index = index_by_target.get(target_key)
        if existing_index is None:
            merged = dict(candidate)
            merged["source_name"] = candidate["source_name"]
            deduped.append(merged)
            index_by_target[target_key] = len(deduped) - 1
            continue

        existing = deduped[existing_index]
        source_names = existing["source_name"].split(", ")
        if candidate["source_name"] not in source_names:
            existing["source_name"] = ", ".join(source_names + [candidate["source_name"]])

        if candidate.get("matched_on"):
            existing_matches = [part.strip() for part in existing.get("matched_on", "").split(",") if part.strip()]
            for match_part in [part.strip() for part in candidate["matched_on"].split(",") if part.strip()]:
                if match_part not in existing_matches:
                    existing_matches.append(match_part)
            existing["matched_on"] = ", ".join(existing_matches)

        for field in ("headword", "headword_raw", "translation"):
            if not existing.get(field) and candidate.get(field):
                existing[field] = candidate[field]

    return deduped


def _collect_token_exact_candidates(word: str, source_lang: str, target_lang: str) -> list[dict[str, str]]:
    """Collect and deduplicate compact exact candidates for a single token."""
    combined: list[dict[str, str]] = []
    for variant in _lookup_variants_for_token(word, source_lang):
        combined.extend(collect_exact_match_candidates(variant, source_lang, target_lang))

    deduped: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for candidate in combined:
        target_key = _normalize_lookup_value(candidate.get("target_text", ""))
        if not target_key or target_key in seen_targets:
            continue
        deduped.append(candidate)
        seen_targets.add(target_key)
    return deduped


def _join_ranked_context(blocks: list[str], max_chars: int = MAX_RETRIEVAL_CONTEXT_CHARS) -> str:
    """Join prioritized retrieval blocks once, with one total prompt budget."""
    retained: list[str] = []
    seen: set[str] = set()
    chars_used = 0
    for block in blocks:
        cleaned = (block or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        separator = "\n\n" if retained else ""
        remaining = max_chars - chars_used - len(separator)
        if remaining <= 0:
            break
        retained.append(separator + cleaned[:remaining].rstrip())
        chars_used += len(retained[-1])
        if chars_used >= max_chars:
            break
    return "".join(retained)


def _build_high_resource_to_mingrelian_dict_entries(
    sentence: str,
    *,
    input_lang: str,
    lookup_fn: Callable[[str], str],
) -> str:
    """
    Build higher-signal prompt context for English/Georgian -> Mingrelian.

    Strategy:
    - Prefer compact exact candidate lists for meaningful tokens.
    - Skip low-value function words unless they have exact candidates.
    - For English inputs, translate the full sentence to Georgian once and use that
      bridge sentence for token lookup, which preserves word-sense better than
      translating each English token independently.
    """
    lookup_sentence = sentence
    lookup_source_lang = input_lang
    effective_lookup_fn = lookup_fn
    blocks: list[str] = []

    if input_lang == "english" and GoogleTranslator is not None:
        try:
            georgian_bridge = GoogleTranslator(source="en", target="ka").translate(sentence)
        except Exception:
            georgian_bridge = ""
        if georgian_bridge:
            lookup_sentence = georgian_bridge
            lookup_source_lang = "georgian"
            effective_lookup_fn = grep_search_from_georgian
            blocks.append(
                "High-resource bridge translation of the full input for sense disambiguation:\n"
                f"- Georgian: {georgian_bridge}\n"
                "Use this only as context; the final answer must still be Mingrelian."
            )

    tokens = [
        word.strip(string.punctuation + "“”„’'\"")
        for word in lookup_sentence.split()
        if word.strip(string.punctuation + "“”„’'\"")
    ]

    for token in tokens:
        exact_candidates = _collect_token_exact_candidates(token, lookup_source_lang, "mingrelian")
        if exact_candidates:
            blocks.append(
                _format_token_candidate_block(
                    token=token,
                    source_lang=lookup_source_lang,
                    candidates=exact_candidates,
                )
            )
            continue

        if _is_low_value_lookup_term(token, lookup_source_lang, len(tokens)):
            continue

        token_lookup = effective_lookup_fn(token)
        if token_lookup and token_lookup.strip():
            blocks.append(token_lookup)

    return "\n".join(block.strip() for block in blocks if block and block.strip())


def _build_high_resource_to_low_resource_dict_entries(
    sentence: str,
    *,
    input_lang: str,
    lookup_fn: Callable[[str], str],
    pack_id: str,
    target_label: str,
) -> str:
    """Build higher-signal prompt context for English/Georgian -> one target pack."""
    lookup_sentence = sentence
    lookup_source_lang = input_lang
    effective_lookup_fn = lookup_fn
    blocks: list[str] = []

    if input_lang == "english" and GoogleTranslator is not None:
        try:
            georgian_bridge = GoogleTranslator(source="en", target="ka").translate(sentence)
        except Exception:
            georgian_bridge = ""
        if georgian_bridge:
            lookup_sentence = georgian_bridge
            lookup_source_lang = "georgian"
            effective_lookup_fn = lambda token: grep_search_from_georgian_for_pack(token, pack_id)
            blocks.append(
                "High-resource bridge translation of the full input for sense disambiguation:\n"
                f"- Georgian: {georgian_bridge}\n"
                f"Use this only as context; the final answer must still be {target_label}."
            )

    tokens = [
        word.strip(string.punctuation + "“”„’'\"")
        for word in lookup_sentence.split()
        if word.strip(string.punctuation + "“”„’'\"")
    ]

    for token in tokens:
        exact_candidates = _collect_token_exact_candidates(token, lookup_source_lang, pack_id)
        if exact_candidates:
            blocks.append(
                _format_token_candidate_block(
                    token=token,
                    source_lang=lookup_source_lang,
                    candidates=exact_candidates,
                    target_label=target_label,
                )
            )
            continue

        if _is_low_value_lookup_term(token, lookup_source_lang, len(tokens)):
            continue

        token_lookup = effective_lookup_fn(token)
        if token_lookup and token_lookup.strip():
            blocks.append(token_lookup)

    return "\n".join(block.strip() for block in blocks if block and block.strip())


def _build_high_resource_to_tsova_tush_dict_entries(
    sentence: str,
    *,
    input_lang: str,
    lookup_fn: Callable[[str], str],
) -> str:
    """Build higher-signal prompt context for English/Georgian -> Bats."""
    return _build_high_resource_to_low_resource_dict_entries(
        sentence,
        input_lang=input_lang,
        lookup_fn=lookup_fn,
        pack_id="tsova_tush",
        target_label="Bats",
    )


def _build_high_resource_to_svan_dict_entries(
    sentence: str,
    *,
    input_lang: str,
    lookup_fn: Callable[[str], str],
) -> str:
    """Build higher-signal prompt context for English/Georgian -> Svan."""
    return _build_high_resource_to_low_resource_dict_entries(
        sentence,
        input_lang=input_lang,
        lookup_fn=lookup_fn,
        pack_id="svan",
        target_label="Svan",
    )


def _build_svan_to_georgian_dict_entries(
    sentence: str,
    *,
    lookup_fn: Callable[[str], str],
) -> str:
    """Build bounded, source-side-only retrieval context for Svan -> Georgian."""
    tokens = [
        word.strip(string.punctuation + "“”„’'\"")
        for word in sentence.split()
        if word.strip(string.punctuation + "“”„’'\"")
    ]
    exact_blocks: list[str] = []
    fallback_blocks: list[str] = []

    for token in tokens:
        exact_candidates = _collect_token_exact_candidates(token, "svan", "georgian")
        if exact_candidates:
            exact_blocks.append(
                _format_token_candidate_block(
                    token=token,
                    source_lang="svan",
                    candidates=exact_candidates,
                    target_label="Georgian",
                )
            )
            continue
        if _is_low_value_lookup_term(token, "svan", len(tokens)):
            continue
        token_lookup = lookup_fn(token)
        if token_lookup and token_lookup.strip():
            fallback_blocks.append(token_lookup)

    return _join_ranked_context(exact_blocks + fallback_blocks)


def grep_search_pairs(
    word: str,
    *,
    standalone_only: bool = False,
    pack_id: str = "mingrelian",
) -> tuple[str, bool]:
    """
    Search sentence_pairs.tsv for English translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store(pack_id).search_sentence_pairs(word, standalone_only=standalone_only)
    return _truncate_lookup_output(result.output), result.has_standalone_matches


def grep_search_gal(
    word: str,
    *,
    standalone_only: bool = False,
    pack_id: str = "mingrelian",
) -> tuple[str, bool]:
    """
    Search gal.tsv for Russian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store(pack_id).search_gal(word, standalone_only=standalone_only)
    return _truncate_lookup_output(result.output), result.has_standalone_matches


def grep_search_kk(
    word: str,
    *,
    standalone_only: bool = False,
    pack_id: str = "mingrelian",
) -> tuple[str, bool]:
    """
    Search kk.tsv for Russian and Georgian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store(pack_id).search_kk(word, standalone_only=standalone_only)
    return _truncate_lookup_output(result.output), result.has_standalone_matches


def grep_search_context_source(
    word: str,
    *,
    standalone_only: bool = False,
    pack_id: str = "mingrelian",
    match_label: str | None = None,
) -> str:
    """
    Search context_source.txt for relevant entry blocks.
    Splits text by empty lines and returns the block containing the search term.
    Prioritizes standalone word matches over substring matches.
    """
    entries = _load_context_source_entries(pack_id=pack_id)
    if not entries:
        return ""

    # First pass: look for standalone word matches
    standalone_output = LOOKUP_SEPARATOR
    substring_output = LOOKUP_SEPARATOR

    for entry in entries:
        search_text = entry
        if match_label:
            search_text = "\n".join(
                line for line in entry.splitlines()
                if line.startswith(f"{match_label}:")
            )
            if not search_text:
                continue
        if _is_standalone_match(search_text, word):
            # Standalone match found
            standalone_output += entry.strip()
            standalone_output += "\n" + LOOKUP_SEPARATOR
        elif _is_substring_match(search_text, word):
            # Substring match
            substring_output += entry.strip()
            substring_output += "\n" + LOOKUP_SEPARATOR

    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(standalone_output)
    elif (not standalone_only) and substring_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(substring_output)
    return ""


def grep_search_from_english(word: str) -> str:
    return grep_search_from_english_for_pack(word, "mingrelian")


def grep_search_from_english_for_pack(word: str, pack_id: str) -> str:
    """
    Search all dictionaries from English word.
    Short-circuits context-source search if standalone matches found in extractive dictionaries.
    """
    # Try to translate to Russian and Georgian for broader search
    res_ru = word
    res_ge = word

    if GoogleTranslator is not None:
        try:
            res_ru = GoogleTranslator(source='en', target='ru').translate(word)
            res_ge = GoogleTranslator(source='en', target='ka').translate(word)
        except Exception:
            pass  # Fall back to original word if translation fails

    output = f"\nResults for {word}:\n"

    # Run extractive dictionary searches
    pairs_result, pairs_has_standalone = grep_search_pairs(word, pack_id=pack_id)
    gal_result, gal_has_standalone = grep_search_gal(res_ru, pack_id=pack_id)
    kk_ru_result, kk_ru_has_standalone = grep_search_kk(res_ru, pack_id=pack_id)
    kk_ge_result, kk_ge_has_standalone = grep_search_kk(res_ge, pack_id=pack_id)

    output += pairs_result
    output += gal_result
    output += kk_ru_result
    output += kk_ge_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = (pairs_has_standalone or gal_has_standalone or
                          kk_ru_has_standalone or kk_ge_has_standalone)

    if not has_any_standalone:
        output += grep_search_context_source(res_ge, pack_id=pack_id)

    return _truncate_lookup_output(output)


def _grep_search_from_low_resource(word: str, pack_id: str, display_name: str) -> str:
    """
    Search all dictionaries from Mingrelian word.
    Short-circuits context-source search if standalone matches found in extractive dictionaries.
    """
    def _mkhedruli_has_letters(s: str) -> bool:
        return any('\u10D0' <= ch <= '\u10FF' for ch in s)

    def _ends_with_mkhedruli_vowel(s: str) -> bool:
        if not s:
            return False
        # Include schwa letter (ჷ) as vowel-like in our data.
        return s[-1] in {"ა", "ე", "ი", "ო", "უ", "ჷ"}

    def _case_strip_candidates_mkhedruli(w: str) -> list[str]:
        """
        Conservative case-ending stripping for Mingrelian nouns/adjectives.
        Based on Harris (§2.1.1.1): cases include nominative -i, narrative -k, dative -s,
        genitive -iš, allative -iša, ablative -iše, instrumental -it, designative -išo(t),
        adverbial -o(t). After vowel-final stems, initial i drops (so -š, -ša, -še, -t, -šo(t)).

        We keep this intentionally narrow to avoid trash results:
        - only run if no dictionary matches were found across all four sources
        - strip ONE suffix per attempt (longest-first), then optionally strip plural
        - generate a small set of candidates (including optional +ი for consonant-final stems)
        """
        w = (w or "").strip()
        if len(w) < 4 or not _mkhedruli_has_letters(w):
            return []

        EMPHATIC_VOWELS = ["ი", "ჷ", "უ"]  # Harris: -i, -ə, -u (Mkhedruli schwa ≈ ჷ)

        # Base case endings (Harris §2.1.1.1), longest-first to avoid partial stripping.
        base_suffixes = [
            # Ablative
            "იშე",   # -iše (consonant stems)
            "შე",    # -še  (vowel stems)
            # Allative
            "იშა",   # -iša (consonant stems)
            "შა",    # -ša  (vowel stems)
            # Genitive
            "იშ",    # -iš  (consonant stems)
            "შ",     # -š   (vowel stems)
            # Designative
            "იშოთ",  # -išo(t) (consonant stems)
            "შოთ",   # -šo(t)  (vowel stems)
            # Instrumental / adverbial
            "ით",    # instrumental -it (consonant stems)
            "ოთ",    # adverbial -ot
            "ო",     # adverbial -o
            "თ",     # instrumental/adverbial -t (vowel stems)
            # Narrative / dative (single consonant markers)
            "კ",     # narrative -k
            "ს",     # dative -s
            "ც",     # dative variant -c (Z-S dialect, after consonant stems)
        ]

        # Emphatic vowels can follow narrative/dative/genitive/instrumental markers,
        # and occasionally adverbial/designative (Harris p.14).
        emphatic_applicable = [
            "კ", "ს", "ც",     # narrative/dative
            "იშ", "შ",         # genitive
            "ით", "თ",         # instrumental (and t-variant)
            "ოთ", "ო",         # occasionally adverbial
            "იშოთ", "შოთ",     # occasionally designative
        ]

        # Add case+emphatic variants (e.g., კი, კჷ, კუ; იში, etc.).
        emphatic_suffixes = [s + v for s in emphatic_applicable for v in EMPHATIC_VOWELS]

        # Combine and ensure longest-first (then lexicographic for stability).
        suffixes = sorted(set(base_suffixes + emphatic_suffixes), key=lambda x: (-len(x), x))

        out: list[str] = []
        for suf in suffixes:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                stem = w[:-len(suf)]
                if len(stem) >= 3:
                    # Prefer nominative (+ი) form first for consonant-final stems,
                    # since dictionary headwords are typically nominative.
                    if not _ends_with_mkhedruli_vowel(stem):
                        out.append(stem + "ი")
                        out.append(stem)
                    else:
                        out.append(stem)

                    # Optional plural stripping (Harris §2.1.1.2: plural marker -ef in Mkhedruli = "ეფ")
                    # Only after a successful case-strip.
                    # Also handle narrative/dative plural allomorphs -ენ/-ემ (Harris: -en/-em),
                    # but only as a fallback after stripping a case marker.
                    plural_variants = ["ეფ", "ენ", "ემ"]
                    for pv in plural_variants:
                        if stem.endswith(pv) and len(stem) >= len(pv) + 3:
                            base = stem[:-len(pv)]
                            if not _ends_with_mkhedruli_vowel(base):
                                out.append(base + "ი")
                                out.append(base)
                            else:
                                out.append(base)
                break

        # De-dup while preserving order
        seen = set()
        deduped: list[str] = []
        for c in out:
            if c not in seen and c != w:
                deduped.append(c)
                seen.add(c)
        return deduped

    def _preverb_strip_candidates_mkhedruli(w: str) -> list[str]:
        """
        Verb-only fallback: strip SIMPLE preverbs (Ivanishvili & Soselia).
        Runs only if we got zero hits from all other lookup methods.

        Simple preverbs (Mkhedruli):
        - V-type: ა-, ო-, ე-
        - CV-type: გა-, გო-, გე-, და-, დო-, წა-, მე-, მო-, შე-
        """
        w = (w or "").strip()
        if len(w) < 5 or not _mkhedruli_has_letters(w):
            return []

        preverbs = [
            # CV-type (2 chars)
            "გა", "გო", "გე",
            "და", "დო",
            "წა",
            "მე", "მო",
            "შე",
            # V-type (1 char)
            "ა", "ო", "ე",
        ]

        out: list[str] = []
        for pv in preverbs:
            if w.startswith(pv) and len(w) > len(pv) + 2:
                out.append(w[len(pv):])
                break
        return out

    output = f"\nResults for {word}:\n"

    # Run extractive dictionary searches
    pairs_result, pairs_has_standalone = grep_search_pairs(word, pack_id=pack_id)
    gal_result, gal_has_standalone = grep_search_gal(word, pack_id=pack_id)
    kk_result, kk_has_standalone = grep_search_kk(word, pack_id=pack_id)

    output += pairs_result
    output += gal_result
    output += kk_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or gal_has_standalone or kk_has_standalone

    context_source_result = ""
    if not has_any_standalone:
        context_source_result = grep_search_context_source(word, pack_id=pack_id)
        output += context_source_result

    # If absolutely nothing matched across all four sources, try a conservative
    # case-suffix stripping fallback (e.g., ...თ → stem).
    case_fallback_applied = False
    if not (pairs_result or gal_result or kk_result or context_source_result):
        for stem in _case_strip_candidates_mkhedruli(word):
            # First: standalone-only search. If we find any standalone match for this
            # candidate, we return ONLY standalone matches and stop (no partial matches,
            # and no further candidates like the bare stem).
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True, pack_id=pack_id)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True, pack_id=pack_id)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True, pack_id=pack_id)
            context_source2_s = grep_search_context_source(stem, standalone_only=True, pack_id=pack_id)

            output2_s = pairs2_s + gal2_s + kk2_s + context_source2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(context_source2_s)

            if has_any_standalone2:
                output += f"\n[Case-stripped fallback: {word} → {stem}]\n"
                output += output2_s
                case_fallback_applied = True
                break

            # Otherwise, fall back to the normal grep-style matching for this candidate.
            pairs2, pairs2_has = grep_search_pairs(stem, pack_id=pack_id)
            gal2, gal2_has = grep_search_gal(stem, pack_id=pack_id)
            kk2, kk2_has = grep_search_kk(stem, pack_id=pack_id)

            output2 = pairs2 + gal2 + kk2
            has_any_standalone2 = pairs2_has or gal2_has or kk2_has
            context_source2 = "" if has_any_standalone2 else grep_search_context_source(stem, pack_id=pack_id)
            output2 += context_source2

            if output2:
                output += f"\n[Case-stripped fallback: {word} → {stem}]\n"
                output += output2
                case_fallback_applied = True
                break

    # If we STILL have no hits, assume this might be a verb with a preverb attached
    # and try stripping a simple preverb.
    if not (pairs_result or gal_result or kk_result or context_source_result or case_fallback_applied):
        for stem in _preverb_strip_candidates_mkhedruli(word):
            # Prefer standalone-only results for the stripped stem.
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True, pack_id=pack_id)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True, pack_id=pack_id)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True, pack_id=pack_id)
            context_source2_s = grep_search_context_source(stem, standalone_only=True, pack_id=pack_id)

            output2_s = pairs2_s + gal2_s + kk2_s + context_source2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(context_source2_s)

            if has_any_standalone2:
                output += f"\n[Preverb-stripped fallback: {word} → {stem}]\n"
                output += output2_s
                break

            # Otherwise allow normal grep-style matches.
            pairs2, pairs2_has = grep_search_pairs(stem, pack_id=pack_id)
            gal2, gal2_has = grep_search_gal(stem, pack_id=pack_id)
            kk2, kk2_has = grep_search_kk(stem, pack_id=pack_id)

            output2 = pairs2 + gal2 + kk2
            has_any_standalone2 = pairs2_has or gal2_has or kk2_has
            context_source2 = "" if has_any_standalone2 else grep_search_context_source(stem, pack_id=pack_id)
            output2 += context_source2

            if output2:
                output += f"\n[Preverb-stripped fallback: {word} → {stem}]\n"
                output += output2
                break

    return _truncate_lookup_output(output)


def grep_search_from_mingrelian(word: str) -> str:
    """Search all dictionaries from Mingrelian word."""
    return _grep_search_from_low_resource(word, "mingrelian", "Mingrelian")


def grep_search_from_tsova_tush(word: str) -> str:
    """Search all dictionaries from Bats word."""
    return _grep_search_from_low_resource(word, "tsova_tush", "Bats")


def _grep_search_from_svan_source(word: str) -> str:
    """Search only Svan-side fields when translating from Svan."""
    store = get_dictionary_store("svan")
    results = [
        store.search_sentence_low_resource(word),
        store.search_gal_low_resource(word),
        store.search_kk_low_resource(word),
    ]
    output_parts = [result.output for result in results if result.output]
    if not any(result.has_standalone_matches for result in results):
        context_result = grep_search_context_source(
            word,
            pack_id="svan",
            match_label="Svan",
        )
        if context_result:
            output_parts.append(context_result)
    if not output_parts:
        return ""
    return _truncate_lookup_output(f"\nResults for {word}:\n" + "".join(output_parts))


def grep_search_from_svan(word: str) -> str:
    """Search all dictionaries from Svan word for existing non-Georgian paths."""
    return _grep_search_from_low_resource(word, "svan", "Svan")


def grep_search_from_georgian(word: str) -> str:
    return grep_search_from_georgian_for_pack(word, "mingrelian")


def grep_search_from_georgian_for_pack(word: str, pack_id: str) -> str:
    """
    Search all dictionaries from Georgian word.
    Short-circuits context-source search if standalone matches found in extractive dictionaries.
    """
    # Try to translate to English and Russian for broader search
    res_en = word
    res_ru = word

    if GoogleTranslator is not None:
        try:
            res_en = GoogleTranslator(source='ka', target='en').translate(word)
            res_ru = GoogleTranslator(source='ka', target='ru').translate(word)
        except Exception:
            pass  # Fall back to original word if translation fails

    output = f"\nResults for {word}:\n"

    # Run extractive dictionary searches
    pairs_result, pairs_has_standalone = grep_search_pairs(res_en, pack_id=pack_id)
    kk_result, kk_has_standalone = grep_search_kk(word, pack_id=pack_id)
    gal_result, gal_has_standalone = grep_search_gal(res_ru, pack_id=pack_id)

    output += pairs_result
    output += kk_result
    output += gal_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or kk_has_standalone or gal_has_standalone

    if not has_any_standalone:
        output += grep_search_context_source(word, pack_id=pack_id)

    return _truncate_lookup_output(output)


def check_exact_match_simple(input_text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Check if the exact input text exists in extractive dictionaries (not the context source).
    Returns the translation if found, None otherwise.

    This is the simple direct lookup without Google Translate augmentation.
    """
    pack_id = _pack_code_for_pair(source_lang, target_lang)
    store = get_dictionary_store(pack_id)

    overrides = store.exact_translation_overrides(source_lang, target_lang, input_text)
    if overrides:
        return overrides[0].target_text

    # Check sentence_pairs.tsv (low-resource ↔ English)
    if (source_lang, target_lang) in [(pack_id, "english"), ("english", pack_id)]:
        if source_lang == pack_id:
            matches = store.exact_sentence_low_resource(input_text)
            if matches:
                return matches[0].english
        else:
            matches = store.exact_sentence_english(input_text)
            if matches:
                return matches[0].low_resource

    # Check kk.tsv (low-resource ↔ Russian ↔ Georgian)
    if source_lang == pack_id and target_lang == "georgian":
        matches = store.exact_kk_low_resource(input_text)
        if matches:
            georgian_primary, _ = _split_figurative_gloss(matches[0].georgian, "ka")
            return georgian_primary or matches[0].georgian

    elif source_lang == "georgian" and target_lang == pack_id:
        matches = store.exact_kk_georgian(input_text)
        if matches:
            return matches[0].low_resource

    # Check gal.tsv (Russian ↔ low-resource)
    if source_lang == "russian" and target_lang == pack_id:
        matches = store.exact_gal_russian(input_text)
        if matches:
            return matches[0].low_resource

    elif source_lang == pack_id and target_lang == "russian":
        matches = store.exact_gal_low_resource(input_text)
        if matches:
            return matches[0].russian

    return None


def find_low_resource_in_dicts(
    text: str,
    *,
    pack_id: str = "mingrelian",
    target_lang: Optional[str] = None,
) -> Optional[tuple[str, str, str]]:
    """
    Find ANY translation for a text in dictionaries, searching across all columns.
    Returns (low_resource, other_language_text, other_language_code) if found.

    Search order prioritizes clean extractive dictionaries (sentence_pairs, gal) over kk.

    Args:
        text: Text to search for (case-insensitive)

    Returns:
        tuple or None: (low_resource_text, other_lang_text, lang_code) if found
    """
    store = get_dictionary_store(pack_id)

    # Priority 1: Search sentence_pairs.tsv (English ↔ low-resource, cleanest)
    for row in store.exact_sentence_low_resource(text):
        return (row.low_resource, row.english, "en")
    for row in store.exact_sentence_english(text):
        return (row.low_resource, row.english, "en")

    # Priority 2: Search gal.tsv (Russian ↔ low-resource, reliable)
    for row in store.exact_gal_low_resource(text):
        return (row.low_resource, row.russian, "ru")
    for row in store.exact_gal_russian(text):
        return (row.low_resource, row.russian, "ru")

    # Priority 3: Search kk.tsv (may have data quality issues, use as fallback)
    for row in store.exact_kk_low_resource(text):
        bridge_text, lang_code = _choose_kk_bridge_gloss(row.russian, row.georgian, target_lang)
        if bridge_text and lang_code:
            return (row.low_resource, bridge_text, lang_code)
        return (row.low_resource, row.georgian, "ka")
    for row in store.exact_kk_georgian(text):
        georgian_primary, _ = _split_figurative_gloss(row.georgian, "ka")
        return (row.low_resource, georgian_primary or row.georgian, "ka")
    for row in store.exact_kk_russian(text):
        russian_primary, _ = _split_figurative_gloss(row.russian, "ru")
        return (row.low_resource, russian_primary or row.russian, "ru")

    return None


def find_mingrelian_in_dicts(text: str, target_lang: Optional[str] = None) -> Optional[tuple[str, str, str]]:
    return find_low_resource_in_dicts(text, pack_id="mingrelian", target_lang=target_lang)


def find_tsova_tush_in_dicts(text: str, target_lang: Optional[str] = None) -> Optional[tuple[str, str, str]]:
    return find_low_resource_in_dicts(text, pack_id="tsova_tush", target_lang=target_lang)


def find_svan_in_dicts(text: str, target_lang: Optional[str] = None) -> Optional[tuple[str, str, str]]:
    return find_low_resource_in_dicts(text, pack_id="svan", target_lang=target_lang)


def check_exact_match_with_google_translate(input_text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Advanced exact match using Google Translate to bridge high-resource languages.

    SCENARIO 1: Translating TO a low-resource target (from English/Georgian)
    - Translate input to all high-resource languages (en, ka, ru)
    - Search dictionaries for each translated version
    - Return the low-resource match if found

    SCENARIO 2: Translating FROM a low-resource source (to English/Georgian)
    - Search dictionaries for that low-resource word
    - If found with any high-resource language pair
    - Google Translate that language to target
    - Return translation
    """
    if GoogleTranslator is None:
        return None

    pack = _pack_for_pair(source_lang, target_lang)
    pack_code = pack.code if pack else None
    pack_label = pack.display_name if pack else "Mingrelian"

    # SCENARIO 1: Translating TO low-resource language from high-resource language
    if pack_code and target_lang == pack_code and source_lang in ["english", "georgian"]:
        # Try direct lookup first
        direct_match = check_exact_match_simple(input_text, source_lang, target_lang)
        if direct_match:
            return direct_match

        # Translate to other high-resource languages and search
        translations_to_try = [(input_text, source_lang)]  # Start with original

        try:
            # Translate to other languages
            if source_lang == "english":
                # en → ka, en → ru
                ka_trans = GoogleTranslator(source="en", target="ka").translate(input_text)
                ru_trans = GoogleTranslator(source="en", target="ru").translate(input_text)
                translations_to_try.extend([(ka_trans, "georgian"), (ru_trans, "russian")])

            elif source_lang == "georgian":
                # ka → en, ka → ru
                en_trans = GoogleTranslator(source="ka", target="en").translate(input_text)
                ru_trans = GoogleTranslator(source="ka", target="ru").translate(input_text)
                translations_to_try.extend([(en_trans, "english"), (ru_trans, "russian")])

        except Exception:
            pass  # If translation fails, continue with what we have

        # Search for each translated version in dictionaries. Prefer the richer
        # exact-candidate collector so bridged variants can benefit from the
        # master lexicon without forcing an arbitrary ambiguous return.
        for translated_text, lang in translations_to_try:
            exact_candidates = collect_exact_match_candidates(translated_text, lang, pack_code)
            if len(exact_candidates) == 1:
                match = exact_candidates[0]["target_text"]
                logger.info(
                    f"[GOOGLE BRIDGE TO {pack_label.upper()}] {input_text} ({source_lang}) → "
                    f"{translated_text} ({lang}) → {match} ({pack_code})"
                )
                return match

            match = check_exact_match_simple(translated_text, lang, pack_code)
            if match:
                logger.info(
                    f"[GOOGLE BRIDGE TO {pack_label.upper()}] {input_text} ({source_lang}) → "
                    f"{translated_text} ({lang}) → {match} ({pack_code})"
                )
                return match

    # SCENARIO 2: Translating FROM low-resource language to high-resource language
    elif pack_code and source_lang == pack_code and target_lang in ["english", "georgian"]:
        # Try direct lookup first
        direct_match = check_exact_match_simple(input_text, source_lang, target_lang)
        if direct_match:
            return direct_match

        # Search for the low-resource term in ANY dictionary with ANY language pair
        result = find_low_resource_in_dicts(input_text, pack_id=pack_code, target_lang=target_lang)
        if result:
            low_resource_text, other_lang_text, lang_code = result

            # If the found language IS the target, return directly
            lang_map = {"en": "english", "ka": "georgian", "ru": "russian"}
            found_lang = lang_map.get(lang_code)

            if found_lang == target_lang:
                logger.info(
                    "[DIRECT DICT MATCH] %s (%s) → %s (%s)",
                    low_resource_text,
                    pack_code,
                    other_lang_text,
                    target_lang,
                )
                return other_lang_text

            # Otherwise, Google Translate from found language to target
            try:
                if target_lang == "english":
                    target_code = "en"
                elif target_lang == "georgian":
                    target_code = "ka"
                else:
                    return None

                translated = GoogleTranslator(source=lang_code, target=target_code).translate(other_lang_text)
                logger.info(
                    "[GOOGLE BRIDGE FROM %s] %s (%s) → %s (%s) → %s (%s)",
                    pack_label.upper(),
                    low_resource_text,
                    pack_code,
                    other_lang_text,
                    lang_code,
                    translated,
                    target_code,
                )
                return translated

            except Exception as e:
                logger.warning("[GOOGLE BRIDGE ERROR] Failed to translate: %s", e)
                pass

    return None
