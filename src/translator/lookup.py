#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import string
from typing import Callable, Optional

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

from src.logger import setup_logger

from src.translator.data import (
    _load_gal_rows,
    _load_kk_rows,
    _load_sentence_pairs_rows,
    _load_context_source_entries,
    _load_master_lexicon_rows,
)
from src.translator.text_utils import (
    LANG_LABEL,
    LOW_VALUE_LOOKUP_TERMS,
    LOOKUP_SEPARATOR,
    _choose_kk_bridge_gloss,
    _format_kk_entry,
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

def _collect_master_lexicon_exact_candidates(
    input_text: str,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, str]]:
    """Collect exact-match candidates from the master lexicon."""
    input_normalized = _normalize_lookup_value(input_text)
    candidates: list[dict[str, str]] = []

    for headword, headword_raw, translation in _load_master_lexicon_rows():

        matched_fields: list[str] = []
        if headword and _normalize_lookup_value(headword) == input_normalized:
            matched_fields.append("headword")
        if headword_raw and _normalize_lookup_value(headword_raw) == input_normalized:
            matched_fields.append("headword_raw")
        if translation and _normalize_lookup_value(translation) == input_normalized:
            matched_fields.append("translation")

        if not matched_fields:
            continue

        if source_lang == "mingrelian" and target_lang == "english" and any(
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
        elif source_lang == "english" and target_lang == "mingrelian" and "translation" in matched_fields:
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
    input_normalized = _normalize_lookup_value(input_text)
    candidates: list[dict[str, str]] = []

    # sentence_pairs.tsv (Mingrelian ↔ English)
    if (source_lang, target_lang) in [("mingrelian", "english"), ("english", "mingrelian")]:
        for mingrelian, english in _load_sentence_pairs_rows():
            if source_lang == "mingrelian" and _normalize_lookup_value(mingrelian) == input_normalized:
                candidates.append(
                    {
                        "source_name": "sentence_pairs",
                        "target_text": english,
                        "headword": mingrelian,
                        "translation": english,
                        "matched_on": "mingrelian",
                    }
                )
            elif source_lang == "english" and _normalize_lookup_value(english) == input_normalized:
                candidates.append(
                    {
                        "source_name": "sentence_pairs",
                        "target_text": mingrelian,
                        "headword": mingrelian,
                        "translation": english,
                        "matched_on": "english",
                    }
                )

    # kk.tsv (Mingrelian ↔ Georgian)
    for mingrelian, _, russian, georgian in _load_kk_rows():
        if source_lang == "mingrelian" and target_lang == "georgian":
            if _normalize_lookup_value(mingrelian) == input_normalized:
                georgian_primary, _ = _split_figurative_gloss(georgian, "ka")
                candidates.append(
                    {
                        "source_name": "kk.tsv",
                        "target_text": georgian_primary or georgian,
                        "headword": mingrelian,
                        "translation": georgian_primary or georgian,
                        "matched_on": "mingrelian",
                    }
                )
        elif source_lang == "georgian" and target_lang == "mingrelian":
            if _normalize_lookup_value(georgian) == input_normalized:
                candidates.append(
                    {
                        "source_name": "kk.tsv",
                        "target_text": mingrelian,
                        "headword": mingrelian,
                        "translation": georgian,
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
        target_key = _normalize_lookup_value(candidate.get("target_text", ""))
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


def grep_search_pairs(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search sentence_pairs.tsv for English translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    rows = _load_sentence_pairs_rows()
    if not rows:
        return "", False

    # First pass: look for standalone word matches
    standalone_output = LOOKUP_SEPARATOR
    substring_output = LOOKUP_SEPARATOR

    for mingrelian, english in rows:
        entry = f"Mingrelian: {mingrelian}\nEnglish: {english}\n{LOOKUP_SEPARATOR}"
        if _is_standalone_match(mingrelian, word) or _is_standalone_match(english, word):
            standalone_output += entry
        elif _is_substring_match(f"{mingrelian}\t{english}", word):
            substring_output += entry

    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(standalone_output), True
    elif (not standalone_only) and substring_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(substring_output), False
    return "", False


def grep_search_gal(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search gal.tsv for Russian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    rows = _load_gal_rows()
    if not rows:
        return "", False

    # First pass: look for standalone word matches
    standalone_output = LOOKUP_SEPARATOR
    substring_output = LOOKUP_SEPARATOR

    lowered_word = word.lower()
    for russian, mingrelian in rows:
        entry = f"Mingrelian: {mingrelian}\nRussian: {russian}\n{LOOKUP_SEPARATOR}"
        haystack = f"{russian}\t{mingrelian}"
        if _is_standalone_match(mingrelian, word) or _is_standalone_match(russian, word):
            standalone_output += entry
        elif _is_substring_match(haystack, word) or _is_substring_match(haystack, lowered_word):
            substring_output += entry

    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(standalone_output), True
    elif (not standalone_only) and substring_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(substring_output), False
    return "", False


def grep_search_kk(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search kk.tsv for Russian and Georgian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    rows = _load_kk_rows()
    if not rows:
        return "", False

    # First pass: look for standalone word matches
    standalone_output = LOOKUP_SEPARATOR
    substring_output = LOOKUP_SEPARATOR
    lowered_word = word.lower()

    for mingrelian, ipa, russian, georgian in rows:
        formatted_entry = _format_kk_entry(mingrelian, russian, georgian)
        haystack = "\t".join((mingrelian, ipa, russian, georgian))
        if (
            _is_standalone_match(mingrelian, word)
            or _is_standalone_match(russian, word)
            or _is_standalone_match(georgian, word)
        ):
            standalone_output += formatted_entry + "\n" + LOOKUP_SEPARATOR
        elif _is_substring_match(haystack, word) or _is_substring_match(haystack, lowered_word):
            substring_output += formatted_entry + "\n" + LOOKUP_SEPARATOR

    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(standalone_output), True
    elif (not standalone_only) and substring_output != LOOKUP_SEPARATOR:
        return _truncate_lookup_output(substring_output), False
    return "", False


def grep_search_context_source(word: str, *, standalone_only: bool = False) -> str:
    """
    Search context_source.txt for relevant entry blocks.
    Splits text by empty lines and returns the block containing the search term.
    Prioritizes standalone word matches over substring matches.
    """
    entries = _load_context_source_entries()
    if not entries:
        return ""

    # First pass: look for standalone word matches
    standalone_output = LOOKUP_SEPARATOR
    substring_output = LOOKUP_SEPARATOR

    for entry in entries:
        if _is_standalone_match(entry, word):
            # Standalone match found
            standalone_output += entry.strip()
            standalone_output += "\n" + LOOKUP_SEPARATOR
        elif _is_substring_match(entry, word):
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
    pairs_result, pairs_has_standalone = grep_search_pairs(word)
    gal_result, gal_has_standalone = grep_search_gal(res_ru)
    kk_ru_result, kk_ru_has_standalone = grep_search_kk(res_ru)
    kk_ge_result, kk_ge_has_standalone = grep_search_kk(res_ge)

    output += pairs_result
    output += gal_result
    output += kk_ru_result
    output += kk_ge_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = (pairs_has_standalone or gal_has_standalone or
                          kk_ru_has_standalone or kk_ge_has_standalone)

    if not has_any_standalone:
        output += grep_search_context_source(res_ge)

    return _truncate_lookup_output(output)


def grep_search_from_mingrelian(word: str) -> str:
    """
    Search all dictionaries from Mingrelian word.
    Short-circuits context-source search if standalone matches found in extractive dictionaries.
    """
    def _mkhedruli_has_letters(s: str) -> bool:
        return any('\u10D0' <= ch <= '\u10FF' for ch in s)

    def _ends_with_mkhedruli_vowel(s: str) -> bool:
        if not s:
            return False
        # Include Mingrelian schwa letter (ჷ) as vowel-like in our data.
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
    pairs_result, pairs_has_standalone = grep_search_pairs(word)
    gal_result, gal_has_standalone = grep_search_gal(word)
    kk_result, kk_has_standalone = grep_search_kk(word)

    output += pairs_result
    output += gal_result
    output += kk_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or gal_has_standalone or kk_has_standalone

    context_source_result = ""
    if not has_any_standalone:
        context_source_result = grep_search_context_source(word)
        output += context_source_result

    # If absolutely nothing matched across all four sources, try a conservative
    # case-suffix stripping fallback (e.g., ...თ → stem).
    case_fallback_applied = False
    if not (pairs_result or gal_result or kk_result or context_source_result):
        for stem in _case_strip_candidates_mkhedruli(word):
            # First: standalone-only search. If we find any standalone match for this
            # candidate, we return ONLY standalone matches and stop (no partial matches,
            # and no further candidates like the bare stem).
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True)
            context_source2_s = grep_search_context_source(stem, standalone_only=True)

            output2_s = pairs2_s + gal2_s + kk2_s + context_source2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(context_source2_s)

            if has_any_standalone2:
                output += f"\n[Case-stripped fallback: {word} → {stem}]\n"
                output += output2_s
                case_fallback_applied = True
                break

            # Otherwise, fall back to the normal grep-style matching for this candidate.
            pairs2, pairs2_has = grep_search_pairs(stem)
            gal2, gal2_has = grep_search_gal(stem)
            kk2, kk2_has = grep_search_kk(stem)

            output2 = pairs2 + gal2 + kk2
            has_any_standalone2 = pairs2_has or gal2_has or kk2_has
            context_source2 = "" if has_any_standalone2 else grep_search_context_source(stem)
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
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True)
            context_source2_s = grep_search_context_source(stem, standalone_only=True)

            output2_s = pairs2_s + gal2_s + kk2_s + context_source2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(context_source2_s)

            if has_any_standalone2:
                output += f"\n[Preverb-stripped fallback: {word} → {stem}]\n"
                output += output2_s
                break

            # Otherwise allow normal grep-style matches.
            pairs2, pairs2_has = grep_search_pairs(stem)
            gal2, gal2_has = grep_search_gal(stem)
            kk2, kk2_has = grep_search_kk(stem)

            output2 = pairs2 + gal2 + kk2
            has_any_standalone2 = pairs2_has or gal2_has or kk2_has
            context_source2 = "" if has_any_standalone2 else grep_search_context_source(stem)
            output2 += context_source2

            if output2:
                output += f"\n[Preverb-stripped fallback: {word} → {stem}]\n"
                output += output2
                break

    return _truncate_lookup_output(output)


def grep_search_from_georgian(word: str) -> str:
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
    pairs_result, pairs_has_standalone = grep_search_pairs(res_en)
    kk_result, kk_has_standalone = grep_search_kk(word)
    gal_result, gal_has_standalone = grep_search_gal(res_ru)

    output += pairs_result
    output += kk_result
    output += gal_result

    # Only search the context source if no standalone matches were found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or kk_has_standalone or gal_has_standalone

    if not has_any_standalone:
        output += grep_search_context_source(word)

    return _truncate_lookup_output(output)


def check_exact_match_simple(input_text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Check if the exact input text exists in extractive dictionaries (not the context source).
    Returns the translation if found, None otherwise.

    This is the simple direct lookup without Google Translate augmentation.
    """
    input_normalized = _normalize_lookup_value(input_text)

    # Check sentence_pairs.tsv (Mingrelian ↔ English)
    if (source_lang, target_lang) in [("mingrelian", "english"), ("english", "mingrelian")]:
        for mingrelian, english in _load_sentence_pairs_rows():
            if source_lang == "mingrelian" and _normalize_lookup_value(mingrelian) == input_normalized:
                return english
            elif source_lang == "english" and _normalize_lookup_value(english) == input_normalized:
                return mingrelian

    # Check kk.tsv (Mingrelian ↔ Russian ↔ Georgian)
    for mingrelian, ipa, russian, georgian in _load_kk_rows():
        # Mingrelian → Georgian
        if source_lang == "mingrelian" and target_lang == "georgian":
            if _normalize_lookup_value(mingrelian) == input_normalized:
                georgian_primary, _ = _split_figurative_gloss(georgian, "ka")
                return georgian_primary or georgian

        # Georgian → Mingrelian
        elif source_lang == "georgian" and target_lang == "mingrelian":
            if _normalize_lookup_value(georgian) == input_normalized:
                return mingrelian

        # Mingrelian → English
        elif source_lang == "mingrelian" and target_lang == "english":
            if _normalize_lookup_value(mingrelian) == input_normalized:
                # We don't have English in kk, skip
                pass

    # Check gal.tsv (Russian ↔ Mingrelian)
    for russian, mingrelian in _load_gal_rows():
        # Russian → Mingrelian
        if source_lang == "russian" and target_lang == "mingrelian":
            if _normalize_lookup_value(russian) == input_normalized:
                return mingrelian

        # Mingrelian → Russian
        elif source_lang == "mingrelian" and target_lang == "russian":
            if _normalize_lookup_value(mingrelian) == input_normalized:
                return russian

    return None


def find_mingrelian_in_dicts(text: str, target_lang: Optional[str] = None) -> Optional[tuple[str, str, str]]:
    """
    Find ANY translation for a text in dictionaries, searching across all columns.
    Returns (mingrelian, other_language_text, other_language_code) if found.

    Search order prioritizes clean extractive dictionaries (sentence_pairs, gal) over kk.

    Args:
        text: Text to search for (case-insensitive)

    Returns:
        tuple or None: (mingrelian_text, other_lang_text, lang_code) if found
    """
    text_normalized = _normalize_lookup_value(text)

    # Priority 1: Search sentence_pairs.tsv (English ↔ Mingrelian, cleanest)
    for mingrelian, english in _load_sentence_pairs_rows():
        if _normalize_lookup_value(mingrelian) == text_normalized:
            return (mingrelian, english, "en")
        elif _normalize_lookup_value(english) == text_normalized:
            return (mingrelian, english, "en")

    # Priority 2: Search gal.tsv (Russian ↔ Mingrelian, reliable)
    for russian, mingrelian in _load_gal_rows():
        if _normalize_lookup_value(mingrelian) == text_normalized:
            return (mingrelian, russian, "ru")
        elif _normalize_lookup_value(russian) == text_normalized:
            return (mingrelian, russian, "ru")

    # Priority 3: Search kk.tsv (may have data quality issues, use as fallback)
    for mingrelian, ipa, russian, georgian in _load_kk_rows():
        if _normalize_lookup_value(mingrelian) == text_normalized:
            bridge_text, lang_code = _choose_kk_bridge_gloss(russian, georgian, target_lang)
            if bridge_text and lang_code:
                return (mingrelian, bridge_text, lang_code)
            return (mingrelian, georgian, "ka")
        elif _normalize_lookup_value(georgian) == text_normalized:
            georgian_primary, _ = _split_figurative_gloss(georgian, "ka")
            return (mingrelian, georgian_primary or georgian, "ka")
        elif _normalize_lookup_value(russian) == text_normalized:
            russian_primary, _ = _split_figurative_gloss(russian, "ru")
            return (mingrelian, russian_primary or russian, "ru")

    return None


def check_exact_match_with_google_translate(input_text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Advanced exact match using Google Translate to bridge high-resource languages.

    SCENARIO 1: Translating TO Mingrelian (from English/Georgian)
    - Translate input to all high-resource languages (en, ka, ru)
    - Search dictionaries for each translated version
    - Return Mingrelian if found

    SCENARIO 2: Translating FROM Mingrelian (to English/Georgian)
    - Search dictionaries for Mingrelian word
    - If found with any high-resource language pair
    - Google Translate that language to target
    - Return translation
    """
    if GoogleTranslator is None:
        return None

    # SCENARIO 1: Translating TO Mingrelian from high-resource language
    if target_lang == "mingrelian" and source_lang in ["english", "georgian"]:
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
            exact_candidates = collect_exact_match_candidates(translated_text, lang, "mingrelian")
            if len(exact_candidates) == 1:
                match = exact_candidates[0]["target_text"]
                logger.info(
                    f"[GOOGLE BRIDGE TO MINGRELIAN] {input_text} ({source_lang}) → "
                    f"{translated_text} ({lang}) → {match} (mingrelian)"
                )
                return match

            match = check_exact_match_simple(translated_text, lang, "mingrelian")
            if match:
                logger.info(
                    f"[GOOGLE BRIDGE TO MINGRELIAN] {input_text} ({source_lang}) → "
                    f"{translated_text} ({lang}) → {match} (mingrelian)"
                )
                return match

    # SCENARIO 2: Translating FROM Mingrelian to high-resource language
    elif source_lang == "mingrelian" and target_lang in ["english", "georgian"]:
        # Try direct lookup first
        direct_match = check_exact_match_simple(input_text, source_lang, target_lang)
        if direct_match:
            return direct_match

        # Search for Mingrelian in ANY dictionary with ANY language pair
        result = find_mingrelian_in_dicts(input_text, target_lang=target_lang)
        if result:
            mingrelian_text, other_lang_text, lang_code = result

            # If the found language IS the target, return directly
            lang_map = {"en": "english", "ka": "georgian", "ru": "russian"}
            found_lang = lang_map.get(lang_code)

            if found_lang == target_lang:
                logger.info(
                    "[DIRECT DICT MATCH] %s (mingrelian) → %s (%s)",
                    mingrelian_text,
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
                    "[GOOGLE BRIDGE FROM MINGRELIAN] %s (mingrelian) → %s (%s) → %s (%s)",
                    mingrelian_text,
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
