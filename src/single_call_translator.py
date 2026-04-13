#!/usr/bin/env python3
"""
Single-call translation system using dictionary lookups and one LLM API call.
Adapted from explore_rag_dict.ipynb notebook.
"""
import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

from src.logger import (
    setup_logger, 
    log_prompt, 
    log_llm_response, 
    log_instant_lookup,
    log_stage_timing
)
import time

# Setup logger for translator module
logger = setup_logger('translator')


# Language labels for prompts
LANG_LABEL = {
    "mingrelian": "Mingrelian",
    "english": "English",
    "georgian": "Georgian",
}

FIGURATIVE_MARKERS = {
    "ru": ("переносное значение", "перен."),
    "ka": ("გადატანილი მნიშვნელობით", "გადატ."),
}


def _normalize_gloss_segment(text: str) -> str:
    """Strip numbering and separators that are not meaningful translation content."""
    normalized = re.sub(r"\s+", " ", text).strip(" ;,\n\t")
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
    normalized = re.sub(r"\s+\d+\.$", "", normalized)
    return normalized.strip(" ;,\n\t")


def _split_figurative_gloss(gloss: str, lang_code: str) -> tuple[str, Optional[str]]:
    """Split a dictionary gloss into primary and figurative senses."""
    markers = FIGURATIVE_MARKERS.get(lang_code, ())
    if not gloss or not markers:
        normalized = _normalize_gloss_segment(gloss)
        return normalized, None

    lowered = gloss.lower()
    marker_position = min(
        (idx for marker in markers if (idx := lowered.find(marker)) != -1),
        default=-1,
    )
    if marker_position == -1:
        normalized = _normalize_gloss_segment(gloss)
        return normalized, None

    matched_marker = next(marker for marker in markers if lowered.find(marker) == marker_position)
    primary = _normalize_gloss_segment(gloss[:marker_position])
    secondary = _normalize_gloss_segment(gloss[marker_position + len(matched_marker):])
    return primary, secondary or None


def _format_kk_entry(mingrelian: str, russian: str, georgian: str) -> str:
    """Format a kk.tsv entry with figurative senses clearly marked as secondary."""
    ru_primary, ru_figurative = _split_figurative_gloss(russian, "ru")
    ka_primary, ka_figurative = _split_figurative_gloss(georgian, "ka")

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


def _choose_kk_bridge_gloss(russian: str, georgian: str, target_lang: Optional[str]) -> tuple[str, str]:
    """
    Pick the safest gloss for direct bridging from kk.tsv.

    Russian glosses bridge to English more reliably than Georgian glosses in the
    current pipeline, so prefer Russian when producing English.
    """
    russian_primary, _ = _split_figurative_gloss(russian, "ru")
    georgian_primary, _ = _split_figurative_gloss(georgian, "ka")

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


def _get_data_path(filename: str) -> str:
    """Get the path to a data file, checking multiple possible locations."""
    # Try fastapi_app/data first (for API usage)
    fastapi_data = Path(__file__).parent.parent / 'fastapi_app' / 'data' / filename
    if fastapi_data.exists():
        return str(fastapi_data)
    
    # Try parent data directory
    parent_data = Path(__file__).parent.parent / 'data' / filename
    if parent_data.exists():
        return str(parent_data)
    
    # Try notebooks directory (for development)
    notebooks_data = Path(__file__).parent.parent / 'notebooks' / filename
    if notebooks_data.exists():
        return str(notebooks_data)
    
    # Try notebooks/dicts directory
    notebooks_dicts = Path(__file__).parent.parent / 'notebooks' / 'dicts' / filename
    if notebooks_dicts.exists():
        return str(notebooks_dicts)
    
    # Default to fastapi_app/data
    return str(fastapi_data)


def _is_standalone_match(text: str, word: str) -> bool:
    """
    Check if word appears as a standalone word in text (not as part of another word).
    Supports Mingrelian headword notations like `ნუმ(უ)` by allowing non-word
    characters (punctuation/whitespace) between letters while still enforcing
    standalone boundaries.
    
    Args:
        text: Text to search in
        word: Word to search for
        
    Returns:
        bool: True if word appears standalone, False otherwise
    """
    word = (word or "").strip()
    if not word:
        return False
    return _compiled_word_pattern(word, standalone=True).search(text) is not None


def _is_substring_match(text: str, word: str) -> bool:
    """
    Like a basic `word in text` check, but also surfaces dictionary headword
    variants where punctuation appears between letters (e.g., `नुम(უ)`).
    """
    word = (word or "").strip()
    if not word:
        return False
    # Fast path: direct substring
    if word in text:
        return True
    # Slow path: fuzzy "letters separated by punctuation" regex
    return _compiled_word_pattern(word, standalone=False).search(text) is not None


@lru_cache(maxsize=4096)
def _compiled_word_pattern(word: str, standalone: bool) -> re.Pattern:
    """
    Compile and cache the regex used for matching `word`.

    - For short tokens (<=2 chars), we use strict contiguous matching to avoid
      excessive false-positives and keep matching fast.
    - For longer tokens, we allow *punctuation (not whitespace)* between
      characters so queries like `ნუმუ` match `ნუმ(უ)` without accidentally
      matching across separate words like `... ანუ ... მუზმა ...`.
    """
    escaped = re.escape(word)
    if len(word) <= 2:
        core = escaped
    else:
        # Allow punctuation/symbols between letters, but do NOT allow whitespace.
        # This keeps matches within a token (e.g., parentheses in headwords).
        sep = r"[^\w\s]*"
        core = sep.join(re.escape(ch) for ch in word)

    if standalone:
        # Prefer lookarounds over \b for robust Unicode behavior.
        pattern = r"(?<!\w)" + core + r"(?!\w)"
    else:
        pattern = core

    return re.compile(pattern, re.IGNORECASE)


# English
def grep_search_pairs(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search sentence_pairs.tsv for English translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    file_path = _get_data_path("sentence_pairs.tsv")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "", False
    
    # First pass: look for standalone word matches
    standalone_output = "========\n"
    substring_output = "========\n"
    
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            mingrelian, english = parts[0], parts[1]
            if mingrelian and english:
                # Check if word appears as standalone in either mingrelian or english
                if _is_standalone_match(mingrelian, word) or _is_standalone_match(english, word):
                    standalone_output += "Mingrelian: " + mingrelian + "\n"
                    standalone_output += "English: " + english
                    standalone_output += "========\n"
                elif _is_substring_match(line, word):
                    # Word appears as substring
                    substring_output += "Mingrelian: " + mingrelian + "\n"
                    substring_output += "English: " + english
                    substring_output += "========\n"
    
    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != "========\n":
        return standalone_output, True
    elif (not standalone_only) and substring_output != "========\n":
        return substring_output, False
    return "", False


# Russian
def grep_search_gal(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search gal.tsv for Russian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    file_path = _get_data_path("gal.tsv")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "", False
    
    # First pass: look for standalone word matches
    standalone_output = "========\n"
    substring_output = "========\n"
    
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            russian, mingrelian = parts[0], parts[1]
            if mingrelian and russian:
                # Check if word appears as standalone
                if _is_standalone_match(mingrelian, word) or _is_standalone_match(russian, word):
                    standalone_output += "Mingrelian: " + mingrelian
                    standalone_output += "Russian: " + russian + "\n"
                    standalone_output += "========\n"
                elif _is_substring_match(line, word) or _is_substring_match(line, word.lower()):
                    # Word appears as substring
                    substring_output += "Mingrelian: " + mingrelian
                    substring_output += "Russian: " + russian + "\n"
                    substring_output += "========\n"
    
    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != "========\n":
        return standalone_output, True
    elif (not standalone_only) and substring_output != "========\n":
        return substring_output, False
    return "", False


# Russian and Georgian
def grep_search_kk(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search kk.tsv for Russian and Georgian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    file_path = _get_data_path("kk.tsv")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return "", False
    
    # First pass: look for standalone word matches
    standalone_output = "========\n"
    substring_output = "========\n"
    
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 4:
            mingrelian, ipa, russian, georgian = parts[0], parts[1], parts[2], parts[3]
            if mingrelian and russian and georgian:
                formatted_entry = _format_kk_entry(
                    mingrelian.strip(),
                    russian.strip(),
                    georgian.strip(),
                )
                # Check if word appears as standalone
                if (_is_standalone_match(mingrelian, word) or 
                    _is_standalone_match(russian, word) or 
                    _is_standalone_match(georgian, word)):
                    standalone_output += formatted_entry + "\n"
                    standalone_output += "========\n"
                elif _is_substring_match(line, word) or _is_substring_match(line, word.lower()):
                    # Word appears as substring
                    substring_output += formatted_entry + "\n"
                    substring_output += "========\n"
    
    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != "========\n":
        return standalone_output, True
    elif (not standalone_only) and substring_output != "========\n":
        return substring_output, False
    return "", False


# Unstructured fallback context source
def grep_search_context_source(word: str, *, standalone_only: bool = False) -> str:
    """
    Search context_source.txt for relevant entry blocks.
    Splits text by empty lines and returns the block containing the search term.
    Prioritizes standalone word matches over substring matches.
    """
    file_path = _get_data_path("context_source.txt")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            context_source_text = file.read()
    except FileNotFoundError:
        return ""
    
    entries = re.split(r'\n\s*\n', context_source_text.strip())
    
    # First pass: look for standalone word matches
    standalone_output = "========\n"
    substring_output = "========\n"
    
    for entry in entries:
        if _is_standalone_match(entry, word):
            # Standalone match found
            standalone_output += entry.strip()
            standalone_output += "\n========\n"
        elif _is_substring_match(entry, word):
            # Substring match
            substring_output += entry.strip()
            substring_output += "\n========\n"
    
    # Return standalone matches if found, otherwise substring matches (unless standalone_only)
    if standalone_output != "========\n":
        return standalone_output
    elif (not standalone_only) and substring_output != "========\n":
        return substring_output
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

    if len(output) > 10000:
        return output[:10000]

    return output


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

    if len(output) > 10000:
        return output[:10000]
    
    return output


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

    if len(output) > 10000:
        return output[:10000]
    return output


def _load_grammar(path: Optional[str] = None) -> str:
    """Load the Mingrelian grammar file."""
    if path is None:
        path = _get_data_path("harris.txt")
    
    try:
        with open(path, "r", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""


def _build_dict_entries(sentence: str, lookup_fn: Callable[[str], str]) -> str:
    """Build dictionary entries by looking up each word in the sentence."""
    dict_entries = ""
    for word in sentence.split():
        cleaned_word = word.strip(string.punctuation)
        if cleaned_word:
            dict_entries += lookup_fn(cleaned_word)
    return dict_entries


def _construct_translation_prompt(
    *,
    input_lang: str,
    output_lang: str,
    sentence: str,
    dict_entries: str,
    grammar: str,
) -> str:
    """Construct the complete translation prompt for a single LLM call."""
    in_label = LANG_LABEL.get(input_lang, input_lang)
    out_label = LANG_LABEL.get(output_lang, output_lang)

    # Build the base prompt
    prompt = f'''Your task is to translate a phrase or a sentence from {in_label} to {out_label}.

To accomplish this, I will provide you with a set of dictionary entries from Mingrelian dictionaries of different kinds.

The dictionary may have definitions in Russian, Georgian, or English.'''
    
    # Only add grammar section if we have grammar content
    if grammar:
        prompt += f''' I will also provide you with Mingrelian grammar information, describing the morphological and syntactual patterns of Mingrelian.'''
    
    prompt += f'''

Please use these resources to aid you in your translation.

You will translate the following phrase/sentence: "{sentence}". Return any notes you want, then end with:
<<<TRANSLATION>>>
FINAL_TRANSLATION_HERE
<<<END_TRANSLATION>>>

Here are some various dictionary entries for word(s) in that phrase:

{dict_entries}
'''
    
    # Only add grammar if we have it
    if grammar:
        prompt += f'''
Here is the Mingrelian grammar information:

{grammar}

That is the end of the grammar information.
'''
    
    prompt += f'''
Now remember, we are translating the following sentence: "{sentence}" from {in_label} to {out_label}.

Return any notes you want, then end with:
<<<TRANSLATION>>>
FINAL_TRANSLATION_HERE
<<<END_TRANSLATION>>>
'''
    
    return prompt


def _construct_prompt(
    sentence: str, 
    *, 
    input_lang: str, 
    output_lang: str, 
    lookup_fn: Callable[[str], str]
) -> str:
    """Construct a prompt for translation."""
    dict_entries = _build_dict_entries(sentence, lookup_fn)
    
    # Only load the massive grammar file if we actually have dictionary entries
    # Otherwise use simplified prompt (saves ~96K tokens and 40+ seconds!)
    if dict_entries and dict_entries.strip():
        grammar = _load_grammar()
    else:
        grammar = ""
    
    return _construct_translation_prompt(
        input_lang=input_lang,
        output_lang=output_lang,
        sentence=sentence,
        dict_entries=dict_entries,
        grammar=grammar,
    )


def construct_prompt_from_mingrelian_to_english(mingrelian_sentence: str) -> str:
    """Construct prompt for Mingrelian → English translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="english",
        lookup_fn=grep_search_from_mingrelian,
    )


def construct_prompt_from_english_to_mingrelian(english_sentence: str) -> str:
    """Construct prompt for English → Mingrelian translation."""
    return _construct_prompt(
        english_sentence,
        input_lang="english",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_english,
    )


def construct_prompt_from_georgian_to_mingrelian(georgian_sentence: str) -> str:
    """Construct prompt for Georgian → Mingrelian translation."""
    return _construct_prompt(
        georgian_sentence,
        input_lang="georgian",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_georgian,
    )


def construct_prompt_from_mingrelian_to_georgian(mingrelian_sentence: str) -> str:
    """Construct prompt for Mingrelian → Georgian translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="georgian",
        lookup_fn=grep_search_from_mingrelian,
    )


# Prompt builder routing
PROMPT_BUILDERS = {
    ("mingrelian", "english"): construct_prompt_from_mingrelian_to_english,
    ("english", "mingrelian"): construct_prompt_from_english_to_mingrelian,
    ("mingrelian", "georgian"): construct_prompt_from_mingrelian_to_georgian,
    ("georgian", "mingrelian"): construct_prompt_from_georgian_to_mingrelian,
}


def check_exact_match_simple(input_text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Check if the exact input text exists in extractive dictionaries (not the context source).
    Returns the translation if found, None otherwise.
    
    This is the simple direct lookup without Google Translate augmentation.
    """
    input_lower = input_text.lower().strip()
    
    # Check sentence_pairs.tsv (Mingrelian ↔ English)
    if (source_lang, target_lang) in [("mingrelian", "english"), ("english", "mingrelian")]:
        try:
            file_path = _get_data_path("sentence_pairs.tsv")
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        mingrelian, english = parts[0].strip(), parts[1].strip()
                        if source_lang == "mingrelian" and mingrelian.lower() == input_lower:
                            return english
                        elif source_lang == "english" and english.lower() == input_lower:
                            return mingrelian
        except FileNotFoundError:
            pass
    
    # Check kk.tsv (Mingrelian ↔ Russian ↔ Georgian)
    try:
        file_path = _get_data_path("kk.tsv")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    mingrelian, ipa, russian, georgian = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    
                    # Mingrelian → Georgian
                    if source_lang == "mingrelian" and target_lang == "georgian":
                        if mingrelian.lower() == input_lower:
                            georgian_primary, _ = _split_figurative_gloss(georgian, "ka")
                            return georgian_primary or georgian
                    
                    # Georgian → Mingrelian
                    elif source_lang == "georgian" and target_lang == "mingrelian":
                        if georgian.lower() == input_lower:
                            return mingrelian
                    
                    # Mingrelian → English
                    elif source_lang == "mingrelian" and target_lang == "english":
                        if mingrelian.lower() == input_lower:
                            # We don't have English in kk, skip
                            pass
    except FileNotFoundError:
        pass
    
    # Check gal.tsv (Russian ↔ Mingrelian)
    try:
        file_path = _get_data_path("gal.tsv")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    russian, mingrelian = parts[0].strip(), parts[1].strip()
                    
                    # Russian → Mingrelian
                    if source_lang == "russian" and target_lang == "mingrelian":
                        if russian.lower() == input_lower:
                            return mingrelian
                    
                    # Mingrelian → Russian
                    elif source_lang == "mingrelian" and target_lang == "russian":
                        if mingrelian.lower() == input_lower:
                            return russian
    except FileNotFoundError:
        pass
    
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
    text_lower = text.lower().strip()
    
    # Priority 1: Search sentence_pairs.tsv (English ↔ Mingrelian, cleanest)
    try:
        file_path = _get_data_path("sentence_pairs.tsv")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mingrelian, english = parts[0].strip(), parts[1].strip()
                    if mingrelian.lower() == text_lower:
                        return (mingrelian, english, "en")
                    elif english.lower() == text_lower:
                        return (mingrelian, english, "en")
    except FileNotFoundError:
        pass
    
    # Priority 2: Search gal.tsv (Russian ↔ Mingrelian, reliable)
    try:
        file_path = _get_data_path("gal.tsv")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    russian, mingrelian = parts[0].strip(), parts[1].strip()
                    if mingrelian.lower() == text_lower:
                        return (mingrelian, russian, "ru")
                    elif russian.lower() == text_lower:
                        return (mingrelian, russian, "ru")
    except FileNotFoundError:
        pass
    
    # Priority 3: Search kk.tsv (may have data quality issues, use as fallback)
    try:
        file_path = _get_data_path("kk.tsv")
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    mingrelian, ipa, russian, georgian = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    if mingrelian.lower() == text_lower:
                        bridge_text, lang_code = _choose_kk_bridge_gloss(russian, georgian, target_lang)
                        if bridge_text and lang_code:
                            return (mingrelian, bridge_text, lang_code)
                        return (mingrelian, georgian, "ka")
                    elif georgian.lower() == text_lower:
                        georgian_primary, _ = _split_figurative_gloss(georgian, "ka")
                        return (mingrelian, georgian_primary or georgian, "ka")
                    elif russian.lower() == text_lower:
                        russian_primary, _ = _split_figurative_gloss(russian, "ru")
                        return (mingrelian, russian_primary or russian, "ru")
    except FileNotFoundError:
        pass
    
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
        
        # Search for each translated version in dictionaries
        for translated_text, lang in translations_to_try:
            match = check_exact_match_simple(translated_text, lang, "mingrelian")
            if match:
                print(f"[GOOGLE BRIDGE TO MINGRELIAN] {input_text} ({source_lang}) → {translated_text} ({lang}) → {match} (mingrelian)")
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
                print(f"[DIRECT DICT MATCH] {mingrelian_text} (mingrelian) → {other_lang_text} ({target_lang})")
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
                print(f"[GOOGLE BRIDGE FROM MINGRELIAN] {mingrelian_text} (mingrelian) → {other_lang_text} ({lang_code}) → {translated} ({target_code})")
                return translated
            
            except Exception as e:
                print(f"[GOOGLE BRIDGE ERROR] Failed to translate: {e}")
                pass
    
    return None


def extract_translation(response_text: str) -> str:
    """
    Extract the final translation from LLM response using <<<TRANSLATION>>> markers.
    
    Args:
        response_text: The model's response text
        
    Returns:
        str: The extracted translation, or the full response if markers not found
    """
    # Look for content between <<<TRANSLATION>>> and <<<END_TRANSLATION>>>
    match = re.search(r'<<<TRANSLATION>>>\s*(.*?)\s*<<<END_TRANSLATION>>>', 
                     response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: return the full response if markers not found
    return response_text.strip()


def translate(
    input_text: str,
    source_lang: str,
    target_lang: str,
    llm_client
) -> dict:
    """
    Translate text using single LLM call approach.
    
    Args:
        input_text: Text to translate
        source_lang: Source language (mingrelian, georgian, or english)
        target_lang: Target language (mingrelian, georgian, or english)
        llm_client: LLM client instance
        
    Returns:
        dict: Translation results with keys: translation, full_response
    """
    overall_start = time.time()
    
    # OPTIMIZATION 1: Check for exact match with Google Translate bridge
    stage_start = time.time()
    exact_match = check_exact_match_with_google_translate(input_text, source_lang, target_lang)
    log_stage_timing(logger, "Google Translate Bridge Check", time.time() - stage_start)
    
    if exact_match is not None:
        log_stage_timing(logger, "TOTAL (instant lookup)", time.time() - overall_start, "✅ No LLM call")
        logger.info(f"Instant lookup: '{input_text}' ({source_lang}) → '{exact_match}' ({target_lang})")
        log_instant_lookup(logger, input_text, exact_match, "dictionary+google_translate")
        return {
            'translation': exact_match,
            'full_response': f"Dictionary match (via Google Translate bridge):\n{exact_match}"
        }
    
    logger.info(f"No instant lookup found, proceeding to LLM for '{input_text}' ({source_lang} → {target_lang})")
    
    # OPTIMIZATION 2: Handle Georgian ↔ English with Google Translate (no Mingrelian involved)
    if GoogleTranslator is not None:
        if source_lang == "english" and target_lang == "georgian":
            stage_start = time.time()
            translation = GoogleTranslator(source="en", target="ka").translate(input_text)
            log_stage_timing(logger, "Google Translate Direct", time.time() - stage_start)
            log_stage_timing(logger, "TOTAL (Google Translate)", time.time() - overall_start, "✅ No LLM call")
            log_instant_lookup(logger, input_text, translation, "google_translate_en_ka")
            return {
                'translation': translation,
                'full_response': f"Translation (via Google Translate):\n{translation}"
            }
        
        if source_lang == "georgian" and target_lang == "english":
            stage_start = time.time()
            translation = GoogleTranslator(source="ka", target="en").translate(input_text)
            log_stage_timing(logger, "Google Translate Direct", time.time() - stage_start)
            log_stage_timing(logger, "TOTAL (Google Translate)", time.time() - overall_start, "✅ No LLM call")
            log_instant_lookup(logger, input_text, translation, "google_translate_ka_en")
            return {
                'translation': translation,
                'full_response': f"Translation (via Google Translate):\n{translation}"
            }
    
    # Get the appropriate prompt builder
    builder = PROMPT_BUILDERS.get((source_lang, target_lang))
    if builder is None:
        raise ValueError(f"Unsupported translation direction: {source_lang} → {target_lang}")
    
    # Build the prompt (includes dictionary searches)
    stage_start = time.time()
    prompt = builder(input_text)
    log_stage_timing(logger, "Prompt Construction (with dictionary searches)", time.time() - stage_start)
    log_prompt(logger, prompt, source_lang, target_lang)
    
    # Call the LLM - THIS IS THE KEY TIMING
    stage_start = time.time()
    response = llm_client.complete(prompt)
    llm_time = time.time() - stage_start
    log_stage_timing(logger, "🔥 LLM API CALL", llm_time, f"provider={llm_client.provider}, model={llm_client.model}")
    log_llm_response(logger, response, source_lang, target_lang)
    
    # Extract the translation
    stage_start = time.time()
    translation = extract_translation(response)
    log_stage_timing(logger, "Response Extraction", time.time() - stage_start)
    
    total_time = time.time() - overall_start
    log_stage_timing(logger, "TOTAL (with LLM)", total_time, f"LLM={llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
    
    logger.info(f"Extracted translation: '{translation}'")
    
    return {
        'translation': translation,
        'full_response': response
    }
