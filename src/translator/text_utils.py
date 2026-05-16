#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import re
import string
from functools import lru_cache
from typing import Optional

LANG_LABEL = {
    "mingrelian": "Mingrelian",
    "english": "English",
    "georgian": "Georgian",
}

FIGURATIVE_MARKERS = {
    "ru": ("переносное значение", "перен."),
    "ka": ("გადატანილი მნიშვნელობით", "გადატ."),
}

LOW_VALUE_LOOKUP_TERMS = {
    "english": {
        "a", "an", "the", "am", "are", "is", "was", "were", "be", "been", "being",
        "i", "me", "my", "you", "your", "he", "she", "it", "we", "they", "our",
        "their", "this", "that", "these", "those", "in", "on", "at", "to", "of",
        "for", "from", "and", "or",
    },
    "georgian": {
        "მე", "ჩემი", "ჩემს", "ჩემმა", "ვარ", "არის", "არიან", "იყო", "იყოს",
        "ეს", "ესეც", "ის", "მას", "მასაც", "შენ", "თქვენ", "ჩვენ",
    },
}

LOOKUP_SEPARATOR = "========\n"

MAX_LOOKUP_OUTPUT_CHARS = 10000


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


def _normalize_lookup_value(text: str) -> str:
    """Normalize user input and dictionary text for exact-match comparisons."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _truncate_lookup_output(output: str) -> str:
    """Keep lookup payloads bounded before they enter the LLM prompt."""
    if len(output) > MAX_LOOKUP_OUTPUT_CHARS:
        return output[:MAX_LOOKUP_OUTPUT_CHARS]
    return output


def _lookup_variants_for_token(word: str, source_lang: str) -> list[str]:
    """Generate conservative lookup variants for token-level candidate searches."""
    cleaned = word.strip(string.punctuation + "“”„’'\"")
    if not cleaned:
        return []

    variants = [cleaned]

    if source_lang == "english":
        lowered = cleaned.casefold()
        variants.append(lowered)
        if lowered.endswith("'s") and len(lowered) > 2:
            variants.append(lowered[:-2])
        if lowered.endswith("ies") and len(lowered) > 4:
            variants.append(lowered[:-3] + "y")
        elif lowered.endswith("s") and len(lowered) > 3 and not lowered.endswith("ss"):
            variants.append(lowered[:-1])

    elif source_lang == "georgian":
        if cleaned.endswith("ები") and len(cleaned) > 4:
            stem = cleaned[:-3]
            variants.extend([stem, stem + "ი"])
        if cleaned.endswith("ში") and len(cleaned) > 3:
            stem = cleaned[:-2]
            variants.extend([stem, stem + "ი"])

    deduped: list[str] = []
    seen = set()
    for variant in variants:
        normalized = variant.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _format_token_candidate_block(
    *,
    token: str,
    source_lang: str,
    candidates: list[dict[str, str]],
) -> str:
    """Format a small candidate list for a single source token."""
    source_label = LANG_LABEL.get(source_lang, source_lang)
    lines = [
        f'Candidate translations for token "{token}" ({source_label} → Mingrelian):',
        "Treat these as lexicon candidates for this token, not as a full-sentence translation.",
        "",
    ]

    for index, candidate in enumerate(candidates[:6], start=1):
        lines.append(f"Candidate {index}:")
        lines.append(f"- Mingrelian: {candidate['target_text']}")
        if candidate.get("headword_raw"):
            lines.append(f"- Mingrelian (Latinized): {candidate['headword_raw']}")
        if candidate.get("translation"):
            lines.append(f"- Gloss: {candidate['translation']}")
        if candidate.get("matched_on"):
            lines.append(f"- Matched on: {candidate['matched_on']}")
        if candidate.get("source_name"):
            lines.append(f"- Source: {candidate['source_name']}")
        lines.append("")

    return "\n".join(lines).strip()


def _is_low_value_lookup_term(word: str, source_lang: str, token_count: int) -> bool:
    """Skip high-frequency function words in multi-word lookup contexts."""
    if token_count <= 1:
        return False

    normalized = _normalize_lookup_value(word)
    return normalized in LOW_VALUE_LOOKUP_TERMS.get(source_lang, set())


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
