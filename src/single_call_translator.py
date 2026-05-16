#!/usr/bin/env python3
"""
Single-call translation system using dictionary lookups and one LLM API call.
Adapted from explore_rag_dict.ipynb notebook.
"""
import csv
import os
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
    log_translation_result,
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


def _master_lexicon_enabled() -> bool:
    """Allow master lexicon ablations without editing the core pipeline."""
    value = os.getenv("ARGO_ENABLE_MASTER_LEXICON", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


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

    # Try eval datasets directory
    eval_datasets = Path(__file__).parent.parent / 'eval' / 'datasets' / filename
    if eval_datasets.exists():
        return str(eval_datasets)
    
    # Default to fastapi_app/data
    return str(fastapi_data)


def _normalize_lookup_value(text: str) -> str:
    """Normalize user input and dictionary text for exact-match comparisons."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _data_file_cache_key(filename: str) -> tuple[str, Optional[int]]:
    """Build a cache key that invalidates when a data file changes on disk."""
    file_path = _get_data_path(filename)
    try:
        mtime_ns = Path(file_path).stat().st_mtime_ns
    except FileNotFoundError:
        return file_path, None
    return file_path, mtime_ns


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
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
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
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
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


def _truncate_lookup_output(output: str) -> str:
    """Keep lookup payloads bounded before they enter the LLM prompt."""
    if len(output) > MAX_LOOKUP_OUTPUT_CHARS:
        return output[:MAX_LOOKUP_OUTPUT_CHARS]
    return output


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


def _format_exact_candidate_block(
    *,
    input_text: str,
    source_lang: str,
    target_lang: str,
    candidates: list[dict[str, str]],
) -> str:
    """Format exact-match candidates for the LLM as a compact high-priority shortlist."""
    in_label = LANG_LABEL.get(source_lang, source_lang)
    out_label = LANG_LABEL.get(target_lang, target_lang)

    lines = [
        f'Exact full-input candidate matches for "{input_text}" ({in_label} → {out_label}):',
        "These candidates come from exact lexicon matches for the full input, so treat them as high-priority evidence.",
        "If several candidates are plausible, choose the best fit for the context.",
        "If the context is weak or absent, prefer the most canonical/default dictionary form over marked or over-specific variants.",
        "",
    ]

    for index, candidate in enumerate(candidates[:12], start=1):
        lines.append(f"Candidate {index}:")
        if target_lang == "mingrelian":
            lines.append(f"- Mingrelian: {candidate['target_text']}")
            if candidate.get("headword_raw"):
                lines.append(f"- Mingrelian (Latinized): {candidate['headword_raw']}")
            if candidate.get("translation"):
                lines.append(f"- English gloss: {candidate['translation']}")
        elif target_lang == "english":
            lines.append(f"- English: {candidate['target_text']}")
            if candidate.get("headword"):
                lines.append(f"- Mingrelian: {candidate['headword']}")
            if candidate.get("headword_raw"):
                lines.append(f"- Mingrelian (Latinized): {candidate['headword_raw']}")
        elif target_lang == "georgian":
            lines.append(f"- Georgian: {candidate['target_text']}")
            if candidate.get("headword"):
                lines.append(f"- Mingrelian: {candidate['headword']}")
        else:
            lines.append(f"- Candidate translation: {candidate['target_text']}")

        if candidate.get("source_name"):
            lines.append(f"- Source: {candidate['source_name']}")
        if candidate.get("matched_on"):
            lines.append(f"- Matched on: {candidate['matched_on']}")
        lines.append("")

    return "\n".join(lines).strip()


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


# Russian
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


# Russian and Georgian
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


# Unstructured fallback context source
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


def _build_dict_entries(
    sentence: str,
    *,
    input_lang: str,
    output_lang: str,
    lookup_fn: Callable[[str], str],
) -> str:
    """Build dictionary entries by looking up each word in the sentence."""
    if output_lang == "mingrelian" and input_lang in {"english", "georgian"}:
        return _build_high_resource_to_mingrelian_dict_entries(
            sentence,
            input_lang=input_lang,
            lookup_fn=lookup_fn,
        )

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
    exact_candidates_block: str,
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
    
    if exact_candidates_block:
        prompt += f'''
Please use these resources to aid you in your translation.

You will translate the following phrase/sentence: "{sentence}". Return any notes you want, then end with:
<<<TRANSLATION>>>
FINAL_TRANSLATION_HERE
<<<END_TRANSLATION>>>

Here are exact candidate translations for the full input. Treat these as high-priority evidence:

{exact_candidates_block}
'''
    else:
        prompt += f'''

Please use these resources to aid you in your translation.

You will translate the following phrase/sentence: "{sentence}". Return any notes you want, then end with:
<<<TRANSLATION>>>
FINAL_TRANSLATION_HERE
<<<END_TRANSLATION>>>
'''

    if dict_entries and dict_entries.strip():
        prompt += f'''

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
    lookup_fn: Callable[[str], str],
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
) -> str:
    """Construct a prompt for translation."""
    dict_entries = "" if skip_word_lookups else _build_dict_entries(
        sentence,
        input_lang=input_lang,
        output_lang=output_lang,
        lookup_fn=lookup_fn,
    )
    
    # Only load the massive grammar file if we actually have dictionary entries
    # Otherwise use simplified prompt (saves ~96K tokens and 40+ seconds!)
    if (dict_entries and dict_entries.strip()) or (exact_candidates_block and exact_candidates_block.strip()):
        grammar = _load_grammar()
    else:
        grammar = ""
    
    return _construct_translation_prompt(
        input_lang=input_lang,
        output_lang=output_lang,
        sentence=sentence,
        exact_candidates_block=exact_candidates_block,
        dict_entries=dict_entries,
        grammar=grammar,
    )


def construct_prompt_from_mingrelian_to_english(
    mingrelian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
) -> str:
    """Construct prompt for Mingrelian → English translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="english",
        lookup_fn=grep_search_from_mingrelian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
    )


def construct_prompt_from_english_to_mingrelian(
    english_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
) -> str:
    """Construct prompt for English → Mingrelian translation."""
    return _construct_prompt(
        english_sentence,
        input_lang="english",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_english,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
    )


def construct_prompt_from_georgian_to_mingrelian(
    georgian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
) -> str:
    """Construct prompt for Georgian → Mingrelian translation."""
    return _construct_prompt(
        georgian_sentence,
        input_lang="georgian",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_georgian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
    )


def construct_prompt_from_mingrelian_to_georgian(
    mingrelian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
) -> str:
    """Construct prompt for Mingrelian → Georgian translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="georgian",
        lookup_fn=grep_search_from_mingrelian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
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


def extract_translation(response_text: str) -> str:
    """
    Extract the final translation from LLM response using <<<TRANSLATION>>> markers.
    
    Args:
        response_text: The model's response text
        
    Returns:
        str: The extracted translation, or the full response if markers not found
    """
    def _clean_extracted_text(text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line in {"<<<TRANSLATION>>>", "<<<END_TRANSLATION>>>"}:
                continue
            if re.fullmatch(r"FINAL_TRANSLATION_HERE[:\-\s]*", line, re.IGNORECASE):
                continue
            if re.fullmatch(r"(Final\s+)?Translation\s*:\s*", line, re.IGNORECASE):
                continue
            line = re.sub(r"^(Final\s+)?Translation\s*:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line.strip("`\"' "))

        cleaned = "\n".join(line for line in lines if line).strip()
        return cleaned

    # Primary path: content between explicit translation markers.
    match = re.search(
        r'<<<TRANSLATION>>>\s*(.*?)\s*<<<END_TRANSLATION>>>',
        response_text,
        re.DOTALL,
    )
    if match:
        cleaned = _clean_extracted_text(match.group(1))
        if cleaned:
            return cleaned

    # Secondary path: content after a translation marker if the model omitted the closing marker.
    trailing_marker_match = re.search(r'<<<TRANSLATION>>>\s*(.*)$', response_text, re.DOTALL)
    if trailing_marker_match:
        cleaned = _clean_extracted_text(trailing_marker_match.group(1))
        if cleaned:
            return cleaned

    # Tertiary path: recover from models that ignore the markers but provide a final label.
    label_matches = re.findall(
        r'(?im)^(?:final\s+translation|translation)\s*:\s*(.+)$',
        response_text,
    )
    if label_matches:
        cleaned = _clean_extracted_text(label_matches[-1])
        if cleaned:
            return cleaned

    # Final fallback: use the last non-empty, non-marker line rather than the whole response blob.
    fallback_lines = [
        line.strip("`\"' ")
        for line in response_text.splitlines()
        if line.strip()
        and "<<<TRANSLATION>>>" not in line
        and "<<<END_TRANSLATION>>>" not in line
        and "FINAL_TRANSLATION_HERE" not in line
    ]
    if fallback_lines:
        return fallback_lines[-1]

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
    exact_candidates_block = ""
    skip_word_lookups = False
    instant_lookup_method = ""
    master_lexicon_enabled = _master_lexicon_enabled()
    
    # OPTIMIZATION 1: Resolve exact full-input candidates before broader bridge logic.
    stage_start = time.time()
    exact_candidates = collect_exact_match_candidates(input_text, source_lang, target_lang)
    exact_match = None

    if len(exact_candidates) == 1:
        exact_match = exact_candidates[0]["target_text"]
        instant_lookup_method = "exact_lexicon"
    elif len(exact_candidates) > 1:
        exact_candidates_block = _format_exact_candidate_block(
            input_text=input_text,
            source_lang=source_lang,
            target_lang=target_lang,
            candidates=exact_candidates,
        )
        skip_word_lookups = True

    if exact_match is None and not exact_candidates_block:
        exact_match = check_exact_match_with_google_translate(input_text, source_lang, target_lang)
        if exact_match is not None:
            instant_lookup_method = "dictionary+google_translate"

    log_stage_timing(logger, "Exact Match Resolution", time.time() - stage_start)
    
    if exact_match is not None:
        log_stage_timing(logger, "TOTAL (instant lookup)", time.time() - overall_start, "✅ No LLM call")
        logger.info(f"Instant lookup: '{input_text}' ({source_lang}) → '{exact_match}' ({target_lang})")
        log_instant_lookup(logger, input_text, exact_match, instant_lookup_method or "exact_lexicon")
        full_response_label = (
            "Exact lexicon match"
            if instant_lookup_method == "exact_lexicon"
            else "Dictionary match (via Google Translate bridge)"
        )
        return {
            'translation': exact_match,
            'full_response': f"{full_response_label}:\n{exact_match}",
            'response_source': (
                "exact_lexicon"
                if instant_lookup_method == "exact_lexicon"
                else "dictionary_google_bridge"
            ),
            'prompt_metrics': {
                'reason': 'instant_lookup',
                'method': instant_lookup_method or 'exact_lexicon',
                'used_llm': False,
                'exact_candidate_count': len(exact_candidates),
                'master_lexicon_enabled': master_lexicon_enabled,
            },
        }
    
    if exact_candidates_block:
        logger.info(
            "Ambiguous exact candidates found for '%s' (%s → %s); proceeding to LLM with shortlist",
            input_text,
            source_lang,
            target_lang,
        )
    else:
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
                'full_response': f"Translation (via Google Translate):\n{translation}",
                'response_source': 'google_translate_direct',
                'prompt_metrics': {
                    'reason': 'google_translate_direct',
                    'method': 'google_translate_en_ka',
                    'used_llm': False,
                    'master_lexicon_enabled': master_lexicon_enabled,
                },
            }
        
        if source_lang == "georgian" and target_lang == "english":
            stage_start = time.time()
            translation = GoogleTranslator(source="ka", target="en").translate(input_text)
            log_stage_timing(logger, "Google Translate Direct", time.time() - stage_start)
            log_stage_timing(logger, "TOTAL (Google Translate)", time.time() - overall_start, "✅ No LLM call")
            log_instant_lookup(logger, input_text, translation, "google_translate_ka_en")
            return {
                'translation': translation,
                'full_response': f"Translation (via Google Translate):\n{translation}",
                'response_source': 'google_translate_direct',
                'prompt_metrics': {
                    'reason': 'google_translate_direct',
                    'method': 'google_translate_ka_en',
                    'used_llm': False,
                    'master_lexicon_enabled': master_lexicon_enabled,
                },
            }
    
    # Get the appropriate prompt builder
    builder = PROMPT_BUILDERS.get((source_lang, target_lang))
    if builder is None:
        raise ValueError(f"Unsupported translation direction: {source_lang} → {target_lang}")
    
    # Build the prompt (includes dictionary searches)
    stage_start = time.time()
    prompt = builder(
        input_text,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
    )
    prompt_metrics = {
        'reason': 'llm',
        'used_llm': True,
        'prompt_characters': len(prompt),
        'used_grammar': 'Here is the Mingrelian grammar information:' in prompt,
        'has_dictionary_entries': 'Here are some various dictionary entries for word(s) in that phrase:' in prompt,
        'exact_candidate_count': len(exact_candidates),
        'used_exact_candidate_shortlist': bool(exact_candidates_block),
        'skip_word_lookups': skip_word_lookups,
        'master_lexicon_enabled': master_lexicon_enabled,
    }
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
    
    prompt_metrics['llm_call_ms'] = int(llm_time * 1000)
    prompt_metrics['response_characters'] = len(response)
    prompt_metrics['translation_characters'] = len(translation)
    log_translation_result(logger, translation, source_lang, target_lang)
    
    return {
        'translation': translation,
        'full_response': response,
        'response_source': 'llm',
        'prompt_metrics': prompt_metrics,
    }
