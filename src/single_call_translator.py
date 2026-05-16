#!/usr/bin/env python3
"""Backward-compatible facade for the split translator modules.

The translation implementation now lives under :mod:`src.translator`. This
module keeps legacy imports such as ``from src.single_call_translator import
translate`` working for the API, eval scripts, and external callers.
"""

from src.translator import lookup as _lookup
from src.translator import pipeline as _pipeline
from src.translator.data import (
    _data_file_cache_key,
    _get_data_path,
    _load_context_source_entries,
    _load_context_source_entries_cached,
    _load_gal_rows,
    _load_gal_rows_cached,
    _load_grammar,
    _load_grammar_cached,
    _load_kk_rows,
    _load_kk_rows_cached,
    _load_master_lexicon_rows,
    _load_master_lexicon_rows_cached,
    _load_sentence_pairs_rows,
    _load_sentence_pairs_rows_cached,
    _master_lexicon_enabled,
)
from src.translator.extraction import extract_translation
from src.translator.lookup import (
    _build_high_resource_to_mingrelian_dict_entries as _lookup_build_high_resource_to_mingrelian_dict_entries,
    _collect_master_lexicon_exact_candidates,
    _collect_simple_exact_match_candidates,
    _collect_token_exact_candidates,
    check_exact_match_simple,
    collect_exact_match_candidates,
    find_mingrelian_in_dicts,
    grep_search_context_source,
    grep_search_gal,
    grep_search_kk,
    grep_search_pairs,
)
from src.translator.prompts import (
    PROMPT_BUILDERS,
    _build_dict_entries,
    _construct_prompt,
    _construct_translation_prompt,
    _format_exact_candidate_block,
    construct_prompt_from_english_to_mingrelian,
    construct_prompt_from_georgian_to_mingrelian,
    construct_prompt_from_mingrelian_to_english,
    construct_prompt_from_mingrelian_to_georgian,
)
from src.translator.text_utils import (
    FIGURATIVE_MARKERS,
    LANG_LABEL,
    LOOKUP_SEPARATOR,
    LOW_VALUE_LOOKUP_TERMS,
    MAX_LOOKUP_OUTPUT_CHARS,
    _choose_kk_bridge_gloss,
    _compiled_word_pattern,
    _format_kk_entry,
    _format_token_candidate_block,
    _is_low_value_lookup_term,
    _is_standalone_match,
    _is_substring_match,
    _lookup_variants_for_token,
    _normalize_gloss_segment,
    _normalize_lookup_value,
    _split_figurative_gloss,
    _truncate_lookup_output,
)

GoogleTranslator = _lookup.GoogleTranslator
logger = _pipeline.logger


def _sync_compat_state() -> None:
    """Propagate legacy monkeypatches on this facade to implementation modules."""
    _lookup.GoogleTranslator = GoogleTranslator
    _pipeline.GoogleTranslator = GoogleTranslator


def _build_high_resource_to_mingrelian_dict_entries(*args, **kwargs):
    _sync_compat_state()
    return _lookup_build_high_resource_to_mingrelian_dict_entries(*args, **kwargs)


def grep_search_from_english(*args, **kwargs):
    _sync_compat_state()
    return _lookup.grep_search_from_english(*args, **kwargs)


def grep_search_from_georgian(*args, **kwargs):
    _sync_compat_state()
    return _lookup.grep_search_from_georgian(*args, **kwargs)


def grep_search_from_mingrelian(*args, **kwargs):
    _sync_compat_state()
    return _lookup.grep_search_from_mingrelian(*args, **kwargs)


def check_exact_match_with_google_translate(*args, **kwargs):
    _sync_compat_state()
    return _lookup.check_exact_match_with_google_translate(*args, **kwargs)


def translate(*args, **kwargs):
    _sync_compat_state()
    return _pipeline.translate(*args, **kwargs)


__all__ = [
    "FIGURATIVE_MARKERS",
    "GoogleTranslator",
    "LANG_LABEL",
    "LOOKUP_SEPARATOR",
    "LOW_VALUE_LOOKUP_TERMS",
    "MAX_LOOKUP_OUTPUT_CHARS",
    "PROMPT_BUILDERS",
    "check_exact_match_simple",
    "check_exact_match_with_google_translate",
    "collect_exact_match_candidates",
    "construct_prompt_from_english_to_mingrelian",
    "construct_prompt_from_georgian_to_mingrelian",
    "construct_prompt_from_mingrelian_to_english",
    "construct_prompt_from_mingrelian_to_georgian",
    "extract_translation",
    "find_mingrelian_in_dicts",
    "grep_search_context_source",
    "grep_search_from_english",
    "grep_search_from_georgian",
    "grep_search_from_mingrelian",
    "grep_search_gal",
    "grep_search_kk",
    "grep_search_pairs",
    "logger",
    "translate",
]
