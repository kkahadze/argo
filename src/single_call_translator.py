#!/usr/bin/env python3
"""Backward-compatible facade for the split translator modules.

The translation implementation now lives under :mod:`src.translator`. This
module keeps legacy imports such as ``from src.single_call_translator import
translate`` working for the API, eval scripts, and external callers.
"""

from src.translator import data as _data
from src.translator import extraction as _extraction
from src.translator import lookup as _lookup
from src.translator import pipeline as _pipeline
from src.translator import prompts as _prompts
from src.translator import text_utils as _text_utils

_DATA_FILE_CACHE_KEY = _data._data_file_cache_key
_GET_DATA_PATH = _data._get_data_path
_IS_HEADER_ROW = _data._is_header_row
_LOAD_CONTEXT_SOURCE_ENTRIES = _data._load_context_source_entries
_LOAD_CONTEXT_SOURCE_ENTRIES_CACHED = _data._load_context_source_entries_cached
_LOAD_GAL_ROWS = _data._load_gal_rows
_LOAD_GAL_ROWS_CACHED = _data._load_gal_rows_cached
_LOAD_COMPACT_GRAMMAR = _data._load_compact_grammar
_LOAD_GRAMMAR = _data._load_grammar
_LOAD_GRAMMAR_CACHED = _data._load_grammar_cached
_LOAD_KK_ROWS = _data._load_kk_rows
_LOAD_KK_ROWS_CACHED = _data._load_kk_rows_cached
_LOAD_MASTER_LEXICON_ROWS = _data._load_master_lexicon_rows
_LOAD_MASTER_LEXICON_ROWS_CACHED = _data._load_master_lexicon_rows_cached
_LOAD_SENTENCE_PAIRS_ROWS = _data._load_sentence_pairs_rows
_LOAD_SENTENCE_PAIRS_ROWS_CACHED = _data._load_sentence_pairs_rows_cached
_MASTER_LEXICON_ENABLED = _data._master_lexicon_enabled

_BUILD_HIGH_RESOURCE_TO_MINGRELIAN_DICT_ENTRIES = _lookup._build_high_resource_to_mingrelian_dict_entries
_COLLECT_MASTER_LEXICON_EXACT_CANDIDATES = _lookup._collect_master_lexicon_exact_candidates
_COLLECT_SIMPLE_EXACT_MATCH_CANDIDATES = _lookup._collect_simple_exact_match_candidates
_COLLECT_TOKEN_EXACT_CANDIDATES = _lookup._collect_token_exact_candidates
_CHECK_EXACT_MATCH_SIMPLE = _lookup.check_exact_match_simple
_CHECK_EXACT_MATCH_WITH_GOOGLE_TRANSLATE = _lookup.check_exact_match_with_google_translate
_COLLECT_EXACT_MATCH_CANDIDATES = _lookup.collect_exact_match_candidates
_FIND_MINGRELIAN_IN_DICTS = _lookup.find_mingrelian_in_dicts
_GREP_SEARCH_CONTEXT_SOURCE = _lookup.grep_search_context_source
_GREP_SEARCH_FROM_ENGLISH = _lookup.grep_search_from_english
_GREP_SEARCH_FROM_GEORGIAN = _lookup.grep_search_from_georgian
_GREP_SEARCH_FROM_MINGRELIAN = _lookup.grep_search_from_mingrelian
_GREP_SEARCH_GAL = _lookup.grep_search_gal
_GREP_SEARCH_KK = _lookup.grep_search_kk
_GREP_SEARCH_PAIRS = _lookup.grep_search_pairs

_BUILD_DICT_ENTRIES = _prompts._build_dict_entries
_CONSTRUCT_PROMPT = _prompts._construct_prompt
_CONSTRUCT_TRANSLATION_PROMPT = _prompts._construct_translation_prompt
_LOAD_GRAMMAR_FOR_POLICY = _prompts._load_grammar_for_policy
_MEASURE_PROMPT_SECTIONS = _prompts._measure_prompt_sections
_NORMALIZE_GRAMMAR_POLICY = _prompts._normalize_grammar_policy
_CONSTRUCT_PROMPT_FROM_ENGLISH_TO_MINGRELIAN = _prompts.construct_prompt_from_english_to_mingrelian
_CONSTRUCT_PROMPT_FROM_GEORGIAN_TO_MINGRELIAN = _prompts.construct_prompt_from_georgian_to_mingrelian
_CONSTRUCT_PROMPT_FROM_MINGRELIAN_TO_ENGLISH = _prompts.construct_prompt_from_mingrelian_to_english
_CONSTRUCT_PROMPT_FROM_MINGRELIAN_TO_GEORGIAN = _prompts.construct_prompt_from_mingrelian_to_georgian
_FORMAT_EXACT_CANDIDATE_BLOCK = _prompts._format_exact_candidate_block

_EXTRACT_TRANSLATION = _extraction.extract_translation
_TRANSLATE = _pipeline.translate

FIGURATIVE_MARKERS = _text_utils.FIGURATIVE_MARKERS
GoogleTranslator = _lookup.GoogleTranslator
LANG_LABEL = _text_utils.LANG_LABEL
LOOKUP_SEPARATOR = _text_utils.LOOKUP_SEPARATOR
LOW_VALUE_LOOKUP_TERMS = _text_utils.LOW_VALUE_LOOKUP_TERMS
MAX_LOOKUP_OUTPUT_CHARS = _text_utils.MAX_LOOKUP_OUTPUT_CHARS
SUPPORTED_GRAMMAR_POLICIES = _prompts.SUPPORTED_GRAMMAR_POLICIES
logger = _pipeline.logger

_choose_kk_bridge_gloss = _text_utils._choose_kk_bridge_gloss
_compiled_word_pattern = _text_utils._compiled_word_pattern
_format_kk_entry = _text_utils._format_kk_entry
_format_token_candidate_block = _text_utils._format_token_candidate_block
_is_low_value_lookup_term = _text_utils._is_low_value_lookup_term
_is_standalone_match = _text_utils._is_standalone_match
_is_substring_match = _text_utils._is_substring_match
_lookup_variants_for_token = _text_utils._lookup_variants_for_token
_normalize_gloss_segment = _text_utils._normalize_gloss_segment
_normalize_lookup_value = _text_utils._normalize_lookup_value
_split_figurative_gloss = _text_utils._split_figurative_gloss
_truncate_lookup_output = _text_utils._truncate_lookup_output

extract_translation = _EXTRACT_TRANSLATION


def _sync_compat_state() -> None:
    """Propagate legacy monkeypatches on this facade to implementation modules."""
    _text_utils.FIGURATIVE_MARKERS = FIGURATIVE_MARKERS
    _text_utils.LANG_LABEL = LANG_LABEL
    _text_utils.LOOKUP_SEPARATOR = LOOKUP_SEPARATOR
    _text_utils.LOW_VALUE_LOOKUP_TERMS = LOW_VALUE_LOOKUP_TERMS
    _text_utils.MAX_LOOKUP_OUTPUT_CHARS = MAX_LOOKUP_OUTPUT_CHARS

    _data._data_file_cache_key = _data_file_cache_key
    _data._get_data_path = _get_data_path
    _data._is_header_row = _is_header_row
    _data._load_context_source_entries = _load_context_source_entries
    _data._load_context_source_entries_cached = _load_context_source_entries_cached
    _data._load_gal_rows = _load_gal_rows
    _data._load_gal_rows_cached = _load_gal_rows_cached
    _data._load_compact_grammar = _load_compact_grammar
    _data._load_grammar = _load_grammar
    _data._load_grammar_cached = _load_grammar_cached
    _data._load_kk_rows = _load_kk_rows
    _data._load_kk_rows_cached = _load_kk_rows_cached
    _data._load_master_lexicon_rows = _load_master_lexicon_rows
    _data._load_master_lexicon_rows_cached = _load_master_lexicon_rows_cached
    _data._load_sentence_pairs_rows = _load_sentence_pairs_rows
    _data._load_sentence_pairs_rows_cached = _load_sentence_pairs_rows_cached
    _data._master_lexicon_enabled = _master_lexicon_enabled
    _data._normalize_lookup_value = _normalize_lookup_value

    _lookup.GoogleTranslator = GoogleTranslator
    _lookup.LANG_LABEL = LANG_LABEL
    _lookup.LOOKUP_SEPARATOR = LOOKUP_SEPARATOR
    _lookup.LOW_VALUE_LOOKUP_TERMS = LOW_VALUE_LOOKUP_TERMS
    _lookup._choose_kk_bridge_gloss = _choose_kk_bridge_gloss
    _lookup._collect_master_lexicon_exact_candidates = _collect_master_lexicon_exact_candidates
    _lookup._collect_simple_exact_match_candidates = _collect_simple_exact_match_candidates
    _lookup._collect_token_exact_candidates = _collect_token_exact_candidates
    _lookup._format_kk_entry = _format_kk_entry
    _lookup._format_token_candidate_block = _format_token_candidate_block
    _lookup._is_low_value_lookup_term = _is_low_value_lookup_term
    _lookup._is_standalone_match = _is_standalone_match
    _lookup._is_substring_match = _is_substring_match
    _lookup._load_context_source_entries = _load_context_source_entries
    _lookup._load_gal_rows = _load_gal_rows
    _lookup._load_kk_rows = _load_kk_rows
    _lookup._load_master_lexicon_rows = _load_master_lexicon_rows
    _lookup._load_sentence_pairs_rows = _load_sentence_pairs_rows
    _lookup._lookup_variants_for_token = _lookup_variants_for_token
    _lookup._normalize_lookup_value = _normalize_lookup_value
    _lookup._split_figurative_gloss = _split_figurative_gloss
    _lookup._truncate_lookup_output = _truncate_lookup_output
    _lookup.check_exact_match_simple = check_exact_match_simple
    _lookup.check_exact_match_with_google_translate = check_exact_match_with_google_translate
    _lookup.collect_exact_match_candidates = collect_exact_match_candidates
    _lookup.find_mingrelian_in_dicts = find_mingrelian_in_dicts
    _lookup.grep_search_context_source = grep_search_context_source
    _lookup.grep_search_from_english = grep_search_from_english
    _lookup.grep_search_from_georgian = grep_search_from_georgian
    _lookup.grep_search_from_mingrelian = grep_search_from_mingrelian
    _lookup.grep_search_gal = grep_search_gal
    _lookup.grep_search_kk = grep_search_kk
    _lookup.grep_search_pairs = grep_search_pairs
    _lookup.logger = logger

    _prompts.LANG_LABEL = LANG_LABEL
    _prompts.PROMPT_BUILDERS = PROMPT_BUILDERS
    _prompts._build_dict_entries = _build_dict_entries
    _prompts._build_high_resource_to_mingrelian_dict_entries = _build_high_resource_to_mingrelian_dict_entries
    _prompts._construct_prompt = _construct_prompt
    _prompts._construct_translation_prompt = _construct_translation_prompt
    _prompts._format_exact_candidate_block = _format_exact_candidate_block
    _prompts._load_compact_grammar = _load_compact_grammar
    _prompts._load_grammar = _load_grammar
    _prompts._load_grammar_for_policy = _load_grammar_for_policy
    _prompts._measure_prompt_sections = _measure_prompt_sections
    _prompts._normalize_grammar_policy = _normalize_grammar_policy
    _prompts.SUPPORTED_GRAMMAR_POLICIES = SUPPORTED_GRAMMAR_POLICIES
    _prompts.construct_prompt_from_english_to_mingrelian = construct_prompt_from_english_to_mingrelian
    _prompts.construct_prompt_from_georgian_to_mingrelian = construct_prompt_from_georgian_to_mingrelian
    _prompts.construct_prompt_from_mingrelian_to_english = construct_prompt_from_mingrelian_to_english
    _prompts.construct_prompt_from_mingrelian_to_georgian = construct_prompt_from_mingrelian_to_georgian
    _prompts.grep_search_from_english = grep_search_from_english
    _prompts.grep_search_from_georgian = grep_search_from_georgian
    _prompts.grep_search_from_mingrelian = grep_search_from_mingrelian

    _pipeline.GoogleTranslator = GoogleTranslator
    _pipeline.PROMPT_BUILDERS = PROMPT_BUILDERS
    _pipeline._format_exact_candidate_block = _format_exact_candidate_block
    _pipeline._master_lexicon_enabled = _master_lexicon_enabled
    _pipeline._measure_prompt_sections = _measure_prompt_sections
    _pipeline._normalize_grammar_policy = _normalize_grammar_policy
    _pipeline.check_exact_match_with_google_translate = check_exact_match_with_google_translate
    _pipeline.collect_exact_match_candidates = collect_exact_match_candidates
    _pipeline.extract_translation = extract_translation
    _pipeline.logger = logger


def _data_file_cache_key(*args, **kwargs):
    _sync_compat_state()
    return _DATA_FILE_CACHE_KEY(*args, **kwargs)


def _get_data_path(*args, **kwargs):
    return _GET_DATA_PATH(*args, **kwargs)


def _is_header_row(*args, **kwargs):
    _sync_compat_state()
    return _IS_HEADER_ROW(*args, **kwargs)


def _master_lexicon_enabled(*args, **kwargs):
    return _MASTER_LEXICON_ENABLED(*args, **kwargs)


def _load_master_lexicon_rows_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_MASTER_LEXICON_ROWS_CACHED(*args, **kwargs)


def _load_master_lexicon_rows(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_MASTER_LEXICON_ROWS(*args, **kwargs)


def _load_sentence_pairs_rows_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_SENTENCE_PAIRS_ROWS_CACHED(*args, **kwargs)


def _load_sentence_pairs_rows(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_SENTENCE_PAIRS_ROWS(*args, **kwargs)


def _load_gal_rows_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_GAL_ROWS_CACHED(*args, **kwargs)


def _load_gal_rows(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_GAL_ROWS(*args, **kwargs)


def _load_kk_rows_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_KK_ROWS_CACHED(*args, **kwargs)


def _load_kk_rows(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_KK_ROWS(*args, **kwargs)


def _load_context_source_entries_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_CONTEXT_SOURCE_ENTRIES_CACHED(*args, **kwargs)


def _load_context_source_entries(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_CONTEXT_SOURCE_ENTRIES(*args, **kwargs)


def _load_grammar_cached(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_GRAMMAR_CACHED(*args, **kwargs)


def _load_grammar(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_GRAMMAR(*args, **kwargs)


def _load_compact_grammar(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_COMPACT_GRAMMAR(*args, **kwargs)


_load_master_lexicon_rows_cached.cache_clear = _LOAD_MASTER_LEXICON_ROWS_CACHED.cache_clear
_load_master_lexicon_rows_cached.cache_info = _LOAD_MASTER_LEXICON_ROWS_CACHED.cache_info
_load_sentence_pairs_rows_cached.cache_clear = _LOAD_SENTENCE_PAIRS_ROWS_CACHED.cache_clear
_load_sentence_pairs_rows_cached.cache_info = _LOAD_SENTENCE_PAIRS_ROWS_CACHED.cache_info
_load_gal_rows_cached.cache_clear = _LOAD_GAL_ROWS_CACHED.cache_clear
_load_gal_rows_cached.cache_info = _LOAD_GAL_ROWS_CACHED.cache_info
_load_kk_rows_cached.cache_clear = _LOAD_KK_ROWS_CACHED.cache_clear
_load_kk_rows_cached.cache_info = _LOAD_KK_ROWS_CACHED.cache_info
_load_context_source_entries_cached.cache_clear = _LOAD_CONTEXT_SOURCE_ENTRIES_CACHED.cache_clear
_load_context_source_entries_cached.cache_info = _LOAD_CONTEXT_SOURCE_ENTRIES_CACHED.cache_info
_load_grammar_cached.cache_clear = _LOAD_GRAMMAR_CACHED.cache_clear
_load_grammar_cached.cache_info = _LOAD_GRAMMAR_CACHED.cache_info


def _build_high_resource_to_mingrelian_dict_entries(*args, **kwargs):
    _sync_compat_state()
    return _BUILD_HIGH_RESOURCE_TO_MINGRELIAN_DICT_ENTRIES(*args, **kwargs)


def _collect_master_lexicon_exact_candidates(*args, **kwargs):
    _sync_compat_state()
    return _COLLECT_MASTER_LEXICON_EXACT_CANDIDATES(*args, **kwargs)


def _collect_simple_exact_match_candidates(*args, **kwargs):
    _sync_compat_state()
    return _COLLECT_SIMPLE_EXACT_MATCH_CANDIDATES(*args, **kwargs)


def _collect_token_exact_candidates(*args, **kwargs):
    _sync_compat_state()
    return _COLLECT_TOKEN_EXACT_CANDIDATES(*args, **kwargs)


def collect_exact_match_candidates(*args, **kwargs):
    _sync_compat_state()
    return _COLLECT_EXACT_MATCH_CANDIDATES(*args, **kwargs)


def grep_search_pairs(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_PAIRS(*args, **kwargs)


def grep_search_gal(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_GAL(*args, **kwargs)


def grep_search_kk(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_KK(*args, **kwargs)


def grep_search_context_source(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_CONTEXT_SOURCE(*args, **kwargs)


def grep_search_from_english(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_FROM_ENGLISH(*args, **kwargs)


def grep_search_from_georgian(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_FROM_GEORGIAN(*args, **kwargs)


def grep_search_from_mingrelian(*args, **kwargs):
    _sync_compat_state()
    return _GREP_SEARCH_FROM_MINGRELIAN(*args, **kwargs)


def check_exact_match_simple(*args, **kwargs):
    _sync_compat_state()
    return _CHECK_EXACT_MATCH_SIMPLE(*args, **kwargs)


def find_mingrelian_in_dicts(*args, **kwargs):
    _sync_compat_state()
    return _FIND_MINGRELIAN_IN_DICTS(*args, **kwargs)


def check_exact_match_with_google_translate(*args, **kwargs):
    _sync_compat_state()
    return _CHECK_EXACT_MATCH_WITH_GOOGLE_TRANSLATE(*args, **kwargs)


def _build_dict_entries(*args, **kwargs):
    _sync_compat_state()
    return _BUILD_DICT_ENTRIES(*args, **kwargs)


def _construct_prompt(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_PROMPT(*args, **kwargs)


def _construct_translation_prompt(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_TRANSLATION_PROMPT(*args, **kwargs)


def _normalize_grammar_policy(*args, **kwargs):
    _sync_compat_state()
    return _NORMALIZE_GRAMMAR_POLICY(*args, **kwargs)


def _load_grammar_for_policy(*args, **kwargs):
    _sync_compat_state()
    return _LOAD_GRAMMAR_FOR_POLICY(*args, **kwargs)


def _measure_prompt_sections(*args, **kwargs):
    _sync_compat_state()
    return _MEASURE_PROMPT_SECTIONS(*args, **kwargs)


def _format_exact_candidate_block(*args, **kwargs):
    _sync_compat_state()
    return _FORMAT_EXACT_CANDIDATE_BLOCK(*args, **kwargs)


def construct_prompt_from_mingrelian_to_english(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_PROMPT_FROM_MINGRELIAN_TO_ENGLISH(*args, **kwargs)


def construct_prompt_from_english_to_mingrelian(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_PROMPT_FROM_ENGLISH_TO_MINGRELIAN(*args, **kwargs)


def construct_prompt_from_georgian_to_mingrelian(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_PROMPT_FROM_GEORGIAN_TO_MINGRELIAN(*args, **kwargs)


def construct_prompt_from_mingrelian_to_georgian(*args, **kwargs):
    _sync_compat_state()
    return _CONSTRUCT_PROMPT_FROM_MINGRELIAN_TO_GEORGIAN(*args, **kwargs)


PROMPT_BUILDERS = {
    ("mingrelian", "english"): construct_prompt_from_mingrelian_to_english,
    ("english", "mingrelian"): construct_prompt_from_english_to_mingrelian,
    ("mingrelian", "georgian"): construct_prompt_from_mingrelian_to_georgian,
    ("georgian", "mingrelian"): construct_prompt_from_georgian_to_mingrelian,
}


def translate(*args, **kwargs):
    _sync_compat_state()
    return _TRANSLATE(*args, **kwargs)


__all__ = [
    "FIGURATIVE_MARKERS",
    "GoogleTranslator",
    "LANG_LABEL",
    "LOOKUP_SEPARATOR",
    "LOW_VALUE_LOOKUP_TERMS",
    "MAX_LOOKUP_OUTPUT_CHARS",
    "PROMPT_BUILDERS",
    "SUPPORTED_GRAMMAR_POLICIES",
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
