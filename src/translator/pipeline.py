#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import time
from typing import Optional

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
    log_stage_timing,
)
from src.translator.data import _master_lexicon_enabled
from src.translator.extraction import extract_translation
from src.translator.lookup import (
    check_exact_match_with_google_translate,
    collect_exact_match_candidates,
)
from src.translator.prompts import (
    PROMPT_BUILDERS,
    _format_exact_candidate_block,
    _measure_prompt_sections,
    _normalize_grammar_policy,
)

logger = setup_logger('translator')

def translate(
    input_text: str,
    source_lang: str,
    target_lang: str,
    llm_client,
    grammar_policy: Optional[str] = None,
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
    resolved_grammar_policy = _normalize_grammar_policy(grammar_policy)

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
                'grammar_policy': resolved_grammar_policy,
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
                    'grammar_policy': resolved_grammar_policy,
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
                    'grammar_policy': resolved_grammar_policy,
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
        grammar_policy=resolved_grammar_policy,
    )
    prompt_section_metrics = _measure_prompt_sections(prompt)
    prompt_metrics = {
        'reason': 'llm',
        'used_llm': True,
        'grammar_policy': resolved_grammar_policy,
        'prompt_characters': prompt_section_metrics['prompt_characters'],
        'prompt_chars': prompt_section_metrics['prompt_chars'],
        'dict_entries_chars': prompt_section_metrics['dict_entries_chars'],
        'grammar_chars': prompt_section_metrics['grammar_chars'],
        'grammar_included': prompt_section_metrics['grammar_included'],
        'used_grammar': prompt_section_metrics['grammar_included'],
        'has_dictionary_entries': prompt_section_metrics['dict_entries_chars'] > 0,
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
