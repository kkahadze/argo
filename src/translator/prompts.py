#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import string
from typing import Callable

from src.translator.data import _load_grammar
from src.translator.lookup import (
    _build_high_resource_to_mingrelian_dict_entries,
    grep_search_from_english,
    grep_search_from_georgian,
    grep_search_from_mingrelian,
)
from src.translator.text_utils import LANG_LABEL

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


PROMPT_BUILDERS = {
    ("mingrelian", "english"): construct_prompt_from_mingrelian_to_english,
    ("english", "mingrelian"): construct_prompt_from_english_to_mingrelian,
    ("mingrelian", "georgian"): construct_prompt_from_mingrelian_to_georgian,
    ("georgian", "mingrelian"): construct_prompt_from_georgian_to_mingrelian,
}
