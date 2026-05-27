#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import os
import string
from typing import Callable, Optional

from src.translator.data import _load_compact_grammar, _load_grammar
from src.translator.lookup import (
    _build_high_resource_to_mingrelian_dict_entries,
    _build_high_resource_to_svan_dict_entries,
    _build_high_resource_to_tsova_tush_dict_entries,
    _build_svan_to_georgian_dict_entries,
    grep_search_from_english,
    grep_search_from_english_for_pack,
    grep_search_from_georgian,
    grep_search_from_georgian_for_pack,
    grep_search_from_mingrelian,
)
from src.translator.text_utils import LANG_LABEL
from src.language_packs import get_low_resource_pack_for_pair

SUPPORTED_GRAMMAR_POLICIES = {"full", "compact", "none"}


def _normalize_grammar_policy(grammar_policy: Optional[str] = None) -> str:
    """Resolve the grammar inclusion policy for prompt construction."""
    value = (grammar_policy or os.getenv("ARGO_GRAMMAR_POLICY", "full")).strip().lower()
    if value not in SUPPORTED_GRAMMAR_POLICIES:
        return "full"
    return value


def _load_grammar_for_policy(
    grammar_policy: Optional[str] = None,
    pack_id: str = "mingrelian",
) -> str:
    """Load grammar text according to the selected policy."""
    policy = _normalize_grammar_policy(grammar_policy)
    if policy == "none":
        return ""
    if policy == "compact":
        return _load_compact_grammar(pack_id=pack_id)
    return _load_grammar(pack_id=pack_id)


def _measure_prompt_sections(
    prompt: str,
    grammar_marker: str | None = None,
) -> dict[str, object]:
    """Measure prompt sections so evals can compare prompt strategies."""
    dict_marker = "Here are some various dictionary entries for word(s) in that phrase:"
    grammar_marker = grammar_marker or "Here is the Mingrelian grammar information:"
    grammar_end_marker = "That is the end of the grammar information."
    final_instruction_marker = "Now remember, we are translating the following sentence:"

    dict_entries_chars = 0
    if dict_marker in prompt:
        dict_section = prompt.split(dict_marker, 1)[1]
        end_offsets = [
            dict_section.find(marker)
            for marker in (grammar_marker, final_instruction_marker)
            if marker in dict_section
        ]
        if end_offsets:
            dict_section = dict_section[: min(end_offsets)]
        dict_entries_chars = len(dict_section.strip())

    grammar_chars = 0
    if grammar_marker in prompt:
        grammar_section = prompt.split(grammar_marker, 1)[1]
        if grammar_end_marker in grammar_section:
            grammar_section = grammar_section.split(grammar_end_marker, 1)[0]
        grammar_chars = len(grammar_section.strip())

    return {
        "prompt_chars": len(prompt),
        "prompt_characters": len(prompt),
        "dict_entries_chars": dict_entries_chars,
        "grammar_chars": grammar_chars,
        "grammar_included": grammar_chars > 0,
    }

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
    pack = get_low_resource_pack_for_pair(source_lang, target_lang)
    low_resource_label = pack.display_name if pack else "Mingrelian"

    lines = [
        f'Exact full-input candidate matches for "{input_text}" ({in_label} → {out_label}):',
        "These candidates come from exact lexicon matches for the full input, so treat them as high-priority evidence.",
        "If several candidates are plausible, choose the best fit for the context.",
        "If the context is weak or absent, prefer the most canonical/default dictionary form over marked or over-specific variants.",
        "",
    ]

    for index, candidate in enumerate(candidates[:12], start=1):
        lines.append(f"Candidate {index}:")
        if target_lang in {"mingrelian", "tsova_tush", "svan"}:
            lines.append(f"- {low_resource_label}: {candidate['target_text']}")
            if candidate.get("headword_raw"):
                lines.append(f"- {low_resource_label} (Latinized): {candidate['headword_raw']}")
            if candidate.get("translation"):
                lines.append(f"- English gloss: {candidate['translation']}")
        elif target_lang == "english":
            lines.append(f"- English: {candidate['target_text']}")
            if candidate.get("headword"):
                lines.append(f"- {low_resource_label}: {candidate['headword']}")
            if candidate.get("headword_raw"):
                lines.append(f"- {low_resource_label} (Latinized): {candidate['headword_raw']}")
        elif target_lang == "georgian":
            lines.append(f"- Georgian: {candidate['target_text']}")
            if candidate.get("headword"):
                lines.append(f"- {low_resource_label}: {candidate['headword']}")
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
    if input_lang == "svan" and output_lang == "georgian":
        return _build_svan_to_georgian_dict_entries(sentence, lookup_fn=lookup_fn)

    if output_lang in {"mingrelian", "tsova_tush", "svan"} and input_lang in {"english", "georgian"}:
        builder = {
            "mingrelian": _build_high_resource_to_mingrelian_dict_entries,
            "tsova_tush": _build_high_resource_to_tsova_tush_dict_entries,
            "svan": _build_high_resource_to_svan_dict_entries,
        }[output_lang]
        return builder(
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
    pack = get_low_resource_pack_for_pair(input_lang, output_lang)
    dictionary_heading = pack.dictionary_heading if pack else "Mingrelian dictionaries"
    grammar_subject = pack.grammar_subject if pack else "Mingrelian"
    grammar_heading = pack.grammar_heading if pack else "Here is the Mingrelian grammar information:"

    # Build the base prompt
    prompt = f'''Your task is to translate a phrase or a sentence from {in_label} to {out_label}.

To accomplish this, I will provide you with a set of dictionary entries from {dictionary_heading} of different kinds.

The dictionary may have definitions in Russian, Georgian, or English.'''

    # Only add grammar section if we have grammar content
    if grammar:
        prompt += f''' I will also provide you with {grammar_subject} grammar information, describing the morphological and syntactual patterns of {grammar_subject}.'''

    if exact_candidates_block:
        prompt += f'''
Please use these resources to aid you in your translation.

You will translate the following phrase/sentence: "{sentence}".
Return only the final translation in this exact block. Do not include notes, explanations, glosses, markdown, bullets, or text outside the block:
<<<TRANSLATION>>>
FINAL_TRANSLATION_HERE
<<<END_TRANSLATION>>>

Here are exact candidate translations for the full input. Treat these as high-priority evidence:

{exact_candidates_block}
'''
    else:
        prompt += f'''

Please use these resources to aid you in your translation.

You will translate the following phrase/sentence: "{sentence}".
Return only the final translation in this exact block. Do not include notes, explanations, glosses, markdown, bullets, or text outside the block:
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
{grammar_heading}

{grammar}

That is the end of the grammar information.
'''

    prompt += f'''
Now remember, we are translating the following sentence: "{sentence}" from {in_label} to {out_label}.

Return only the final translation in this exact block. Do not include notes, explanations, glosses, markdown, bullets, or text outside the block:
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
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct a prompt for translation."""
    resolved_grammar_policy = _normalize_grammar_policy(grammar_policy)
    pack = get_low_resource_pack_for_pair(input_lang, output_lang)
    if (
        grammar_policy is None
        and pack
        and pack.code == "svan"
        and input_lang == "svan"
        and output_lang == "georgian"
    ):
        resolved_grammar_policy = "none"
    dict_entries = "" if skip_word_lookups else _build_dict_entries(
        sentence,
        input_lang=input_lang,
        output_lang=output_lang,
        lookup_fn=lookup_fn,
    )

    # Only load grammar when retrieval context exists. The grammar policy selects
    # full Harris, compact translator notes, or no grammar context.
    if (dict_entries and dict_entries.strip()) or (exact_candidates_block and exact_candidates_block.strip()):
        grammar = _load_grammar_for_policy(
            resolved_grammar_policy,
            pack_id=pack.code if pack else "mingrelian",
        )
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
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Mingrelian → English translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="english",
        lookup_fn=grep_search_from_mingrelian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_english_to_mingrelian(
    english_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for English → Mingrelian translation."""
    return _construct_prompt(
        english_sentence,
        input_lang="english",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_english,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_georgian_to_mingrelian(
    georgian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Georgian → Mingrelian translation."""
    return _construct_prompt(
        georgian_sentence,
        input_lang="georgian",
        output_lang="mingrelian",
        lookup_fn=grep_search_from_georgian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_mingrelian_to_georgian(
    mingrelian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Mingrelian → Georgian translation."""
    return _construct_prompt(
        mingrelian_sentence,
        input_lang="mingrelian",
        output_lang="georgian",
        lookup_fn=grep_search_from_mingrelian,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_tsova_tush_to_english(
    tsova_tush_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Bats → English translation."""
    from src.translator.lookup import grep_search_from_tsova_tush

    return _construct_prompt(
        tsova_tush_sentence,
        input_lang="tsova_tush",
        output_lang="english",
        lookup_fn=grep_search_from_tsova_tush,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_english_to_tsova_tush(
    english_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for English → Bats translation."""
    return _construct_prompt(
        english_sentence,
        input_lang="english",
        output_lang="tsova_tush",
        lookup_fn=lambda word: grep_search_from_english_for_pack(word, "tsova_tush"),
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_georgian_to_tsova_tush(
    georgian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Georgian → Bats translation."""
    return _construct_prompt(
        georgian_sentence,
        input_lang="georgian",
        output_lang="tsova_tush",
        lookup_fn=lambda word: grep_search_from_georgian_for_pack(word, "tsova_tush"),
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_tsova_tush_to_georgian(
    tsova_tush_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Bats → Georgian translation."""
    from src.translator.lookup import grep_search_from_tsova_tush

    return _construct_prompt(
        tsova_tush_sentence,
        input_lang="tsova_tush",
        output_lang="georgian",
        lookup_fn=grep_search_from_tsova_tush,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_svan_to_english(
    svan_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Svan → English translation."""
    from src.translator.lookup import grep_search_from_svan

    return _construct_prompt(
        svan_sentence,
        input_lang="svan",
        output_lang="english",
        lookup_fn=grep_search_from_svan,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_english_to_svan(
    english_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for English → Svan translation."""
    return _construct_prompt(
        english_sentence,
        input_lang="english",
        output_lang="svan",
        lookup_fn=lambda word: grep_search_from_english_for_pack(word, "svan"),
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_georgian_to_svan(
    georgian_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Georgian → Svan translation."""
    return _construct_prompt(
        georgian_sentence,
        input_lang="georgian",
        output_lang="svan",
        lookup_fn=lambda word: grep_search_from_georgian_for_pack(word, "svan"),
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


def construct_prompt_from_svan_to_georgian(
    svan_sentence: str,
    *,
    exact_candidates_block: str = "",
    skip_word_lookups: bool = False,
    grammar_policy: Optional[str] = None,
) -> str:
    """Construct prompt for Svan → Georgian translation."""
    from src.translator.lookup import _grep_search_from_svan_source

    return _construct_prompt(
        svan_sentence,
        input_lang="svan",
        output_lang="georgian",
        lookup_fn=_grep_search_from_svan_source,
        exact_candidates_block=exact_candidates_block,
        skip_word_lookups=skip_word_lookups,
        grammar_policy=grammar_policy,
    )


PROMPT_BUILDERS = {
    ("mingrelian", "english"): construct_prompt_from_mingrelian_to_english,
    ("english", "mingrelian"): construct_prompt_from_english_to_mingrelian,
    ("mingrelian", "georgian"): construct_prompt_from_mingrelian_to_georgian,
    ("georgian", "mingrelian"): construct_prompt_from_georgian_to_mingrelian,
    ("tsova_tush", "english"): construct_prompt_from_tsova_tush_to_english,
    ("english", "tsova_tush"): construct_prompt_from_english_to_tsova_tush,
    ("tsova_tush", "georgian"): construct_prompt_from_tsova_tush_to_georgian,
    ("georgian", "tsova_tush"): construct_prompt_from_georgian_to_tsova_tush,
    ("svan", "english"): construct_prompt_from_svan_to_english,
    ("english", "svan"): construct_prompt_from_english_to_svan,
    ("svan", "georgian"): construct_prompt_from_svan_to_georgian,
    ("georgian", "svan"): construct_prompt_from_georgian_to_svan,
}
