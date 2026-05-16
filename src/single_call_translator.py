#!/usr/bin/env python3
"""
Single-call translation system using dictionary lookups and one LLM API call.
Adapted from explore_rag_dict.ipynb notebook.
"""
import re
import string
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
from src.dictionary_store import (
    choose_kk_bridge_gloss as _choose_kk_bridge_gloss,
    get_data_path as _get_data_path,
    get_dictionary_store,
    split_figurative_gloss as _split_figurative_gloss,
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

# English
def grep_search_pairs(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search sentence_pairs.tsv for English translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store().search_sentence_pairs(word, standalone_only=standalone_only)
    return result.output, result.has_standalone_matches


# Russian
def grep_search_gal(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search gal.tsv for Russian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store().search_gal(word, standalone_only=standalone_only)
    return result.output, result.has_standalone_matches


# Russian and Georgian
def grep_search_kk(word: str, *, standalone_only: bool = False) -> tuple[str, bool]:
    """
    Search kk.tsv for Russian and Georgian translations, prioritizing standalone word matches.
    Returns: (result_string, has_standalone_matches)
    """
    result = get_dictionary_store().search_kk(word, standalone_only=standalone_only)
    return result.output, result.has_standalone_matches


# Georgian
def grep_search_kajaia(word: str, *, standalone_only: bool = False) -> str:
    """
    Search kajaia_cleaned.txt for Georgian dictionary entries.
    Splits text by empty lines and returns the block containing the search term.
    Prioritizes standalone word matches over substring matches.
    """
    return get_dictionary_store().search_context(word, standalone_only=standalone_only)


def grep_search_from_english(word: str) -> str:
    """
    Search all dictionaries from English word.
    Short-circuits kajaia search if standalone matches found in extractive dictionaries.
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
    
    # Only search kajaia if no standalone matches found in extractive dictionaries
    has_any_standalone = (pairs_has_standalone or gal_has_standalone or 
                          kk_ru_has_standalone or kk_ge_has_standalone)
    
    if not has_any_standalone:
        output += grep_search_kajaia(res_ge)

    if len(output) > 10000:
        return output[:10000]

    return output


def grep_search_from_mingrelian(word: str) -> str:
    """
    Search all dictionaries from Mingrelian word.
    Short-circuits kajaia search if standalone matches found in extractive dictionaries.
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
    
    # Only search kajaia if no standalone matches found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or gal_has_standalone or kk_has_standalone
    
    kajaia_result = ""
    if not has_any_standalone:
        kajaia_result = grep_search_kajaia(word)
        output += kajaia_result

    # If absolutely nothing matched across all four sources, try a conservative
    # case-suffix stripping fallback (e.g., ...თ → stem).
    case_fallback_applied = False
    if not (pairs_result or gal_result or kk_result or kajaia_result):
        for stem in _case_strip_candidates_mkhedruli(word):
            # First: standalone-only search. If we find any standalone match for this
            # candidate, we return ONLY standalone matches and stop (no partial matches,
            # and no further candidates like the bare stem).
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True)
            kajaia2_s = grep_search_kajaia(stem, standalone_only=True)

            output2_s = pairs2_s + gal2_s + kk2_s + kajaia2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(kajaia2_s)

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
            kajaia2 = "" if has_any_standalone2 else grep_search_kajaia(stem)
            output2 += kajaia2

            if output2:
                output += f"\n[Case-stripped fallback: {word} → {stem}]\n"
                output += output2
                case_fallback_applied = True
                break

    # If we STILL have no hits, assume this might be a verb with a preverb attached
    # and try stripping a simple preverb.
    if not (pairs_result or gal_result or kk_result or kajaia_result or case_fallback_applied):
        for stem in _preverb_strip_candidates_mkhedruli(word):
            # Prefer standalone-only results for the stripped stem.
            pairs2_s, pairs2_has_s = grep_search_pairs(stem, standalone_only=True)
            gal2_s, gal2_has_s = grep_search_gal(stem, standalone_only=True)
            kk2_s, kk2_has_s = grep_search_kk(stem, standalone_only=True)
            kajaia2_s = grep_search_kajaia(stem, standalone_only=True)

            output2_s = pairs2_s + gal2_s + kk2_s + kajaia2_s
            has_any_standalone2 = pairs2_has_s or gal2_has_s or kk2_has_s or bool(kajaia2_s)

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
            kajaia2 = "" if has_any_standalone2 else grep_search_kajaia(stem)
            output2 += kajaia2

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
    Short-circuits kajaia search if standalone matches found in extractive dictionaries.
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
    
    # Only search kajaia if no standalone matches found in extractive dictionaries
    has_any_standalone = pairs_has_standalone or kk_has_standalone or gal_has_standalone
    
    if not has_any_standalone:
        output += grep_search_kajaia(word)

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
    Check if the exact input text exists in extractive dictionaries (not kajaia).
    Returns the translation if found, None otherwise.
    
    This is the simple direct lookup without Google Translate augmentation.
    """
    store = get_dictionary_store()
    
    # Check sentence_pairs.tsv (Mingrelian ↔ English)
    if (source_lang, target_lang) in [("mingrelian", "english"), ("english", "mingrelian")]:
        if source_lang == "mingrelian":
            matches = store.exact_sentence_mingrelian(input_text)
            if matches:
                return matches[0].english
        else:
            matches = store.exact_sentence_english(input_text)
            if matches:
                return matches[0].mingrelian
    
    # Check kk.tsv (Mingrelian ↔ Russian ↔ Georgian)
    if source_lang == "mingrelian" and target_lang == "georgian":
        matches = store.exact_kk_mingrelian(input_text)
        if matches:
            georgian_primary, _ = _split_figurative_gloss(matches[0].georgian, "ka")
            return georgian_primary or matches[0].georgian

    elif source_lang == "georgian" and target_lang == "mingrelian":
        matches = store.exact_kk_georgian(input_text)
        if matches:
            return matches[0].mingrelian
    
    # Check gal.tsv (Russian ↔ Mingrelian)
    if source_lang == "russian" and target_lang == "mingrelian":
        matches = store.exact_gal_russian(input_text)
        if matches:
            return matches[0].mingrelian

    elif source_lang == "mingrelian" and target_lang == "russian":
        matches = store.exact_gal_mingrelian(input_text)
        if matches:
            return matches[0].russian
    
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
    store = get_dictionary_store()
    
    # Priority 1: Search sentence_pairs.tsv (English ↔ Mingrelian, cleanest)
    for row in store.exact_sentence_mingrelian(text):
        return (row.mingrelian, row.english, "en")
    for row in store.exact_sentence_english(text):
        return (row.mingrelian, row.english, "en")
    
    # Priority 2: Search gal.tsv (Russian ↔ Mingrelian, reliable)
    for row in store.exact_gal_mingrelian(text):
        return (row.mingrelian, row.russian, "ru")
    for row in store.exact_gal_russian(text):
        return (row.mingrelian, row.russian, "ru")
    
    # Priority 3: Search kk.tsv (may have data quality issues, use as fallback)
    for row in store.exact_kk_mingrelian(text):
        bridge_text, lang_code = _choose_kk_bridge_gloss(row.russian, row.georgian, target_lang)
        if bridge_text and lang_code:
            return (row.mingrelian, bridge_text, lang_code)
        return (row.mingrelian, row.georgian, "ka")
    for row in store.exact_kk_georgian(text):
        georgian_primary, _ = _split_figurative_gloss(row.georgian, "ka")
        return (row.mingrelian, georgian_primary or row.georgian, "ka")
    for row in store.exact_kk_russian(text):
        russian_primary, _ = _split_figurative_gloss(row.russian, "ru")
        return (row.mingrelian, russian_primary or row.russian, "ru")
    
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
