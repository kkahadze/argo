#!/usr/bin/env python3
from typing import List, Optional

# Example prompt showing how to translate dictionary entries
FIRST_SHOT_EXAMPLE = """
    Input:    

    Latinized Mingrelian word: vemxi 
    Mkhedruli Mingrelian word: ვემხი

    ========================================
    COMPLETE TRANSLATION ENTRIES:
    ========================================

    Lemma: vemx-i
    Number: 8022

    ვემხ-ი, ვემხვ-ი (ვემხ{ვ}ის) ზოოლ. ვეფხვი. ჩქჷნიანეფქ ვემხიცალო უზოგალო ქათირსეს: კ. სამუშ., ქართ.ზეპ., გვ. 101 -ჩვენიანები ვეფხვივით უზოგველად დაეჯახნენ (ეცნენ).
    
    Output:
    Lemma: vemx-i
    Number: 8022

    vemx-i, vemxv-i (vemxv᾽is): zoological term, "tiger."

    Example sentence (translation):
    "Swiftly, like a tiger, they swooped down upon the flock."
    — K. Samush, Georgian Oral Traditions, p. 101

    Glossed remark (translation):
    "Our people, like tigers, attacked en masse."
"""

def log_to_file(content: str, filename: str, logging_mode: bool = True) -> None:
    """
    Helper function to handle file logging based on logging mode.
    
    Args:
        content (str): Content to write to the file
        filename (str): Path to the log file
        logging_mode (bool): Whether to log the content to file (always True for file logging)
    """
    if logging_mode:
        with open(filename, 'w', encoding='utf-8') as log_file:
            log_file.write(content)

# Initial translation prompt
def get_initial_translation_prompt(dict_entries: List[str], logging_mode: bool = True) -> str:
    """
    Generate the initial translation prompt with the example and dictionary entries.
    
    Args:
        dict_entries (List[str]): List of dictionary entries to translate
        logging_mode (bool): Whether to log the prompt to file
        
    Returns:
        str: The complete initial translation prompt
    """
    command = f"\n\nCan you translate these entries from a Mingrelian-Georgian dictionary into English. Don't translate the Mingrelian into English, just the Georgian which you should be able to understand.\nHere is an example of what I expect you to do: {FIRST_SHOT_EXAMPLE}\n\nNow that you've seen an example, translate the following entries into English: \n"
    prompt = command + "\n" + "\n".join(dict_entries) + "\n"
    
    # Log the initial prompt if in logging mode
    log_to_file(prompt, 'initial_prompt_log.txt', logging_mode)
    
    return prompt

# Follow-up translation prompts
def get_follow_up_phrase(latinized: str, mkhedruli: str) -> str:
    """
    Generate the follow-up phrase prompt.
    
    Args:
        latinized (str): Latinized Mingrelian phrase
        mkhedruli (str): Mkhedruli Mingrelian phrase
        
    Returns:
        str: The formatted follow-up phrase prompt
    """
    return f"""
                        Now you will translate the following phrase into Georgian and then English
                        Phrase in Mingrelian (latinized): {latinized}
                        Phrase in Mingrelian (mkhedruli): {mkhedruli}
                        """

def get_grammar_text(grammar: str) -> str:
    """
    Generate the grammar text section of the prompt.
    
    Args:
        grammar (str): Grammar description text
        
    Returns:
        str: The formatted grammar section
    """
    return f"\nYou will also be given a Grammar written in English to help you translate the phrase. Beware of any differences in transcription that may occur between the grammer which is written in English by a linguist and uses IPA, and the dictionarty entries which use latinized Mingrelian at times and Mkhedruli (Georgian script) at other times.\n\n GRAMMAR START\n{grammar}\nGRAMMAR END\n"

def get_after_phrase(latinized: str, mkhedruli: str) -> str:
    """
    Generate the after phrase prompt with output format instructions.
    
    Args:
        latinized (str): Latinized Mingrelian phrase
        mkhedruli (str): Mkhedruli Mingrelian phrase
        
    Returns:
        str: The formatted after phrase prompt
    """
    return f"""
                        Now you will translate the phrase into Georgian and then English
                        Phrase in Mingrelian (latinized): {latinized}
                        Phrase in Mingrelian (mkhedruli): {mkhedruli}
                        After analyzing the phrase and thinking through the translation step by step, please output your answer in exactly this format:

                        Translation:
                        Georgian: [Georgian translation]  
                        English: [English translation]
                        """

def get_follow_up_prompt(
    follow_up_phrase: str, 
    grammar_text: str, 
    initial_response_text: str, 
    dict_entries: List[str],
    logging_mode: bool = True
) -> str:
    """
    Generate the complete follow-up prompt combining all components.
    
    Args:
        follow_up_phrase (str): The follow-up phrase prompt
        grammar_text (str): The grammar section
        initial_response_text (str): The initial translation response
        dict_entries (List[str]): The dictionary entries
        logging_mode (bool): Whether to log the prompt to file
        
    Returns:
        str: The complete follow-up prompt
    """
    prompt = follow_up_phrase + grammar_text + "\n\nDictionary entries translated into English:\n" + initial_response_text + follow_up_phrase
    
    # Log the follow-up prompt if in logging mode
    log_to_file(prompt, 'followup_prompt_log.txt', logging_mode)
    
    return prompt 