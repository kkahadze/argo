#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import re

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
