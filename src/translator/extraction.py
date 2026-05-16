#!/usr/bin/env python3
"""Translator package helpers split from src.single_call_translator."""

import re
from typing import Optional

_GEORGIAN_SCRIPT_RE = re.compile(r"[\u10A0-\u10FF]")
_LATIN_SCRIPT_RE = re.compile(r"[A-Za-z]")
_COMMENTARY_LINE_RE = re.compile(
    r"""(?ix)
    ^\s*
    (?:
        [-*]\s+
        |\d+[\.)]\s+
        |notes?\s*:
        |based\s+on\b
        |to\s+translate\b
        |therefore\b
        |combining\b
        |putting\s+it\s+together\b
        |sentence\s+structure\b
        |grammar(?:atical)?\b
        |according\s+to\b
    )
    """
)


def _strip_list_prefix(line: str) -> str:
    """Remove common bullet/list wrappers without treating commentary as content."""
    return re.sub(r"^\s*(?:[-*]\s+|\d+[\.)]\s+)", "", line).strip("`\"' ")


def _looks_like_commentary(line: str) -> bool:
    """Detect explanation lines that should not become the user-facing translation."""
    stripped = line.strip()
    if _COMMENTARY_LINE_RE.search(stripped):
        return True
    if "**" in stripped or "->" in stripped:
        return True
    return False


def _fallback_translation_line(
    response_text: str,
    target_language: Optional[str],
    clean_line,
) -> Optional[str]:
    """Find a clean translation-looking line when the model omitted markers."""
    lines = [
        clean_line(_strip_list_prefix(line))
        for line in response_text.splitlines()
        if line.strip()
        and "<<<TRANSLATION>>>" not in line
        and "<<<END_TRANSLATION>>>" not in line
        and "FINAL_TRANSLATION_HERE" not in line
    ]
    lines = [line for line in lines if line and not _looks_like_commentary(line)]

    if not lines:
        return None

    if target_language in {"mingrelian", "georgian"}:
        for line in reversed(lines):
            if len(line) > 180:
                continue
            georgian_chars = len(_GEORGIAN_SCRIPT_RE.findall(line))
            latin_chars = len(_LATIN_SCRIPT_RE.findall(line))
            if georgian_chars and latin_chars <= max(2, georgian_chars // 2):
                return line

    if target_language == "english":
        for line in reversed(lines):
            if len(line) <= 180 and not _GEORGIAN_SCRIPT_RE.search(line):
                return line

    if len(lines[-1]) <= 180:
        return lines[-1]

    return None


def extract_translation(response_text: str, target_language: Optional[str] = None) -> str:
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

    # Final fallback: when markers are missing, prefer a short target-looking line.
    # If none exists, return the full text so protocol violations remain visible.
    fallback_line = _fallback_translation_line(
        response_text,
        target_language,
        _clean_extracted_text,
    )
    if fallback_line:
        return fallback_line

    return response_text.strip()
