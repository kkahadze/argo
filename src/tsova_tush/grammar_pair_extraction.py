#!/usr/bin/env python3
"""Strict grammar-example pair extraction for Tsova-Tush / Batsbi texts."""
from __future__ import annotations

import re
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
EXAMPLE_PREFIX_RE = re.compile(r"^(?:\(\d+\)\s*)?(?:\([a-z]\)\s*)?", re.IGNORECASE)
MARKER_ONLY_RE = re.compile(r"^(?:\(\d+\)\s*)?(?:\([a-z]\)\s*)?$", re.IGNORECASE)
GLOSS_TAG_RE = re.compile(
    r"(?:^|[-/ ])(?:[123](?:SG|PL)|[A-Z]{2,})(?:$|[-/ ])"
)
TSOVA_SOURCE_HINT_RE = re.compile(
    r"[£¨™§¥ß|0-9ʼ:čšžħʕǯɬāēīōūŏŭ]|(?:^|\\s)[A-Z][a-z]*[A-Z]"
)

OPENING_QUOTES = frozenset({"‘", "'", '"', "“"})
CLOSING_QUOTES = frozenset({"’", "ʼ", "'", '"', "”"})


def _clean_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line or "").strip()


def _strip_example_prefix(line: str) -> str:
    return _clean_line(EXAMPLE_PREFIX_RE.sub("", _clean_line(line), count=1))


def _parse_translation_payload(payload: str) -> str | None:
    text = _clean_line(payload)
    if not text or text[0] not in OPENING_QUOTES:
        return None

    remainder = text[1:]
    for index, character in enumerate(remainder):
        if character not in CLOSING_QUOTES:
            continue

        tail = remainder[index + 1 :].strip()
        if tail and not tail.startswith(("(", "[")):
            continue

        translation = _clean_line(remainder[:index])
        return translation or None

    return None


def _translation_from_line(line: str) -> tuple[str, str] | None:
    cleaned = _clean_line(line)
    if cleaned.startswith("="):
        translation = _parse_translation_payload(cleaned[1:].strip())
        if translation:
            return translation, "explicit_equals_translation"
        return None

    translation = _parse_translation_payload(cleaned)
    if translation:
        return translation, "standalone_quoted_translation"
    return None


def _looks_like_gloss_line(line: str) -> bool:
    return bool(GLOSS_TAG_RE.search(_clean_line(line)))


def _previous_nonempty_line(lines: list[str], start_index: int) -> tuple[int, str] | None:
    for index in range(start_index, -1, -1):
        line = _clean_line(lines[index])
        if line:
            return index, line
    return None


def _looks_like_batsbi_source(line: str) -> bool:
    cleaned = _strip_example_prefix(line)
    if not cleaned or MARKER_ONLY_RE.fullmatch(_clean_line(line)):
        return False

    tokens = cleaned.split()
    if len(tokens) <= 2:
        return True
    return bool(TSOVA_SOURCE_HINT_RE.search(cleaned))


def _source_for_translation(
    lines: list[str],
    translation_index: int,
) -> tuple[str, bool] | None:
    previous = _previous_nonempty_line(lines, translation_index - 1)
    if previous is None:
        return None

    previous_index, previous_line = previous
    if _looks_like_gloss_line(previous_line):
        source_line = _previous_nonempty_line(lines, previous_index - 1)
        if source_line is None:
            return None
        _, source = source_line
        if _looks_like_batsbi_source(source):
            return _strip_example_prefix(source), True
        return None

    if _looks_like_batsbi_source(previous_line):
        return _strip_example_prefix(previous_line), False
    return None


def extract_grammar_translation_pairs(
    text_or_path: str | Path,
    *,
    source_id: str | None = None,
    source_name: str | None = None,
) -> list[dict[str, str]]:
    """Return only high-confidence grammar/example translation candidates.

    Accepted layouts are intentionally narrow:
    - a Batsbi line followed immediately by a standalone quoted translation;
    - a Batsbi line, an interlinear gloss line, then a quoted translation;
    - the same interlinear layout with an explicit ``= ‘translation’`` line.
    """

    resolved_source_id = source_id or source_name
    resolved_source_name = source_name or source_id
    if not resolved_source_id or not resolved_source_name:
        raise ValueError("source_id or source_name is required")

    if isinstance(text_or_path, Path):
        if not text_or_path.exists():
            return []
        text = text_or_path.read_text(encoding="utf-8")
    else:
        text = text_or_path

    rows: list[dict[str, str]] = []
    lines = (text or "").splitlines()

    for index, line in enumerate(lines):
        parsed_translation = _translation_from_line(line)
        if parsed_translation is None:
            continue

        english_translation, notes = parsed_translation
        source = _source_for_translation(lines, index)
        if source is None:
            continue

        batsbi_text, saw_gloss_line = source
        if notes == "standalone_quoted_translation" and saw_gloss_line:
            notes = "interlinear_quoted_translation"

        rows.append(
            {
                "source_id": resolved_source_id,
                "source_name": resolved_source_name,
                "pair_type": "batsbi_english",
                "batsbi_text": batsbi_text,
                "batsbi_text_tokenized": batsbi_text,
                "georgian_translation": "",
                "english_translation": english_translation,
                "confidence": "high",
                "notes": notes,
            }
        )

    return rows
