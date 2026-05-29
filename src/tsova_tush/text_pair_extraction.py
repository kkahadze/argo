#!/usr/bin/env python3
"""Deterministic translation-pair extraction for Tsovatush text books."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator


NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.,-]\s*(.*?)\s*$")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextPairCandidate:
    """High-confidence bilingual row suitable for TSV export."""

    source_id: str
    batsbi_text: str
    georgian_translation: str
    english_translation: str | None
    confidence: float
    notes: str


@dataclass(frozen=True)
class _NumberedItem:
    number: int
    text: str


@dataclass(frozen=True)
class _NumberedRun:
    items: tuple[_NumberedItem, ...]

    @property
    def sequence(self) -> tuple[int, ...]:
        return tuple(item.number for item in self.items)


def _clean_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line).strip()


def _finalize_item(
    items: list[_NumberedItem],
    number: int | None,
    parts: list[str],
) -> None:
    if number is None:
        return

    text = _clean_line(" ".join(part for part in parts if part))
    if text:
        items.append(_NumberedItem(number=number, text=text))


def _finalize_run(
    runs: list[_NumberedRun],
    items: list[_NumberedItem],
    *,
    min_run_items: int,
) -> None:
    if len(items) >= min_run_items:
        runs.append(_NumberedRun(items=tuple(items)))


def _parse_numbered_runs(text: str, *, min_run_items: int) -> tuple[_NumberedRun, ...]:
    runs: list[_NumberedRun] = []
    items: list[_NumberedItem] = []
    current_number: int | None = None
    current_parts: list[str] = []
    saw_blank_after_item = False

    for raw_line in (text or "").splitlines():
        line = _clean_line(raw_line)
        numbered_match = NUMBERED_LINE_RE.match(line)

        if numbered_match:
            if saw_blank_after_item and items:
                _finalize_item(items, current_number, current_parts)
                _finalize_run(runs, items, min_run_items=min_run_items)
                items = []
                current_number = None
                current_parts = []

            _finalize_item(items, current_number, current_parts)
            current_number = int(numbered_match.group(1))
            current_parts = [numbered_match.group(2)]
            saw_blank_after_item = False
            continue

        if current_number is None:
            continue

        if not line:
            saw_blank_after_item = True
            continue

        if saw_blank_after_item:
            _finalize_item(items, current_number, current_parts)
            _finalize_run(runs, items, min_run_items=min_run_items)
            items = []
            current_number = None
            current_parts = []
            saw_blank_after_item = False
            continue

        current_parts.append(line)

    _finalize_item(items, current_number, current_parts)
    _finalize_run(runs, items, min_run_items=min_run_items)
    return tuple(runs)


def iter_numbered_translation_pairs(
    text: str,
    *,
    source_id: str,
    min_run_items: int = 2,
) -> Iterator[TextPairCandidate]:
    """Yield strict Part-I-style Batsbi -> Georgian translation candidates.

    The extractor only accepts adjacent numbered runs with identical item-number
    sequences. This matches the clean repeated-numbering pattern in Part I and
    intentionally ignores damaged or loosely aligned blocks.
    """

    runs = _parse_numbered_runs(text, min_run_items=min_run_items)
    emitted_run_index = 0
    run_index = 0

    while run_index < len(runs) - 1:
        batsbi_run = runs[run_index]
        georgian_run = runs[run_index + 1]

        if batsbi_run.sequence != georgian_run.sequence:
            run_index += 1
            continue

        emitted_run_index += 1
        for batsbi_item, georgian_item in zip(
            batsbi_run.items,
            georgian_run.items,
            strict=True,
        ):
            yield TextPairCandidate(
                source_id=(
                    f"{source_id}:run-{emitted_run_index:03d}:"
                    f"item-{batsbi_item.number:03d}"
                ),
                batsbi_text=batsbi_item.text,
                georgian_translation=georgian_item.text,
                english_translation=None,
                confidence=1.0,
                notes="adjacent numbered runs with identical item sequence",
            )

        run_index += 2
