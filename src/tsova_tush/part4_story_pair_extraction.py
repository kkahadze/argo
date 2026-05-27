#!/usr/bin/env python3
"""Story-level review-pair extraction for Tsovatush Texts Part IV."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
PAGE_NUMBER_RE = re.compile(r"^\d+$")
BATSBI_ANCHOR_RE = re.compile(r"^(?:befcu×nç|×efcu×nç):", re.IGNORECASE)
GEORGIAN_ANCHOR_RE = re.compile(r"^(?:mTxrobeli|mTxr\.):", re.IGNORECASE)
ENGLISH_ANCHOR_RE = re.compile(r"^Narrator:", re.IGNORECASE)


@dataclass(frozen=True)
class _StoryAnchor:
    kind: str
    anchor_index: int
    title_index: int


@dataclass(frozen=True)
class _StoryTriple:
    batsbi_anchor: _StoryAnchor
    georgian_anchor: _StoryAnchor
    english_anchor: _StoryAnchor


def _clean_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line or "").strip()


def _contents_start(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if "Contents" in lines[index]:
            return index
    return None


def _title_index_before(lines: list[str], *, anchor_index: int) -> int | None:
    for index in range(anchor_index - 1, -1, -1):
        line = _clean_line(lines[index])
        if not line or PAGE_NUMBER_RE.fullmatch(line):
            continue
        return index
    return None


def _story_anchors(lines: list[str], *, stop_index: int) -> list[_StoryAnchor]:
    anchors: list[_StoryAnchor] = []

    for index, raw_line in enumerate(lines[:stop_index]):
        line = _clean_line(raw_line)
        kind: str | None = None
        if BATSBI_ANCHOR_RE.match(line):
            kind = "batsbi"
        elif GEORGIAN_ANCHOR_RE.match(line):
            kind = "georgian"
        elif ENGLISH_ANCHOR_RE.match(line):
            kind = "english"

        if kind is None:
            continue

        title_index = _title_index_before(lines, anchor_index=index)
        if title_index is None:
            continue

        anchors.append(
            _StoryAnchor(
                kind=kind,
                anchor_index=index,
                title_index=title_index,
            )
        )

    return anchors


def _story_triples(lines: list[str], *, stop_index: int) -> list[_StoryTriple]:
    anchors = _story_anchors(lines, stop_index=stop_index)
    triples: list[_StoryTriple] = []
    anchor_index = 0

    while anchor_index <= len(anchors) - 3:
        batsbi_anchor, georgian_anchor, english_anchor = anchors[anchor_index : anchor_index + 3]
        if tuple(anchor.kind for anchor in (batsbi_anchor, georgian_anchor, english_anchor)) != (
            "batsbi",
            "georgian",
            "english",
        ):
            anchor_index += 1
            continue

        triples.append(
            _StoryTriple(
                batsbi_anchor=batsbi_anchor,
                georgian_anchor=georgian_anchor,
                english_anchor=english_anchor,
            )
        )
        anchor_index += 3

    return triples


def _section_text(lines: list[str], *, start_index: int, end_index: int) -> str:
    content_lines = [_clean_line(line) for line in lines[start_index + 1 : end_index]]
    filtered = [
        line
        for line in content_lines
        if line and not PAGE_NUMBER_RE.fullmatch(line)
    ]
    return _clean_line(" ".join(filtered))


def extract_part4_story_translation_pairs(
    text_or_path: str | Path,
    *,
    source_name: str,
) -> list[dict[str, str]]:
    """Return review-grade story triples from Part IV's repeated section layout."""

    if isinstance(text_or_path, Path):
        if not text_or_path.exists():
            return []
        text = text_or_path.read_text(encoding="utf-8")
    else:
        text = text_or_path

    lines = (text or "").splitlines()
    stop_index = _contents_start(lines)
    if stop_index is None:
        stop_index = len(lines)

    triples = _story_triples(lines, stop_index=stop_index)
    rows: list[dict[str, str]] = []

    for story_number, triple in enumerate(triples, start=1):
        next_batsbi_title_index = (
            triples[story_number].batsbi_anchor.title_index
            if story_number < len(triples)
            else stop_index
        )

        batsbi_text = _section_text(
            lines,
            start_index=triple.batsbi_anchor.anchor_index,
            end_index=triple.georgian_anchor.title_index,
        )
        georgian_text = _section_text(
            lines,
            start_index=triple.georgian_anchor.anchor_index,
            end_index=triple.english_anchor.title_index,
        )
        english_text = _section_text(
            lines,
            start_index=triple.english_anchor.anchor_index,
            end_index=next_batsbi_title_index,
        )

        if not batsbi_text or not georgian_text or not english_text:
            continue

        rows.append(
            {
                "source_id": f"{source_name}:story-{story_number:03d}",
                "source_name": source_name,
                "pair_type": "batsbi_georgian_english_story",
                "batsbi_text": batsbi_text,
                "batsbi_text_tokenized": batsbi_text,
                "georgian_translation": georgian_text,
                "english_translation": english_text,
                "confidence": "review",
                "notes": (
                    "Part IV story-level triad inferred from repeated "
                    "Batsbi/Georgian/English narrator-anchor order"
                ),
            }
        )

    return rows
