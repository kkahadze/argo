#!/usr/bin/env python3
"""Strict TITUS Svan-Georgian parallel-text extraction helpers."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup


LINE_BLOCK_RE = re.compile(
    r'<span id=h5>.*?SPT_II_(?P<part>\d+)_(?P<page>\d+)_(?P<line>\d+).*?</span>'
    r'(?P<body>.*?)<span id=n16>',
    re.IGNORECASE | re.DOTALL,
)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TitusTextLine:
    part: int
    page: int
    line: int
    text: str
    source_url: str


@dataclass(frozen=True)
class TitusAlignedPair:
    source_id: str
    part: int
    page: int
    ordinal: int
    svan_text: str
    georgian_translation: str
    confidence: str
    notes: str
    svan_source_url: str
    georgian_source_url: str


@dataclass(frozen=True)
class TitusLinkBlock:
    source_coord: tuple[int, int, int]
    target_coord: tuple[int, int, int]
    text: str
    source_url: str


def _clean_fragment(fragment: str) -> str:
    soup = BeautifulSoup(fragment or "", "html.parser")
    rendered = soup.get_text(" ", strip=True)
    rendered = html.unescape(rendered)
    return WHITESPACE_RE.sub(" ", rendered).strip()


def parse_titus_text_lines(html_text: str, *, source_url: str) -> tuple[TitusTextLine, ...]:
    """Extract structured text lines from one TITUS prose-text HTML page."""

    lines: list[TitusTextLine] = []
    for match in LINE_BLOCK_RE.finditer(html_text or ""):
        text = _clean_fragment(match.group("body"))
        if not text:
            continue

        lines.append(
            TitusTextLine(
                part=int(match.group("part")),
                page=int(match.group("page")),
                line=int(match.group("line")),
                text=text,
                source_url=source_url,
            )
        )

    return tuple(lines)


def _coord_from_fragment(fragment: str) -> tuple[int, int, int] | None:
    match = re.search(r"SPT_II_(\d+)_(\d+)_(\d+)", fragment or "")
    if not match:
        return None
    return tuple(int(match.group(index)) for index in range(1, 4))


def _coord_from_locate_href(href: str) -> tuple[int, int, int] | None:
    query = parse_qs(urlparse(href or "").query)
    raw = [query.get(key, [""])[0] for key in ("lx3", "lx4", "lx5")]
    if not all(value.isdigit() for value in raw):
        return None
    return tuple(int(value) for value in raw)


def parse_titus_link_blocks(html_text: str, *, source_url: str) -> tuple[TitusLinkBlock, ...]:
    """Extract TITUS alignment blocks keyed by the embedded cross-reference links."""

    lines = list(parse_titus_text_lines(html_text, source_url=source_url))
    line_index = {(line.part, line.page, line.line): index for index, line in enumerate(lines)}
    soup = BeautifulSoup(html_text or "", "html.parser")

    starts: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href") or ""
        if "locate.asp" not in href:
            continue

        source_span = anchor.find_next("span", id="h5")
        source_coord = _coord_from_fragment(str(source_span)) if source_span else None
        target_coord = _coord_from_locate_href(href)
        if source_coord is None or target_coord is None:
            continue
        if source_coord not in line_index:
            continue

        starts.append((source_coord, target_coord))

    blocks: list[TitusLinkBlock] = []
    for index, (source_coord, target_coord) in enumerate(starts):
        start_index = line_index[source_coord]
        end_index = (
            line_index[starts[index + 1][0]]
            if index + 1 < len(starts)
            else len(lines)
        )
        block_lines = lines[start_index:end_index]
        text = "\n".join(line.text for line in block_lines).strip()
        if not text:
            continue
        blocks.append(
            TitusLinkBlock(
                source_coord=source_coord,
                target_coord=target_coord,
                text=text,
                source_url=source_url,
            )
        )

    return tuple(blocks)


def align_parallel_blocks(
    svan_blocks: Iterable[TitusLinkBlock],
    georgian_blocks: Iterable[TitusLinkBlock],
) -> tuple[list[TitusAlignedPair], list[dict[str, str]]]:
    """Emit only reciprocal TITUS cross-reference blocks."""

    svan_index = {block.source_coord: block for block in svan_blocks}
    georgian_index = {block.source_coord: block for block in georgian_blocks}
    aligned: list[TitusAlignedPair] = []
    review_rows: list[dict[str, str]] = []

    for ordinal, source_coord in enumerate(sorted(svan_index), start=1):
        svan_block = svan_index[source_coord]
        georgian_block = georgian_index.get(svan_block.target_coord)
        if georgian_block is None:
            review_rows.append(
                {
                    "part": str(source_coord[0]),
                    "page": str(source_coord[1]),
                    "issue": "missing_target_block",
                    "svan_line_count": "1",
                    "georgian_line_count": "0",
                }
            )
            continue
        if georgian_block.target_coord != svan_block.source_coord:
            review_rows.append(
                {
                    "part": str(source_coord[0]),
                    "page": str(source_coord[1]),
                    "issue": "non_reciprocal_cross_reference",
                    "svan_line_count": "1",
                    "georgian_line_count": "1",
                }
            )
            continue

        part, page, line_no = source_coord
        aligned.append(
            TitusAlignedPair(
                source_id=f"titus:spto2:part-{part:03d}:page-{page:03d}:line-{line_no:03d}",
                part=part,
                page=page,
                ordinal=ordinal,
                svan_text=svan_block.text,
                georgian_translation=georgian_block.text,
                confidence="high",
                notes="reciprocal TITUS cross-reference block",
                svan_source_url=svan_block.source_url,
                georgian_source_url=georgian_block.source_url,
            )
        )

    return aligned, review_rows


def build_part_review_rows(
    svan_lines: Iterable[TitusTextLine],
    georgian_lines: Iterable[TitusTextLine],
) -> list[dict[str, str]]:
    """Build conservative part-level review rows for broader manual salvage."""

    svan_by_part: dict[int, list[str]] = {}
    georgian_by_part: dict[int, list[str]] = {}

    for line in svan_lines:
        svan_by_part.setdefault(line.part, []).append(line.text)
    for line in georgian_lines:
        georgian_by_part.setdefault(line.part, []).append(line.text)

    rows: list[dict[str, str]] = []
    for part in sorted(set(svan_by_part) & set(georgian_by_part)):
        rows.append(
            {
                "source_id": f"titus:spto2:part-{part:03d}",
                "pair_type": "svan_georgian_part_review",
                "svan_text": "\n".join(svan_by_part[part]),
                "georgian_translation": "\n".join(georgian_by_part[part]),
                "confidence": "review",
                "notes": "whole-part parallel text; requires downstream segmentation or manual review",
            }
        )

    return rows


def write_text_snapshot(path: Path, lines: Iterable[TitusTextLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(line.text for line in lines).rstrip()
    path.write_text((text + "\n") if text else "", encoding="utf-8")
