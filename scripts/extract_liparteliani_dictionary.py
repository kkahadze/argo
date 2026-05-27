#!/usr/bin/env python3
"""Recover conservative Svan-Georgian dictionary rows from Liparteliani OCR."""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "output" / "svan" / "sources" / "liparteliani-svan-georgian-fulltext.txt"
DEFAULT_READY_PATH = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_ready.tsv"
DEFAULT_REVIEW_PATH = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_review.tsv"
DEFAULT_STATS_PATH = REPO_ROOT / "output" / "svan" / "ready" / "liparteliani_dictionary_stats.tsv"

ENTRY_START_RE = re.compile(
    r"^\s*(?P<head>[ა-ჰჱჲჳჴჵჶ][ა-ჰჱჲჳჴჵჶა-ჰ0-9'’ʼ.,() \"/]{0,70}?)"
    r"\s*(?P<sep>=>|[–—-]{1,3})\s*(?P<gloss>.+?)\s*$"
)
LIKELY_ENTRY_HEAD_RE = re.compile(r"^[ა-ჰჱჲჳჴჵჶ][ა-ჰჱჲჳჴჵჶა-ჰ0-9'’ʼ.,() \"/]{0,70}$")
CONTINUATION_RE = re.compile(r"^[\s\"'„“(·]|^[ა-ჰჱჲჳჴჵჶ].{0,90}$")
GEORGIAN_RE = re.compile(r"[ა-ჰჱჲჳჴჵჶ]")
LATIN_NOISE_RE = re.compile(r"[A-Za-z]{2,}")
BAD_NOISE_RE = re.compile(r"[#%$@]|[_]{2,}|[=]{2,}|[|]{2,}")
PAGE_OR_ROMAN_RE = re.compile(r"^\s*(?:\d+|[IVXLCDM]+)\s*$")
MULTISPACE_RE = re.compile(r"\s+")
HEADERISH_HEADWORD_RE = re.compile(
    r"^(?:ნახ\.|ვ\.|სვანურ|ქართული ლექსიკონი)|ლექსიკონი|სვანურ-ქართული|სვანურ ქართული"
)


@dataclass
class Candidate:
    line_no: int
    headword: str
    gloss_parts: list[str]
    separator: str

    @property
    def gloss(self) -> str:
        if not self.gloss_parts:
            return ""

        merged = self.gloss_parts[0]
        for part in self.gloss_parts[1:]:
            if merged.endswith("-"):
                merged = merged[:-1] + part.lstrip(" „“\"'’ʼ")
            else:
                merged += " " + part
        return _clean_text(merged)


def _clean_text(text: str) -> str:
    cleaned = (text or "").replace("\f", " ").replace("¬", "-")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return MULTISPACE_RE.sub(" ", cleaned).strip(" \t")


def _normalize_headword(text: str) -> str:
    return _clean_text(text).strip(" ,.;:")


def _normalize_gloss(text: str) -> str:
    cleaned = _clean_text(text).replace("|", " ")
    cleaned = re.sub(r"^[\s.,;:·=]+", "", cleaned)
    return _clean_text(cleaned).strip()


def _looks_like_new_entry(line: str) -> bool:
    return ENTRY_START_RE.match(line or "") is not None


def _is_continuation(line: str) -> bool:
    clean = _clean_text(line)
    if not clean:
        return False
    if PAGE_OR_ROMAN_RE.match(clean):
        return False
    if _looks_like_new_entry(clean):
        return False
    if BAD_NOISE_RE.search(clean):
        return False
    if len(clean) > 120:
        return False
    return CONTINUATION_RE.match(clean) is not None


def _score_candidate(candidate: Candidate) -> tuple[str, str]:
    headword = candidate.headword
    gloss = candidate.gloss

    if not headword or not gloss:
        return "review", "missing_headword_or_gloss"
    if len(headword) < 2:
        return "review", "headword_too_short"
    if len(headword) > 70:
        return "review", "headword_too_long"
    if len(gloss) < 2:
        return "review", "gloss_too_short"
    if len(gloss) > 260:
        return "review", "gloss_too_long"
    if not LIKELY_ENTRY_HEAD_RE.match(headword):
        return "review", "headword_shape_suspicious"
    if not GEORGIAN_RE.search(headword):
        return "review", "headword_missing_georgian"
    if not GEORGIAN_RE.search(gloss):
        return "review", "gloss_missing_georgian"
    if BAD_NOISE_RE.search(headword) or BAD_NOISE_RE.search(gloss):
        return "review", "ocr_noise_symbols"
    if LATIN_NOISE_RE.search(headword):
        return "review", "latin_noise_in_headword"
    if HEADERISH_HEADWORD_RE.search(headword):
        return "review", "header_or_crossref_fragment"
    if len(headword.split()) > 3:
        return "review", "headword_too_many_tokens"
    if gloss.endswith("-"):
        return "review", "dangling_hyphenated_gloss"
    if "=>" in gloss or "==" in gloss:
        return "review", "gloss_embedded_separator_noise"
    if candidate.separator == "=>":
        return "review", "arrow_separator_lower_confidence"
    return "ready", "stable_dash_entry_shape"


def extract_candidates(text: str, *, body_start_line: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    lines = text.splitlines()
    ready: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    current: Candidate | None = None
    blank_gap = 0

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        status, reason = _score_candidate(current)
        row = {
            "source_id": f"liparteliani:line-{current.line_no:05d}",
            "headword_svan": current.headword,
            "georgian_gloss": current.gloss,
            "separator": current.separator,
            "status_reason": reason,
            "source_line": str(current.line_no),
        }
        if status == "ready":
            ready.append(row)
        else:
            review.append(row)
        current = None

    for idx, raw_line in enumerate(lines, start=1):
        if idx < body_start_line:
            continue

        line = _clean_text(raw_line)
        if not line:
            if current is not None and current.gloss_parts and current.gloss_parts[-1].endswith("-") and blank_gap < 2:
                blank_gap += 1
                continue
            flush()
            blank_gap = 0
            continue

        match = ENTRY_START_RE.match(line)
        if match:
            flush()
            blank_gap = 0
            current = Candidate(
                line_no=idx,
                headword=_normalize_headword(match.group("head")),
                gloss_parts=[_normalize_gloss(match.group("gloss"))],
                separator=match.group("sep"),
            )
            continue

        if current is not None and _is_continuation(line):
            current.gloss_parts.append(_normalize_gloss(line))
            blank_gap = 0
            continue

        flush()
        blank_gap = 0

    flush()
    return ready, review


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "source_id",
                "headword_svan",
                "georgian_gloss",
                "separator",
                "status_reason",
                "source_line",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_stats(path: Path, ready: list[dict[str, str]], review: list[dict[str, str]]) -> None:
    reasons: dict[str, int] = {}
    for row in review:
        reasons[row["status_reason"]] = reasons.get(row["status_reason"], 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["ready_rows", len(ready)])
        writer.writerow(["review_rows", len(review)])
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([f"review_reason:{reason}", count])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--ready-path", type=Path, default=DEFAULT_READY_PATH)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--stats-path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--body-start-line", type=int, default=2920)
    args = parser.parse_args()

    text = args.input_path.read_text(encoding="utf-8", errors="ignore")
    ready, review = extract_candidates(text, body_start_line=args.body_start_line)
    _write_tsv(args.ready_path, ready)
    _write_tsv(args.review_path, review)
    _write_stats(args.stats_path, ready, review)
    print(f"ready={len(ready)} review={len(review)}")


if __name__ == "__main__":
    main()
