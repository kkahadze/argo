#!/usr/bin/env python3
"""Extract raw ICE Svan dictionary rows into a deduplicated TSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "svan" / "ready" / "ice_svan_dictionary_raw.tsv"
ICE_ENDPOINT = "https://ice.tsu.ge/liv/svan1.php"
GEORGIAN_LETTERS = tuple("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")


def _fetch(prefix: str) -> str:
    payload = urlencode({"first": prefix, "second": ""}).encode("utf-8")
    request = Request(
        ICE_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "margo-svan-ingestion/1.0",
        },
    )
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


def _parse_rows(html_text: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_text or "", "html.parser")
    rows: list[tuple[str, str]] = []
    for tr in soup.select("table.result_table tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        headword = _clean_text("".join(cells[0].strings))
        svan_raw = _clean_text("".join(cells[1].strings))
        if headword and svan_raw:
            rows.append((headword, svan_raw))
    return rows


def extract_rows(*, max_depth: int) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    stack = list(GEORGIAN_LETTERS)

    while stack:
        prefix = stack.pop()
        parsed = _parse_rows(_fetch(prefix))
        if len(parsed) >= 25 and len(prefix) < max_depth:
            stack.extend(prefix + letter for letter in GEORGIAN_LETTERS)

        for headword, svan_raw in parsed:
            key = (headword, svan_raw)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "headword_georgian": headword,
                    "svan_raw": svan_raw,
                    "source_query": prefix,
                    "normalization_status": "raw_legacy_font_slots_preserved",
                    "source_url": ICE_ENDPOINT,
                }
            )

    rows.sort(key=lambda row: (row["headword_georgian"], row["svan_raw"]))
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "headword_georgian",
                "svan_raw",
                "source_query",
                "normalization_status",
                "source_url",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-depth", type=int, default=2)
    args = parser.parse_args()

    rows = extract_rows(max_depth=args.max_depth)
    write_rows(args.output_path, rows)
    print(f"rows={len(rows)} output={args.output_path}")


if __name__ == "__main__":
    main()
