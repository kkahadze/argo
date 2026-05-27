#!/usr/bin/env python3
"""Build translator-ready Tsova-Tush / Batsbi ingestion assets."""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.tsova_tush.normalization import (
    detect_batsbi_scheme,
    html_editorial_markup_to_tokens,
    normalize_batsbi_unicode,
    source_specific_cleanup,
    to_canonical_practical,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "tsova-tush" / "ready"
DEFAULT_TITUS_BASE_URL = (
    "https://titus.fkidg1.uni-frankfurt.de/texte/etce/cauc/batsbi/tt_dict"
)
LABEL_PREFIX_RE = re.compile(r"^[A-Za-z ]+:\s*")
BLANK_BLOCK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class AssetBuildInputs:
    ids_tsv: Path
    grammar_full: Path
    grammar_compact: Path
    grammar_classic: Path
    texts_part1: Path
    texts_part4: Path
    output_dir: Path
    titus_pages: tuple[tuple[str, str], ...] = ()


class _SpanCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[dict[str, object]] = []
        self.records: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "span":
            attr_map = {name.lower(): value or "" for name, value in attrs}
            self._stack.append({"id": attr_map.get("id", ""), "parts": []})
            return

        if tag in {"sup", "sub"}:
            for frame in self._stack:
                parts = frame["parts"]
                assert isinstance(parts, list)
                parts.append(f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            if not self._stack:
                return
            frame = self._stack.pop()
            span_id = frame["id"]
            parts = frame["parts"]
            assert isinstance(span_id, str)
            assert isinstance(parts, list)
            if span_id:
                self.records.append((span_id, "".join(parts)))
            return

        if tag in {"sup", "sub"}:
            for frame in self._stack:
                parts = frame["parts"]
                assert isinstance(parts, list)
                parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        for frame in self._stack:
            parts = frame["parts"]
            assert isinstance(parts, list)
            parts.append(data)


def _collect_span_fragments(html_text: str) -> list[tuple[str, str]]:
    parser = _SpanCollector()
    parser.feed(html_text or "")
    parser.close()
    return parser.records


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _split_semicolon_variants(text: str) -> list[str]:
    return [part.strip() for part in (text or "").split(";") if part.strip()]


def _strip_structural_label(text: str) -> str:
    return LABEL_PREFIX_RE.sub("", normalize_batsbi_unicode(text), count=1)


def _clean_document_text(text: str, *, source_id: str) -> str:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        cleaned = source_specific_cleanup(source_id, raw_line)
        lines.append(cleaned)

    collapsed = "\n".join(lines)
    collapsed = BLANK_BLOCK_RE.sub("\n\n", collapsed)
    return collapsed.strip()


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_ids_rows(ids_tsv: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    lexicon_rows: list[dict[str, str]] = []
    override_rows: list[dict[str, str]] = []

    with ids_tsv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            chapter_id = normalize_batsbi_unicode(row.get("chapter_id") or "")
            entry_id = normalize_batsbi_unicode(row.get("entry_id") or "")
            meaning = normalize_batsbi_unicode(row.get("meaning") or "")
            comment = normalize_batsbi_unicode(row.get("comment") or "")
            phonemic_forms = _split_semicolon_variants(row.get("Tsova-Tush_Phonemic") or "")
            cyrillic_forms = _split_semicolon_variants(row.get("Tsova-Tush_CyrillTrans") or "")

            for variant_index, (phonemic, cyrillic) in enumerate(
                zip_longest(phonemic_forms, cyrillic_forms, fillvalue=""),
                start=1,
            ):
                source_form = phonemic or cyrillic
                if not meaning or not source_form:
                    continue

                scheme = detect_batsbi_scheme(source_form)
                canonical = to_canonical_practical(source_form, scheme)
                lexicon_row = {
                    "chapter_id": chapter_id,
                    "entry_id": entry_id,
                    "meaning": meaning,
                    "variant_index": str(variant_index),
                    "cyrillic": normalize_batsbi_unicode(cyrillic),
                    "phonemic": normalize_batsbi_unicode(phonemic),
                    "canonical_practical": canonical,
                    "comment": comment,
                }
                lexicon_rows.append(lexicon_row)

                override_rows.extend(
                    [
                        {
                            "source_language": "english",
                            "target_language": "tsova_tush",
                            "source_text": meaning,
                            "target_text": canonical,
                            "source_id": f"ids:{entry_id}:{variant_index}",
                        },
                        {
                            "source_language": "tsova_tush",
                            "target_language": "english",
                            "source_text": canonical,
                            "target_text": meaning,
                            "source_id": f"ids:{entry_id}:{variant_index}",
                        },
                    ]
                )

    return lexicon_rows, override_rows


def parse_titus_dictionary_page(
    html_text: str,
    *,
    source_url: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    examples: list[dict[str, str]] = []
    current_entry: dict[str, str] = {}
    pending_example: dict[str, str] | None = None

    for span_id, inner_html in _collect_span_fragments(html_text or ""):
        rendered = html_editorial_markup_to_tokens(inner_html)
        plain = rendered.plain_text
        tokenized = rendered.tokenized_text

        if span_id == "h3":
            current_entry = {
                "lemma": _strip_structural_label(plain),
                "lemma_number": "",
                "batsbi_mkhedruli": "",
                "batsbi_transcription": "",
                "georgian_gloss": "",
                "russian_gloss": "",
                "source_url": source_url,
            }
            pending_example = None
            continue

        if not current_entry:
            continue

        if span_id == "h4":
            current_entry["lemma_number"] = _strip_structural_label(plain)
        elif span_id == "emttb22":
            current_entry["batsbi_mkhedruli"] = plain
        elif span_id == "ectt22":
            current_entry["batsbi_transcription"] = plain
        elif span_id == "mxngb16":
            current_entry["georgian_gloss"] = plain
        elif span_id == "slrub16":
            current_entry["russian_gloss"] = plain
            entries.append(dict(current_entry))
        elif span_id == "h5":
            pending_example = {
                "lemma": current_entry["lemma"],
                "lemma_number": current_entry["lemma_number"],
                "example_number": _strip_structural_label(plain),
                "batsbi_text": "",
                "batsbi_text_tokenized": "",
                "georgian_translation": "",
                "source_url": source_url,
            }
        elif span_id == "emtt16" and pending_example is not None:
            pending_example["batsbi_text"] = plain
            pending_example["batsbi_text_tokenized"] = tokenized
        elif span_id == "mxng16" and pending_example is not None:
            pending_example["georgian_translation"] = plain
            if pending_example["batsbi_text"] and pending_example["georgian_translation"]:
                examples.append(dict(pending_example))
            pending_example = None

    return entries, examples


def fetch_titus_pages(
    *,
    base_url: str = DEFAULT_TITUS_BASE_URL,
    max_pages: int = 1100,
    max_workers: int = 8,
) -> tuple[tuple[str, str], ...]:
    def fetch_page(page_number: int) -> tuple[int, str, str | None]:
        url = f"{base_url}/tt_di{page_number:03d}.htm"
        request = Request(url, headers={"User-Agent": "margo-tsova-tush-builder/1.0"})
        try:
            with urlopen(request, timeout=20) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            if error.code == 404:
                return page_number, url, None
            raise
        except URLError:
            return page_number, url, None
        if "Batsbi-Georgian-Russian Dictionary" not in payload:
            return page_number, url, None
        return page_number, url, payload

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fetched = list(executor.map(fetch_page, range(1, max_pages + 1)))

    pages: list[tuple[str, str]] = []
    for _, url, payload in fetched:
        if payload is None:
            break
        pages.append((url, payload))
    return tuple(pages)


def _build_context_source(
    *,
    grammar_full: str,
    grammar_compact: str,
    grammar_classic: str,
    texts_part1: str,
    texts_part4: str,
) -> str:
    sections = [
        ("Wichers Schreur 2025 full grammar", grammar_full),
        ("Hauk and Harris sketch grammar", grammar_compact),
        ("Holisky and Gagua 1994 grammar chapter", grammar_classic),
        ("Tsovatush Texts Part I 2009", texts_part1),
        ("Tsovatush Texts Part IV 2017", texts_part4),
    ]
    blocks = []
    for title, text in sections:
        if text.strip():
            blocks.append(f"===== SOURCE: {title} =====\n{text.strip()}")
    return "\n\n".join(blocks)


def build_assets(inputs: AssetBuildInputs) -> dict[str, object]:
    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ids_rows, override_rows = build_ids_rows(inputs.ids_tsv)
    _write_csv(
        output_dir / "ids_lexicon.csv",
        (
            "chapter_id",
            "entry_id",
            "meaning",
            "variant_index",
            "cyrillic",
            "phonemic",
            "canonical_practical",
            "comment",
        ),
        ids_rows,
    )
    _write_tsv(
        output_dir / "ids_exact_overrides.tsv",
        (
            "source_language",
            "target_language",
            "source_text",
            "target_text",
            "source_id",
        ),
        override_rows,
    )

    grammar_full = _clean_document_text(_read_text(inputs.grammar_full), source_id="wichers")
    grammar_compact = _clean_document_text(
        _read_text(inputs.grammar_compact),
        source_id="hauk_harris_sketch",
    )
    grammar_classic = _clean_document_text(_read_text(inputs.grammar_classic), source_id="holisky")
    texts_part1 = _clean_document_text(_read_text(inputs.texts_part1), source_id="bertlani_pdf")
    texts_part4 = _clean_document_text(_read_text(inputs.texts_part4), source_id="bertlani_pdf")

    text_outputs = {
        "wichers_schreur_2025.txt": grammar_full,
        "hauk_harris_sketch.txt": grammar_compact,
        "holisky_gagua_1994.txt": grammar_classic,
        "tsovatush_texts_part1_2009.txt": texts_part1,
        "tsovatush_texts_part4_2017.txt": texts_part4,
        "context_source.txt": _build_context_source(
            grammar_full=grammar_full,
            grammar_compact=grammar_compact,
            grammar_classic=grammar_classic,
            texts_part1=texts_part1,
            texts_part4=texts_part4,
        ),
    }
    for filename, text in text_outputs.items():
        _write_text(output_dir / filename, text)

    titus_entries: list[dict[str, str]] = []
    titus_examples: list[dict[str, str]] = []
    for source_url, html_text in inputs.titus_pages:
        page_entries, page_examples = parse_titus_dictionary_page(
            html_text,
            source_url=source_url,
        )
        titus_entries.extend(page_entries)
        titus_examples.extend(page_examples)

    _write_csv(
        output_dir / "titus_dictionary.csv",
        (
            "lemma",
            "lemma_number",
            "batsbi_mkhedruli",
            "batsbi_transcription",
            "georgian_gloss",
            "russian_gloss",
            "source_url",
        ),
        titus_entries,
    )
    _write_tsv(
        output_dir / "titus_examples.tsv",
        (
            "lemma",
            "lemma_number",
            "example_number",
            "batsbi_text",
            "batsbi_text_tokenized",
            "georgian_translation",
            "source_url",
        ),
        titus_examples,
    )

    manifest: dict[str, object] = {
        "counts": {
            "ids_lexicon_rows": len(ids_rows),
            "ids_override_rows": len(override_rows),
            "titus_dictionary_rows": len(titus_entries),
            "titus_example_rows": len(titus_examples),
        },
        "outputs": {
            "ids_lexicon_csv": "ids_lexicon.csv",
            "ids_exact_overrides_tsv": "ids_exact_overrides.tsv",
            "titus_dictionary_csv": "titus_dictionary.csv",
            "titus_examples_tsv": "titus_examples.tsv",
            **{filename: filename for filename in text_outputs},
        },
    }
    _write_text(
        output_dir / "build_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids-tsv",
        type=Path,
        default=REPO_ROOT / "tmp" / "data" / "ids-tsova-tush-533.tab",
    )
    parser.add_argument(
        "--grammar-full",
        type=Path,
        default=REPO_ROOT / "tmp" / "data" / "hauk-2025-language-contact-tsova-tush.txt",
    )
    parser.add_argument(
        "--grammar-compact",
        type=Path,
        default=REPO_ROOT / "tmp" / "data" / "hauk-harris-batsbi-sketch.txt",
    )
    parser.add_argument(
        "--grammar-classic",
        type=Path,
        default=REPO_ROOT / "tmp" / "data" / "holisky-gagua-1994-tsova-tush.txt",
    )
    parser.add_argument(
        "--texts-part1",
        type=Path,
        default=REPO_ROOT / "tmp" / "pdfs" / "wova-tushuri-teqstebi-2009-texted.txt",
    )
    parser.add_argument(
        "--texts-part4",
        type=Path,
        default=REPO_ROOT / "tmp" / "data" / "bertlani-tsovatush-texts-part4.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fetch-titus",
        action="store_true",
        help="Download TITUS dictionary pages and emit dictionary/example exports.",
    )
    parser.add_argument(
        "--titus-max-pages",
        type=int,
        default=1100,
        help="Upper bound when crawling TITUS dictionary pages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titus_pages = (
        fetch_titus_pages(max_pages=args.titus_max_pages)
        if args.fetch_titus
        else ()
    )
    manifest = build_assets(
        AssetBuildInputs(
            ids_tsv=args.ids_tsv,
            grammar_full=args.grammar_full,
            grammar_compact=args.grammar_compact,
            grammar_classic=args.grammar_classic,
            texts_part1=args.texts_part1,
            texts_part4=args.texts_part4,
            output_dir=args.output_dir,
            titus_pages=titus_pages,
        )
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
