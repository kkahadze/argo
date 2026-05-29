"""High-precision Svan morphology retrieval over reviewed runtime assets."""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.dictionary_store import _data_file_cache_key
from src.morphology import MorphologyAnalysis


@dataclass(frozen=True)
class _RuntimeIndex:
    variants_by_form: dict[str, tuple[dict[str, str], ...]]
    paradigms_by_form: dict[str, tuple[dict[str, str], ...]]
    lexicon_forms: dict[str, str]


def _normalize_form(value: str) -> str:
    compact = re.sub(r"\s+", " ", (value or "").strip())
    return unicodedata.normalize("NFC", compact).casefold()


def _read_rows(file_path: str, mtime_ns: Optional[int]) -> tuple[dict[str, str], ...]:
    if mtime_ns is None:
        return ()
    with Path(file_path).open("r", encoding="utf-8-sig", newline="") as file:
        return tuple(csv.DictReader(file, delimiter="\t"))


def _index_rows(
    rows: tuple[dict[str, str], ...],
    field: str,
) -> dict[str, tuple[dict[str, str], ...]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = _normalize_form(row.get(field) or "")
        if key:
            index[key].append(row)
    return {key: tuple(value) for key, value in index.items()}


@lru_cache(maxsize=4)
def _get_runtime_index(
    variants_key: tuple[str, Optional[int]],
    paradigms_key: tuple[str, Optional[int]],
    kk_key: tuple[str, Optional[int]],
) -> _RuntimeIndex:
    variant_rows = _read_rows(*variants_key)
    paradigm_rows = _read_rows(*paradigms_key)
    kk_rows = _read_rows(*kk_key)
    lexicon_forms = {
        _normalize_form(row.get("word") or ""): (row.get("word") or "").strip()
        for row in kk_rows
        if (row.get("word") or "").strip()
    }
    return _RuntimeIndex(
        variants_by_form=_index_rows(variant_rows, "query_form_nfc"),
        paradigms_by_form=_index_rows(paradigm_rows, "form_nfc"),
        lexicon_forms=lexicon_forms,
    )


def _runtime_index() -> _RuntimeIndex:
    return _get_runtime_index(
        _data_file_cache_key("attested_variants.tsv", "svan"),
        _data_file_cache_key("paradigm_forms.tsv", "svan"),
        _data_file_cache_key("kk.tsv", "svan"),
    )


class SvanMorphologyAnalyzer:
    """Return only reviewed forms or lexicon-validated conservative analyses."""

    _NOUN_SUFFIXES = (
        ("შუ̂", "INST"),
        ("იშ", "GEN"),
        ("ს", "DAT"),
        ("დ", "ADV/ERG"),
    )

    def analyze(self, surface: str) -> tuple[MorphologyAnalysis, ...]:
        index = _runtime_index()
        normalized = _normalize_form(surface)
        analyses: list[MorphologyAnalysis] = []

        for row in index.variants_by_form.get(normalized, ()):
            analyses.append(
                MorphologyAnalysis(
                    evidence_type="Attested Topuria-Kaldani variant",
                    surface=(row.get("query_form_raw") or surface).strip(),
                    related_form=(row.get("related_form_raw") or "").strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    confidence=(row.get("confidence") or "").strip(),
                    details=(
                        ("Relation", (row.get("relation_type") or "").strip()),
                        ("Dictionary pages", _page_range(row)),
                    ),
                )
            )

        for row in index.paradigms_by_form.get(normalized, ()):
            analyses.append(
                MorphologyAnalysis(
                    evidence_type="Verified She 2024 paradigm form",
                    surface=(row.get("form_raw") or surface).strip(),
                    source_id=(row.get("source_id") or "").strip(),
                    confidence=(row.get("confidence") or "").strip(),
                    details=(
                        ("Dialect", (row.get("dialect") or "").strip()),
                        ("Lemma/gloss", (row.get("lemma_raw") or "").strip()),
                        ("Paradigm", (row.get("paradigm_raw") or "").strip()),
                        (
                            "Slot",
                            " ".join(
                                part
                                for part in (
                                    (row.get("person_slot") or "").strip(),
                                    (row.get("column_raw") or "").strip(),
                                )
                                if part
                            ),
                        ),
                    ),
                )
            )

        if not analyses:
            for suffix, feature in self._NOUN_SUFFIXES:
                if not normalized.endswith(_normalize_form(suffix)):
                    continue
                stem = normalized[: -len(_normalize_form(suffix))]
                related_form = index.lexicon_forms.get(stem)
                if related_form:
                    analyses.append(
                        MorphologyAnalysis(
                            evidence_type="Tuite 2023 noun suffix analysis",
                            surface=surface,
                            related_form=related_form,
                            source_id="Tuite 2023, Svan declension classes VI-VIII",
                            confidence="lexicon-validated conservative inference",
                            details=(("Feature", feature),),
                        )
                    )
                break

        return tuple(analyses)


def _page_range(row: dict[str, str]) -> str:
    start = (row.get("page_start") or "").strip()
    end = (row.get("page_end") or "").strip()
    if not start:
        return ""
    return start if not end or end == start else f"{start}-{end}"


ANALYZER = SvanMorphologyAnalyzer()
