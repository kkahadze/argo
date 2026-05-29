#!/usr/bin/env python3
"""Build Argo-compatible runtime data files for the Svan translator pack."""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READY_DIR = REPO_ROOT / "output" / "svan" / "ready"
DEFAULT_PRIVATE_DATA_DIR = REPO_ROOT / "argo" / "private_data" / "svan"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _write_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_master_lexicon(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("headword", "headword_raw", "translation"))
        writer.writeheader()


def _build_context_blocks(pair_rows: list[dict[str, str]]) -> list[str]:
    blocks: list[str] = []
    for row in pair_rows:
        svan = (row.get("svan_text") or "").strip()
        georgian = (row.get("georgian_translation") or "").strip()
        if not svan or not georgian:
            continue
        source_id = (row.get("source_id") or "titus").strip()
        blocks.append(
            "\n".join(
                (
                    f"===== SOURCE: {source_id} =====",
                    f"Svan: {svan}",
                    f"Georgian: {georgian}",
                )
            )
        )
    return blocks


def _build_phrase_evidence_blocks(
    pair_rows: list[dict[str, str]],
    *,
    source_name: str,
) -> list[str]:
    blocks: list[str] = []
    for row in pair_rows:
        svan = (row.get("svan_text") or "").strip()
        georgian = (row.get("georgian_translation") or "").strip()
        if not svan or not georgian:
            continue
        source_id = (row.get("source_id") or source_name).strip()
        blocks.append(
            "\n".join(
                (
                    f"===== SOURCE: {source_id} ({source_name}; prompt evidence only) =====",
                    f"Svan: {svan}",
                    f"Georgian: {georgian}",
                )
            )
        )
    return blocks


def _build_parallel_overrides(pair_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build deterministic bidirectional mappings for trusted parallel text."""
    candidate_rows: list[tuple[tuple[str, str, str], dict[str, str]]] = []
    target_values: dict[tuple[str, str, str], set[str]] = {}
    for row in pair_rows:
        svan = (row.get("svan_text") or "").strip()
        georgian = (row.get("georgian_translation") or "").strip()
        if not svan or not georgian:
            continue
        mappings = [("georgian", "svan", georgian, svan)]
        variants = (row.get("svan_variants") or svan).split("|||")
        mappings.extend(
            ("svan", "georgian", variant.strip(), georgian)
            for variant in variants
            if variant.strip()
        )
        for source_language, target_language, source_text, target_text in mappings:
            source_key = (
                source_language.casefold(),
                target_language.casefold(),
                re.sub(r"\s+", " ", source_text).casefold(),
            )
            target_values.setdefault(source_key, set()).add(
                re.sub(r"\s+", " ", target_text).casefold()
            )
            candidate_rows.append(
                (
                    source_key,
                    {
                        "source_language": source_language,
                        "target_language": target_language,
                        "source_text": source_text,
                        "target_text": target_text,
                    },
                )
            )

    rows: list[dict[str, str]] = []
    emitted: set[tuple[str, str, str]] = set()
    for source_key, row in candidate_rows:
        if source_key in emitted or len(target_values[source_key]) != 1:
            continue
        emitted.add(source_key)
        rows.append(row)
    return rows


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _normalize_form(value: str) -> str:
    return unicodedata.normalize("NFC", _compact_text(value)).casefold()


def _build_attested_variant_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compile explicit Topuria-Kaldani equivalence edges without choosing a canonical form."""
    compiled: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        relation_type = _compact_text(row.get("relation_type") or "")
        if relation_type != "same_as":
            continue
        source_form = _compact_text(row.get("source_headword_raw") or "")
        target_form = _compact_text(row.get("target_headword_raw") or "")
        if not source_form or not target_form or _normalize_form(source_form) == _normalize_form(target_form):
            continue
        for query_form, related_form in (
            (source_form, target_form),
            (target_form, source_form),
        ):
            compiled_row = {
                "query_form_raw": query_form,
                "query_form_nfc": _normalize_form(query_form),
                "related_form_raw": related_form,
                "related_form_nfc": _normalize_form(related_form),
                "relation_type": relation_type,
                "source_id": _compact_text(row.get("source_id") or ""),
                "page_start": _compact_text(row.get("page_start") or ""),
                "page_end": _compact_text(row.get("page_end") or ""),
                "source_line_start": _compact_text(row.get("source_line_start") or ""),
                "source_line_end": _compact_text(row.get("source_line_end") or ""),
                "confidence": _compact_text(row.get("confidence") or ""),
            }
            key = (
                compiled_row["query_form_nfc"],
                compiled_row["related_form_nfc"],
                relation_type,
                compiled_row["source_id"],
            )
            if key in seen:
                continue
            seen.add(key)
            compiled.append(compiled_row)
    return compiled


def _source_pages(source_id: str) -> tuple[str, str]:
    match = re.search(
        r"PDF pages?\s+(.+?)(?:\s+render)?\s+\(printed pages?\s+(.+?)\)",
        source_id,
    )
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _build_paradigm_form_rows(text: str) -> list[dict[str, str]]:
    """Parse reviewed She tables into exact runtime form cells."""
    rows: list[dict[str, str]] = []
    source_id = ""
    pdf_pages = ""
    printed_pages = ""
    dialect = ""
    lemma = ""
    paradigm = ""
    columns: list[str] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("===== SOURCE:") and line.endswith("====="):
            source_id = line.removeprefix("===== SOURCE:").removesuffix("=====").strip()
            pdf_pages, printed_pages = _source_pages(source_id)
            dialect = ""
            lemma = ""
            paradigm = ""
            columns = []
        elif line.startswith("DIALECT:"):
            dialect = line.removeprefix("DIALECT:").strip()
        elif line.startswith("LEMMA/GLOSS:"):
            lemma = line.removeprefix("LEMMA/GLOSS:").strip()
        elif line.startswith("PARADIGM:"):
            paradigm = line.removeprefix("PARADIGM:").strip()
            columns = []
        elif line.startswith("Person\t"):
            columns = [part.strip() for part in raw_line.split("\t")[1:]]
        elif line.startswith("QA:"):
            columns = []
        elif line and columns and "\t" in raw_line:
            parts = [part.strip() for part in raw_line.split("\t")]
            person_slot = parts[0]
            forms = parts[1:]
            if len(forms) != len(columns):
                raise ValueError(
                    f"She morphology row {line_no} has {len(forms)} cells; expected {len(columns)}"
                )
            for column, form in zip(columns, forms):
                if not form:
                    continue
                rows.append(
                    {
                        "form_raw": form,
                        "form_nfc": _normalize_form(form),
                        "dialect": dialect,
                        "lemma_raw": lemma,
                        "paradigm_raw": paradigm,
                        "column_raw": column,
                        "person_slot": person_slot,
                        "source_id": source_id,
                        "pdf_pages": pdf_pages,
                        "printed_pages": printed_pages,
                        "confidence": "verified",
                    }
                )
    return rows


def _build_kk_rows(
    liparteliani_rows: list[dict[str, str]],
    topuria_rows: list[dict[str, str]],
    supplemental_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], int, int]:
    """Compile lexical resources in priority order, dropping exact duplicates."""
    compiled: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    topuria_added = 0
    supplemental_added = 0
    sources = (
        (liparteliani_rows, "headword_svan", "georgian_gloss", "base"),
        (topuria_rows, "headword_svan_raw", "georgian_definition", "topuria"),
        (supplemental_rows or [], "svan", "georgian", "supplemental"),
    )
    for rows, word_field, gloss_field, source_kind in sources:
        for row in rows:
            word = _compact_text(row.get(word_field) or "")
            gloss = _compact_text(row.get(gloss_field) or "")
            key = (word.casefold(), gloss.casefold())
            if not word or not gloss or key in seen:
                continue
            seen.add(key)
            compiled.append(
                {
                    "word": word,
                    "ipa": "",
                    "russian_def": "",
                    "georgian_def": gloss,
                }
            )
            if source_kind == "topuria":
                topuria_added += 1
            elif source_kind == "supplemental":
                supplemental_added += 1
    return compiled, topuria_added, supplemental_added


def _build_russian_svan_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compile audited Russian-Svan lexical rows for Google bridge retrieval."""
    compiled: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        russian = _compact_text(row.get("russian") or "")
        svan = _compact_text(row.get("svan") or "")
        key = (russian.casefold(), svan.casefold())
        if not russian or not svan or key in seen:
            continue
        seen.add(key)
        compiled.append({"russian": russian, "svan": svan})
    return compiled


def _build_english_svan_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compile audited English-Svan lexical rows for direct retrieval."""
    compiled: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        english = _compact_text(row.get("english") or "")
        svan = _compact_text(row.get("svan") or "")
        key = (svan.casefold(), english.casefold())
        if not english or not svan or key in seen:
            continue
        seen.add(key)
        compiled.append({"svan": svan, "english": english})
    return compiled


def _build_retrieval_parallel_pairs(
    pair_rows: list[dict[str, str]],
    *,
    default_source_family: str = "topuria-kaldani",
    default_evidence_type: str = "ordinary_example",
) -> list[dict[str, str]]:
    """Compile audited examples as retrieval evidence, not exact overrides."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in pair_rows:
        svan = _compact_text(row.get("svan_text") or "")
        georgian = _compact_text(row.get("georgian_translation") or "")
        key = (svan.casefold(), georgian.casefold())
        if not svan or not georgian or key in seen:
            continue
        seen.add(key)
        domain_type = _compact_text(row.get("domain_type") or row.get("domain") or "")
        evidence_type = _compact_text(row.get("evidence_type") or "")
        if not evidence_type:
            evidence_type = "domain_example" if domain_type else default_evidence_type
        rows.append(
            {
                "low_resource": svan,
                "georgian": georgian,
                "source_id": _compact_text(row.get("source_id") or "topuria-kaldani"),
                "source_family": _compact_text(row.get("source_family") or default_source_family),
                "evidence_type": evidence_type,
                "domain_type": domain_type,
            }
        )
    return rows


def _merge_parallel_pairs(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for row in group:
            key = (row["low_resource"].casefold(), row["georgian"].casefold())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def build_runtime_data(
    *,
    ready_dir: Path = DEFAULT_READY_DIR,
    private_data_dir: Path = DEFAULT_PRIVATE_DATA_DIR,
) -> dict[str, int]:
    private_data_dir.mkdir(parents=True, exist_ok=True)

    dictionary_rows = _read_tsv(ready_dir / "liparteliani_dictionary_ready.tsv")
    topuria_dictionary_rows = _read_tsv(ready_dir / "topuria_kaldani_dictionary_ready.tsv")
    supplemental_georgian_lexicon_rows = _read_tsv(ready_dir / "supplemental_svan_georgian_lexicon_ready.tsv")
    supplemental_russian_rows = _read_tsv(ready_dir / "supplemental_russian_svan_lexicon_ready.tsv")
    supplemental_english_rows = _read_tsv(ready_dir / "supplemental_svan_english_lexicon_ready.tsv")
    pair_rows = _read_tsv(ready_dir / "titus_svan_georgian_pairs_high_confidence.tsv")
    topuria_example_rows = _read_tsv(ready_dir / "topuria_kaldani_example_pairs_ready.tsv")
    topuria_domain_rows = _read_tsv(ready_dir / "topuria_kaldani_example_pairs_domain.tsv")
    supplemental_pair_rows = _read_tsv(ready_dir / "supplemental_svan_georgian_pairs_ready.tsv")
    quizlet_phrase_rows = _read_tsv(ready_dir / "quizlet_svan_georgian_conversation_pairs.tsv")
    topuria_cross_reference_rows = _read_tsv(ready_dir / "topuria_kaldani_cross_references.tsv")
    grammar_path = ready_dir / "tuite_svan_grammar_2023.txt"
    morphology_support_path = ready_dir / "she_2024_morphology_compact_verified.txt"

    kk_rows, topuria_dictionary_added, supplemental_georgian_added = _build_kk_rows(
        dictionary_rows,
        topuria_dictionary_rows,
        supplemental_georgian_lexicon_rows,
    )
    gal_rows = _build_russian_svan_rows(supplemental_russian_rows)
    sentence_pair_rows = _build_english_svan_rows(supplemental_english_rows)
    base_parallel_pair_rows = _build_retrieval_parallel_pairs(topuria_example_rows)
    topuria_domain_pair_rows = _build_retrieval_parallel_pairs(
        topuria_domain_rows,
        default_evidence_type="domain_example",
    )
    supplemental_parallel_pair_rows = _build_retrieval_parallel_pairs(
        supplemental_pair_rows,
        default_source_family="supplemental",
    )
    parallel_pair_rows = _merge_parallel_pairs(
        base_parallel_pair_rows,
        topuria_domain_pair_rows,
        supplemental_parallel_pair_rows,
    )
    base_and_domain_pair_rows = _merge_parallel_pairs(base_parallel_pair_rows, topuria_domain_pair_rows)

    _write_master_lexicon(private_data_dir / "master-lexicon-mkhedruli.csv")
    _write_tsv(
        private_data_dir / "sentence_pairs.tsv",
        ("svan", "english"),
        sentence_pair_rows,
    )
    _write_tsv(
        private_data_dir / "gal.tsv",
        ("russian", "svan"),
        gal_rows,
    )
    _write_tsv(
        private_data_dir / "kk.tsv",
        ("word", "ipa", "russian_def", "georgian_def"),
        kk_rows,
    )
    _write_tsv(
        private_data_dir / "parallel_pairs.tsv",
        ("low_resource", "georgian", "source_id", "source_family", "evidence_type", "domain_type"),
        parallel_pair_rows,
    )
    override_rows = _build_parallel_overrides(pair_rows + quizlet_phrase_rows)
    _write_tsv(
        private_data_dir / "translation_overrides.tsv",
        ("source_language", "target_language", "source_text", "target_text"),
        override_rows,
    )

    context_blocks = _build_context_blocks(pair_rows)
    quizlet_context_blocks = _build_phrase_evidence_blocks(
        quizlet_phrase_rows,
        source_name="quizlet_svan_georgian_conversation_pairs",
    )
    (private_data_dir / "context_source.txt").write_text(
        "\n\n".join(context_blocks + quizlet_context_blocks),
        encoding="utf-8",
    )

    grammar = grammar_path.read_text(encoding="utf-8") if grammar_path.exists() else ""
    (private_data_dir / "tuite.txt").write_text(grammar, encoding="utf-8")
    (private_data_dir / "tuite_compact.txt").write_text(grammar, encoding="utf-8")
    morphology_support = (
        morphology_support_path.read_text(encoding="utf-8")
        if morphology_support_path.exists()
        else ""
    )
    (private_data_dir / "morphology_support.txt").write_text(morphology_support, encoding="utf-8")
    attested_variant_rows = _build_attested_variant_rows(topuria_cross_reference_rows)
    _write_tsv(
        private_data_dir / "attested_variants.tsv",
        (
            "query_form_raw",
            "query_form_nfc",
            "related_form_raw",
            "related_form_nfc",
            "relation_type",
            "source_id",
            "page_start",
            "page_end",
            "source_line_start",
            "source_line_end",
            "confidence",
        ),
        attested_variant_rows,
    )
    paradigm_form_rows = _build_paradigm_form_rows(morphology_support)
    _write_tsv(
        private_data_dir / "paradigm_forms.tsv",
        (
            "form_raw",
            "form_nfc",
            "dialect",
            "lemma_raw",
            "paradigm_raw",
            "column_raw",
            "person_slot",
            "source_id",
            "pdf_pages",
            "printed_pages",
            "confidence",
        ),
        paradigm_form_rows,
    )

    return {
        "master_lexicon_rows": 0,
        "kk_rows": len(kk_rows),
        "topuria_dictionary_rows": topuria_dictionary_added,
        "supplemental_russian_rows": len(gal_rows),
        "supplemental_english_rows": len(sentence_pair_rows),
        "supplemental_georgian_lexicon_rows": supplemental_georgian_added,
        "parallel_pair_rows": len(parallel_pair_rows),
        "topuria_domain_pair_rows": max(0, len(base_and_domain_pair_rows) - len(base_parallel_pair_rows)),
        "supplemental_parallel_pair_rows": max(0, len(parallel_pair_rows) - len(base_and_domain_pair_rows)),
        "sentence_pair_rows": len(sentence_pair_rows),
        "override_rows": len(override_rows),
        "context_blocks": len(context_blocks),
        "quizlet_context_blocks": len(quizlet_context_blocks),
        "morphology_support_chars": len(morphology_support),
        "attested_variant_rows": len(attested_variant_rows),
        "paradigm_form_rows": len(paradigm_form_rows),
    }


def main() -> None:
    counts = build_runtime_data()
    for key, value in counts.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
