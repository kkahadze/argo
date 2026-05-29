#!/usr/bin/env python3
"""Build clean held-out DoReCo Svan-to-Georgian Promptfoo evaluation rows."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP_PATH = REPO_ROOT / "output" / "svan" / "sources" / "doreco_svan1243_core_v2.0.zip"
OUTPUT_PATH = Path(__file__).resolve().parent / "datasets" / "svan-doreco-heldout-georgian.csv"
SOURCE_URL = "https://doreco.huma-num.fr/languages/svan1243"
TARGET_PER_BUCKET = 16
BUCKETS = (
    ("short", 20, 80),
    ("medium", 81, 220),
    ("long", 221, 10_000),
)
NOISE_VALUES = {"", "<p:>", "****"}
ARTIFACT_MARKERS = ("_", "...", "…", "{", "}", "[", "]", "(", ")")
TERMINAL_PUNCTUATION = frozenset(".!?")


@dataclass(frozen=True)
class DorecoSegment:
    source_id: str
    recording: str
    speaker: str
    reference: str
    svan: str
    georgian: str


def normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


def bucket_for_length(length: int) -> str | None:
    for name, minimum, maximum in BUCKETS:
        if minimum <= length <= maximum:
            return name
    return None


def _usable_text(value: str) -> bool:
    normalized = normalize_whitespace(value)
    return normalized not in NOISE_VALUES and "****" not in normalized


def _has_georgian_letters(value: str) -> bool:
    return any("\u10a0" <= character <= "\u10ff" for character in value)


def _clean_baseline_segment(svan: str, georgian: str) -> bool:
    if not all(_usable_text(value) and _has_georgian_letters(value) for value in (svan, georgian)):
        return False
    if any(marker in svan or marker in georgian for marker in ARTIFACT_MARKERS):
        return False
    if svan[-1] not in TERMINAL_PUNCTUATION or georgian[-1] not in TERMINAL_PUNCTUATION:
        return False
    return bucket_for_length(len(svan)) in {"short", "medium"}


def _tier_values(tier: ET.Element, *, use_reference_ids: bool) -> dict[str, str]:
    values: dict[str, str] = {}
    for annotation in tier.findall("./ANNOTATION/*"):
        key = annotation.get("ANNOTATION_REF") if use_reference_ids else annotation.get("ANNOTATION_ID")
        value = normalize_whitespace(annotation.findtext("ANNOTATION_VALUE", default=""))
        if key and value:
            values[key] = value
    return values


def _load_eaf_segments(member: str, xml_bytes: bytes) -> list[DorecoSegment]:
    recording = Path(member).stem
    root = ET.fromstring(xml_bytes)
    tiers = {tier.get("TIER_ID", ""): tier for tier in root.findall("TIER")}
    segments: list[DorecoSegment] = []

    for tier_id, svan_tier in tiers.items():
        if not tier_id.startswith("tr1@"):
            continue
        speaker = tier_id.removeprefix("tr1@")
        georgian_tier = tiers.get(f"fg@{speaker}")
        reference_tier = tiers.get(svan_tier.get("PARENT_REF", ""))
        if georgian_tier is None or reference_tier is None:
            continue

        references = _tier_values(reference_tier, use_reference_ids=False)
        svan_values = _tier_values(svan_tier, use_reference_ids=True)
        georgian_values = _tier_values(georgian_tier, use_reference_ids=True)
        for reference_id, svan in svan_values.items():
            georgian = georgian_values.get(reference_id, "")
            reference = references.get(reference_id, "")
            if not _clean_baseline_segment(svan, georgian) or not _usable_text(reference):
                continue
            segments.append(
                DorecoSegment(
                    source_id=f"doreco:{recording}:{speaker}:{reference}",
                    recording=recording,
                    speaker=speaker,
                    reference=reference,
                    svan=svan,
                    georgian=georgian,
                )
            )

    return segments


def load_segments(zip_path: Path) -> list[DorecoSegment]:
    """Load clean aligned Mkhedruli Svan and Georgian segments from DoReCo EAF tiers."""
    segments: list[DorecoSegment] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".eaf"):
                continue
            for segment in _load_eaf_segments(member, archive.read(member)):
                key = (
                    segment.recording,
                    segment.speaker,
                    segment.reference,
                    segment.svan,
                    segment.georgian,
                )
                if key in seen:
                    continue
                seen.add(key)
                segments.append(segment)
    return segments


def select_balanced_segments(
    segments: list[DorecoSegment],
    target_per_bucket: int | None,
) -> list[DorecoSegment]:
    if target_per_bucket is None:
        return sorted(segments, key=lambda segment: (segment.recording, segment.reference, segment.speaker))

    selected: list[DorecoSegment] = []
    for bucket_name, _, _ in BUCKETS:
        candidates = [
            segment
            for segment in segments
            if bucket_for_length(len(segment.svan)) == bucket_name
        ]
        candidates.sort(key=lambda segment: (len(segment.svan), segment.recording, segment.reference))
        count = min(len(candidates), target_per_bucket)
        if count == 0:
            continue

        stride = (len(candidates) - 1) / (count - 1) if count > 1 else 0
        picked_indexes: list[int] = []
        used: set[int] = set()
        for position in range(count):
            index = round(position * stride)
            while index in used and index + 1 < len(candidates):
                index += 1
            if index in used:
                index = next(candidate_index for candidate_index in range(len(candidates)) if candidate_index not in used)
            used.add(index)
            picked_indexes.append(index)
        selected.extend(candidates[index] for index in picked_indexes)

    return selected


def write_dataset(path: Path, segments: list[DorecoSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source_id",
        "recording",
        "speaker",
        "doreco_ref",
        "length_bucket",
        "svan",
        "georgian",
        "confidence",
        "notes",
        "source_url",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "source_id": segment.source_id,
                    "recording": segment.recording,
                    "speaker": segment.speaker,
                    "doreco_ref": segment.reference,
                    "length_bucket": bucket_for_length(len(segment.svan)),
                    "svan": segment.svan,
                    "georgian": segment.georgian,
                    "confidence": "held_out",
                    "notes": "DoReCo Mkhedruli tr1 segment with clean Georgian fg translation",
                    "source_url": SOURCE_URL,
                }
            )


def build_dataset(
    *,
    zip_path: Path = DEFAULT_ZIP_PATH,
    output_path: Path = OUTPUT_PATH,
    target_per_bucket: int | None = TARGET_PER_BUCKET,
) -> int:
    selected = select_balanced_segments(load_segments(zip_path), target_per_bucket)
    write_dataset(output_path, selected)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--target-per-bucket", type=int, default=TARGET_PER_BUCKET)
    args = parser.parse_args()

    count = build_dataset(
        zip_path=args.zip_path,
        output_path=args.output_path,
        target_per_bucket=args.target_per_bucket,
    )
    print(f"rows={count}")
    print(f"output={args.output_path}")


if __name__ == "__main__":
    main()
