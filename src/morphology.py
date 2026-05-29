"""Pack-specific morphology analyzer hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.dictionary_store import normalize_pack_id


@dataclass(frozen=True)
class MorphologyAnalysis:
    evidence_type: str
    surface: str
    related_form: str = ""
    source_id: str = ""
    confidence: str = ""
    details: tuple[tuple[str, str], ...] = ()


class MorphologyAnalyzer(Protocol):
    def analyze(self, surface: str) -> tuple[MorphologyAnalysis, ...]: ...


def get_morphology_analyzer(pack_id: str) -> MorphologyAnalyzer | None:
    """Return a pack-specific analyzer when one exists."""
    if normalize_pack_id(pack_id) != "svan":
        return None
    from src.svan.morphology import ANALYZER

    return ANALYZER
