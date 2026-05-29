#!/usr/bin/env python3
"""Normalization and transliteration helpers for Tsova-Tush / Batsbi assets."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable


APOSTROPHE_NORMALIZATION = str.maketrans(
    {
        "'": "ʼ",
        "’": "ʼ",
        "ʻ": "ʼ",
        "ʹ": "ʼ",
        "`": "ʼ",
    }
)

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
GEORGIAN_RE = re.compile(r"[\u10a0-\u10ff]")
LATIN_RE = re.compile(r"[A-Za-zĀāĒēĪīŌōŪūČčŠšŽžĞğŎŏŬŭ]")
PRACTICAL_WORD_CHAR_CLASS = "A-Za-zĀāĒēĪīŌōŪūČčŠšŽžĞğŎŏŬŭɬħʕǯʒʼⁿɁ"
INLINE_MORPHEME_HYPHEN_RE = re.compile(
    rf"(?<=[{PRACTICAL_WORD_CHAR_CLASS}])-(?=[{PRACTICAL_WORD_CHAR_CLASS}])"
)
LEGACY_REDUCED_VOWEL_MARKER_RE = re.compile(
    rf"(?<=[{PRACTICAL_WORD_CHAR_CLASS}])[0.4ß](?=$|[\s,;:!?\)\]\}}])"
)
Q_PHARYNGEALIZATION_TO_INTENSIVE_RE = re.compile(r"qˁ(?=[A-Za-zĀāĒēĪīŌōŪūŎŏŬŭⁿ])")
INLINE_PHARYNGEALIZATION_MARK_RE = re.compile(
    rf"(?<=[{PRACTICAL_WORD_CHAR_CLASS}])ˁ(?=[{PRACTICAL_WORD_CHAR_CLASS}])"
)

LATIN_PRACTICAL_MARKERS = frozenset("čšžɬħʕǯʒğŏŭʼⁿāēīōū£")


@dataclass(frozen=True)
class EditorialMarkupResult:
    """Lossless-enough rendering of inline TITUS editorial markup."""

    plain_text: str
    tokenized_text: str
    annotations: tuple[dict[str, str], ...]


class _EditorialMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._plain_parts: list[str] = []
        self._token_parts: list[str] = []
        self._active_kind: str | None = None
        self._active_buffer: list[str] = []
        self.annotations: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"sup", "sub"}:
            self._active_kind = tag
            self._active_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag != self._active_kind:
            return

        text = "".join(self._active_buffer)
        if text:
            self._token_parts.append(f"{{{tag}:{text}}}")
            self.annotations.append({"kind": tag, "text": text})
        self._active_kind = None
        self._active_buffer = []

    def handle_data(self, data: str) -> None:
        if not data:
            return

        self._plain_parts.append(data)
        if self._active_kind is not None:
            self._active_buffer.append(data)
        else:
            self._token_parts.append(data)

    def finish(self) -> EditorialMarkupResult:
        plain = _collapse_inline_whitespace("".join(self._plain_parts))
        tokenized = _collapse_inline_whitespace("".join(self._token_parts))
        return EditorialMarkupResult(
            plain_text=plain,
            tokenized_text=tokenized,
            annotations=tuple(self.annotations),
        )


def _collapse_inline_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_batsbi_unicode(text: str) -> str:
    """Normalize short Tsova-Tush strings for lookup and transliteration."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.translate(APOSTROPHE_NORMALIZATION)
    normalized = normalized.replace("£", "ʕ")
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    return _collapse_inline_whitespace(normalized)


def format_batsbi_display_text(text: str) -> str:
    """Remove inline morphology separators from final reader-facing Bats text."""
    normalized = normalize_batsbi_unicode(text)
    normalized = INLINE_MORPHEME_HYPHEN_RE.sub("", normalized)
    normalized = Q_PHARYNGEALIZATION_TO_INTENSIVE_RE.sub("qq", normalized)
    normalized = INLINE_PHARYNGEALIZATION_MARK_RE.sub("", normalized)
    return LEGACY_REDUCED_VOWEL_MARKER_RE.sub("", normalized)


def detect_batsbi_scheme(text: str) -> str:
    """Best-effort detector for the main Tsova-Tush source encodings."""
    sample = text or ""

    if PRIVATE_USE_RE.search(sample) or CONTROL_RE.search(sample):
        return "legacy_pdf_extraction"
    if GEORGIAN_RE.search(sample):
        return "mkhedruli_batsbi"
    if CYRILLIC_RE.search(sample):
        return "cyrillic_academic"

    normalized = normalize_batsbi_unicode(sample)
    if any(marker in normalized for marker in LATIN_PRACTICAL_MARKERS):
        return "latin_practical"
    if LATIN_RE.search(normalized):
        return "latin_practical"
    return "mixed_or_unknown"


def html_editorial_markup_to_tokens(html_text: str) -> EditorialMarkupResult:
    """Render TITUS-style inline markup without losing superscript/subscript data."""
    parser = _EditorialMarkupParser()
    parser.feed(html_text or "")
    parser.close()
    return parser.finish()


def _replace_longest(text: str, replacements: Iterable[tuple[str, str]]) -> str:
    result = text
    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        result = result.replace(source, target)
    return result


MKHEDRULI_TO_PRACTICAL = (
    ("ყყ", "qʼqʼ"),
    ("ჴჴ", "qq"),
    ("თთ", "tt"),
    ("ტტ", "tʼtʼ"),
    ("სს", "ss"),
    ("ლლ", "ll"),
    ("ხხ", "xx"),
    ("ჰʼ", "ħ"),
    ("ჰ’", "ħ"),
    ("ჰ'", "ħ"),
    ("ჼ", "ⁿ"),
    ("ჸჵ", "ʕ"),
    ("ო̆", "ŏ"),
    ("უ̆", "ŭ"),
    ("ა", "a"),
    ("ბ", "b"),
    ("ც", "c"),
    ("წ", "cʼ"),
    ("ჩ", "č"),
    ("ჭ", "čʼ"),
    ("დ", "d"),
    ("ე", "e"),
    ("გ", "g"),
    ("ღ", "ğ"),
    ("ჰ", "h"),
    ("ი", "i"),
    ("ჲ", "j"),
    ("ქ", "k"),
    ("კ", "kʼ"),
    ("ლ", "l"),
    ("ლ’", "ɬ"),
    ("ლ'", "ɬ"),
    ("მ", "m"),
    ("ნ", "n"),
    ("ო", "o"),
    ("ფ", "p"),
    ("პ", "pʼ"),
    ("ჴ", "q"),
    ("ყ", "qʼ"),
    ("რ", "r"),
    ("ს", "s"),
    ("შ", "š"),
    ("თ", "t"),
    ("ტ", "tʼ"),
    ("უ", "u"),
    ("ვ", "v"),
    ("ჳ", "w"),
    ("ხ", "x"),
    ("ზ", "z"),
    ("ჟ", "ž"),
    ("ძ", "ʒ"),
    ("ჯ", "ǯ"),
    ("ჵ", "ʕ"),
    ("ჸ", "Ɂ"),
)

CYRILLIC_TO_PRACTICAL = (
    ("тIтI", "tʼtʼ"),
    ("тІтІ", "tʼtʼ"),
    ("ккх", "qq"),
    ("ккъ", "qʼqʼ"),
    ("цI", "cʼ"),
    ("цІ", "cʼ"),
    ("чI", "čʼ"),
    ("чІ", "čʼ"),
    ("пI", "pʼ"),
    ("пІ", "pʼ"),
    ("кI", "kʼ"),
    ("кІ", "kʼ"),
    ("гI", "ğ"),
    ("гІ", "ğ"),
    ("хI", "h"),
    ("хІ", "h"),
    ("хь", "ħ"),
    ("кх", "q"),
    ("къ", "qʼ"),
    ("тI", "tʼ"),
    ("тІ", "tʼ"),
    ("лл", "ll"),
    ("сс", "ss"),
    ("хх", "xx"),
    ("дз", "ʒ"),
    ("дж", "ǯ"),
    ("(о)", "ŏ"),
    ("(у)", "ŭ"),
    ("а", "a"),
    ("б", "b"),
    ("ц", "c"),
    ("ч", "č"),
    ("д", "d"),
    ("е", "e"),
    ("э", "e"),
    ("г", "g"),
    ("и", "i"),
    ("й", "j"),
    ("к", "k"),
    ("л", "l"),
    ("м", "m"),
    ("н", "n"),
    ("о", "o"),
    ("п", "p"),
    ("р", "r"),
    ("с", "s"),
    ("ш", "š"),
    ("т", "t"),
    ("у", "u"),
    ("в", "v"),
    ("х", "x"),
    ("з", "z"),
    ("ж", "ž"),
    ("I", "Ɂ"),
    ("І", "Ɂ"),
    ("’", "Ɂ"),
    ("'", "Ɂ"),
)

PRACTICAL_TO_MKHEDRULI = (
    ("qʼqʼ", "ყყ"),
    ("tʼtʼ", "ტტ"),
    ("čʼ", "ჭ"),
    ("cʼ", "წ"),
    ("kʼ", "კ"),
    ("pʼ", "პ"),
    ("qʼ", "ყ"),
    ("tʼ", "ტ"),
    ("ɬ", "ლ’"),
    ("tt", "თთ"),
    ("ss", "სს"),
    ("ll", "ლლ"),
    ("xx", "ხხ"),
    ("qq", "ჴჴ"),
    ("ŏ", "ო̆"),
    ("ŭ", "უ̆"),
    ("č", "ჩ"),
    ("š", "შ"),
    ("ž", "ჟ"),
    ("ʒ", "ძ"),
    ("ǯ", "ჯ"),
    ("ğ", "ღ"),
    ("ħ", "ჰʼ"),
    ("ʕ", "ჵ"),
    ("Ɂ", "ჸ"),
    ("ⁿ", "ჼ"),
    ("a", "ა"),
    ("b", "ბ"),
    ("c", "ც"),
    ("d", "დ"),
    ("e", "ე"),
    ("g", "გ"),
    ("h", "ჰ"),
    ("i", "ი"),
    ("j", "ჲ"),
    ("k", "ქ"),
    ("l", "ლ"),
    ("m", "მ"),
    ("n", "ნ"),
    ("o", "ო"),
    ("p", "ფ"),
    ("q", "ჴ"),
    ("r", "რ"),
    ("s", "ს"),
    ("t", "თ"),
    ("u", "უ"),
    ("v", "ვ"),
    ("w", "ჳ"),
    ("x", "ხ"),
    ("z", "ზ"),
)


def to_canonical_practical(text: str, source_scheme: str | None = None) -> str:
    """Convert supported Tsova-Tush source encodings into practical Latin."""
    normalized = normalize_batsbi_unicode(text)
    scheme = source_scheme or detect_batsbi_scheme(normalized)

    if scheme == "latin_practical":
        return normalized
    if scheme == "mkhedruli_batsbi":
        return normalize_batsbi_unicode(
            _replace_longest(normalized, MKHEDRULI_TO_PRACTICAL)
        )
    if scheme == "cyrillic_academic":
        canonical = _replace_longest(normalized, CYRILLIC_TO_PRACTICAL)
        return normalize_batsbi_unicode(canonical)
    if scheme == "legacy_pdf_extraction":
        return normalize_batsbi_unicode(source_specific_cleanup("bertlani_pdf", normalized))
    return normalized


def from_canonical_practical(text: str, target_scheme: str) -> str:
    """Emit a supported Tsova-Tush representation from practical Latin."""
    normalized = normalize_batsbi_unicode(text)
    if target_scheme == "latin_practical":
        return normalized
    if target_scheme == "mkhedruli_batsbi":
        return _replace_longest(normalized, PRACTICAL_TO_MKHEDRULI)
    raise ValueError(f"Unsupported target scheme: {target_scheme}")


def source_specific_cleanup(source_id: str, text: str) -> str:
    """Apply narrowly scoped cleanup for source-specific extraction artifacts."""
    cleaned = text or ""

    if source_id == "bertlani_pdf":
        cleaned = cleaned.replace("\x03", " - ")
        cleaned = PRIVATE_USE_RE.sub("", cleaned)
    elif source_id == "hauk_harris_sketch":
        cleaned = cleaned.replace("\uf0e0", " -> ")
        cleaned = cleaned.replace("d� z", "d͡z")
        cleaned = cleaned.replace("d� ʒ", "d͡ʒ")
    elif source_id == "titus_html":
        return html_editorial_markup_to_tokens(cleaned).plain_text

    return normalize_batsbi_unicode(cleaned)
