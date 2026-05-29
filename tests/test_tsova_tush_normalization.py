import unittest

from src.tsova_tush.normalization import (
    detect_batsbi_scheme,
    format_batsbi_display_text,
    from_canonical_practical,
    html_editorial_markup_to_tokens,
    normalize_batsbi_unicode,
    source_specific_cleanup,
    to_canonical_practical,
)


class TsovaTushNormalizationTests(unittest.TestCase):
    def test_detects_core_source_schemes(self) -> None:
        self.assertEqual(detect_batsbi_scheme("ჴა"), "mkhedruli_batsbi")
        self.assertEqual(detect_batsbi_scheme("kwēqʼaⁿ"), "latin_practical")
        self.assertEqual(detect_batsbi_scheme("кве̄къаᴴ"), "cyrillic_academic")
        self.assertEqual(detect_batsbi_scheme("\x03broken"), "legacy_pdf_extraction")

    def test_normalizes_unicode_whitespace_and_apostrophes(self) -> None:
        self.assertEqual(
            normalize_batsbi_unicode("  k\u2019a\t  e\u0301  "),
            "kʼa é",
        )
        self.assertEqual(
            normalize_batsbi_unicode("mo£"),
            "moʕ",
        )
        self.assertEqual(
            normalize_batsbi_unicode("moħ"),
            "moħ",
        )

    def test_formats_reader_facing_bats_output_without_inline_morpheme_hyphens(self) -> None:
        self.assertEqual(
            format_batsbi_display_text("moʕ ħo v-a-ħ?"),
            "moʕ ħo vaħ?",
        )
        self.assertEqual(
            format_batsbi_display_text("moʕ - ħo"),
            "moʕ - ħo",
        )
        self.assertEqual(
            format_batsbi_display_text("daq̄ōⁿ kʼuitʼ0"),
            "daq̄ōⁿ kʼuitʼ",
        )
        self.assertEqual(
            format_batsbi_display_text("daqˁoⁿ kʼuitʼ"),
            "daqqoⁿ kʼuitʼ",
        )

    def test_preserves_titus_editorial_markup_as_tokens(self) -> None:
        result = html_editorial_markup_to_tokens(
            "<span>ღაზე<sup>ნ</sup> <sub>7</sub></span>"
        )

        self.assertEqual(result.plain_text, "ღაზენ 7")
        self.assertEqual(result.tokenized_text, "ღაზე{sup:ნ} {sub:7}")
        self.assertEqual(
            result.annotations,
            (
                {"kind": "sup", "text": "ნ"},
                {"kind": "sub", "text": "7"},
            ),
        )

    def test_transliterates_representative_mkhedruli_and_cyrillic_forms(self) -> None:
        self.assertEqual(
            to_canonical_practical("ჴა", "mkhedruli_batsbi"),
            "qa",
        )
        self.assertEqual(
            to_canonical_practical("ღაზეჼ", "mkhedruli_batsbi"),
            "ğazeⁿ",
        )
        self.assertEqual(
            to_canonical_practical("кIа", "cyrillic_academic"),
            "kʼa",
        )
        self.assertEqual(
            from_canonical_practical("qa", "mkhedruli_batsbi"),
            "ჴა",
        )
        self.assertEqual(
            from_canonical_practical("ğazeⁿ naqʼbistʼ", "mkhedruli_batsbi"),
            "ღაზეჼ ნაყბისტ",
        )
        self.assertEqual(
            from_canonical_practical("daqqoⁿ kʼuitʼ", "mkhedruli_batsbi"),
            "დაჴჴოჼ კუიტ",
        )

    def test_applies_source_specific_cleanup(self) -> None:
        self.assertEqual(
            source_specific_cleanup("bertlani_pdf", "ა\x03ბ"),
            "ა - ბ",
        )
        self.assertEqual(
            source_specific_cleanup("hauk_harris_sketch", "ა\uf0e0ბ"),
            "ა -> ბ",
        )
        self.assertEqual(
            source_specific_cleanup("hauk_harris_sketch", "ʒ = d� z, ǯ = d� ʒ"),
            "ʒ = d͡z, ǯ = d͡ʒ",
        )


if __name__ == "__main__":
    unittest.main()
