import tempfile
import unittest
from pathlib import Path

from src.tsova_tush.grammar_pair_extraction import (
    extract_grammar_translation_pairs,
)


class TsovaTushGrammarPairExtractionTests(unittest.TestCase):
    def test_extracts_standalone_batsbi_line_followed_by_quoted_translation(self) -> None:
        rows = extract_grammar_translation_pairs(
            "(1)\n"
            "eq:ar\n"
            "‘jumpʼ\n"
            "eq:-Dar\n"
            "‘make (someone) jumpʼ\n",
            source_id="holisky_gagua_1994",
        )

        self.assertEqual(
            rows,
            [
                {
                    "source_id": "holisky_gagua_1994",
                    "source_name": "holisky_gagua_1994",
                    "pair_type": "batsbi_english",
                    "batsbi_text": "eq:ar",
                    "batsbi_text_tokenized": "eq:ar",
                    "georgian_translation": "",
                    "english_translation": "jump",
                    "confidence": "high",
                    "notes": "standalone_quoted_translation",
                },
                {
                    "source_id": "holisky_gagua_1994",
                    "source_name": "holisky_gagua_1994",
                    "pair_type": "batsbi_english",
                    "batsbi_text": "eq:-Dar",
                    "batsbi_text_tokenized": "eq:-Dar",
                    "georgian_translation": "",
                    "english_translation": "make (someone) jump",
                    "confidence": "high",
                    "notes": "standalone_quoted_translation",
                },
            ],
        )

    def test_extracts_numbered_interlinear_example_with_gloss_line(self) -> None:
        rows = extract_grammar_translation_pairs(
            "(93) (a) [ču£ co Ja-š] da™onbadJailn4 Ja-s4\n"
            "home not is-PRES/ABSOL sad is-1SG\n"
            "‘When Iʼm not at home, Iʼm sad.ʼ\n",
            source_id="holisky_gagua_1994",
        )

        self.assertEqual(
            rows,
            [
                {
                    "source_id": "holisky_gagua_1994",
                    "source_name": "holisky_gagua_1994",
                    "pair_type": "batsbi_english",
                    "batsbi_text": "[ču£ co Ja-š] da™onbadJailn4 Ja-s4",
                    "batsbi_text_tokenized": "[ču£ co Ja-š] da™onbadJailn4 Ja-s4",
                    "georgian_translation": "",
                    "english_translation": "When Iʼm not at home, Iʼm sad.",
                    "confidence": "high",
                    "notes": "interlinear_quoted_translation",
                }
            ],
        )

    def test_extracts_marker_only_interlinear_block(self) -> None:
        rows = extract_grammar_translation_pairs(
            "(97) (b)\n"
            "vun-e saubß Vil-0, oqu-mplen muiš0 Va-s4\n"
            "what-REL more laugh-PRES 3SG-as/much badly is-1SG\n"
            "‘The more he laughs, the worse I feel.ʼ\n",
            source_id="holisky_gagua_1994",
        )

        self.assertEqual(rows[0]["batsbi_text"], "vun-e saubß Vil-0, oqu-mplen muiš0 Va-s4")
        self.assertEqual(rows[0]["english_translation"], "The more he laughs, the worse I feel.")
        self.assertEqual(rows[0]["notes"], "interlinear_quoted_translation")

    def test_extracts_explicit_equals_translation_after_interlinear_gloss(self) -> None:
        rows = extract_grammar_translation_pairs(
            "(62) (a) Jel-i-£.\n"
            "laugh-PRES-COND\n"
            "= ‘if she is laughingʼ\n",
            source_id="holisky_gagua_1994",
        )

        self.assertEqual(
            rows,
            [
                {
                    "source_id": "holisky_gagua_1994",
                    "source_name": "holisky_gagua_1994",
                    "pair_type": "batsbi_english",
                    "batsbi_text": "Jel-i-£.",
                    "batsbi_text_tokenized": "Jel-i-£.",
                    "georgian_translation": "",
                    "english_translation": "if she is laughing",
                    "confidence": "high",
                    "notes": "explicit_equals_translation",
                }
            ],
        )

    def test_rejects_inline_prose_quotes_without_pair_structure(self) -> None:
        rows = extract_grammar_translation_pairs(
            "The suffix £e means ‘ifʼ in some examples.\n"
            "In prose, quoted English should not create rows.\n",
            source_id="holisky_gagua_1994",
        )

        self.assertEqual(rows, [])

    def test_accepts_path_input_and_source_name_alias_for_mining_integration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            grammar_path = Path(temp_dir) / "holisky_gagua_1994.txt"
            grammar_path.write_text("eq:ar\n‘jumpʼ\n", encoding="utf-8")

            rows = extract_grammar_translation_pairs(
                grammar_path,
                source_name="holisky_gagua_1994",
            )

        self.assertEqual(rows[0]["source_id"], "holisky_gagua_1994")
        self.assertEqual(rows[0]["source_name"], "holisky_gagua_1994")
        self.assertEqual(rows[0]["pair_type"], "batsbi_english")
        self.assertEqual(rows[0]["english_translation"], "jump")

    def test_missing_path_input_returns_no_candidates(self) -> None:
        rows = extract_grammar_translation_pairs(
            Path("/tmp/tsova-tush-grammar-pairs-do-not-exist.txt"),
            source_name="holisky_gagua_1994",
        )

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
