import unittest

from scripts.extract_svan_tuite_grammar import _extraction_metrics


class SvanTuiteGrammarExtractionTests(unittest.TestCase):
    def test_accepts_layout_extraction_from_2023_source(self) -> None:
        text = (
            "The Svan language.\n"
            "Svan (Tuite) - page 2 - 23 August 2023\n"
            "š č ə ɣ ǯ\n"
            + ("\f" * 97)
        )

        metrics = _extraction_metrics(text)

        self.assertEqual(metrics["pages"], 97)
        self.assertEqual(metrics["legacy_pound_sign_count"], 0)
        self.assertEqual(metrics["replacement_character_count"], 0)

    def test_rejects_legacy_or_damaged_extraction(self) -> None:
        text = (
            "The Svan language.\n"
            "Svan (Tuite) - page 2 - 23 August 2023\n"
            "£wan \ufffd\n"
            + ("\f" * 97)
        )

        with self.assertRaises(ValueError):
            _extraction_metrics(text)


if __name__ == "__main__":
    unittest.main()
