import unittest
from pathlib import Path

import src.single_call_translator as legacy_translator
from src.translator import extraction, lookup, pipeline, prompts


class TranslatorModuleImportSmokeTest(unittest.TestCase):
    def test_legacy_facade_reexports_main_surface(self):
        expected_callables = [
            "translate",
            "extract_translation",
            "collect_exact_match_candidates",
            "grep_search_from_mingrelian",
            "construct_prompt_from_mingrelian_to_english",
        ]

        for name in expected_callables:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(legacy_translator, name)))

        self.assertIs(legacy_translator.extract_translation, extraction.extract_translation)
        self.assertIs(legacy_translator.PROMPT_BUILDERS, prompts.PROMPT_BUILDERS)

    def test_prompt_builder_routing_imports(self):
        self.assertIn(("mingrelian", "english"), prompts.PROMPT_BUILDERS)
        self.assertIn(("english", "mingrelian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("mingrelian", "georgian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("georgian", "mingrelian"), prompts.PROMPT_BUILDERS)

    def test_legacy_data_path_resolution_still_finds_repo_data(self):
        sentence_pairs_path = Path(legacy_translator._get_data_path("sentence_pairs.tsv"))

        self.assertTrue(sentence_pairs_path.exists())
        self.assertEqual(sentence_pairs_path.name, "sentence_pairs.tsv")

    def test_extract_translation_marker_path(self):
        response = "Notes\n<<<TRANSLATION>>>\nTranslation: test answer\n<<<END_TRANSLATION>>>"

        self.assertEqual(legacy_translator.extract_translation(response), "test answer")

    def test_legacy_google_translator_monkeypatch_sync(self):
        original = legacy_translator.GoogleTranslator
        try:
            legacy_translator.GoogleTranslator = None
            legacy_translator._sync_compat_state()

            self.assertIsNone(lookup.GoogleTranslator)
            self.assertIsNone(pipeline.GoogleTranslator)
        finally:
            legacy_translator.GoogleTranslator = original
            legacy_translator._sync_compat_state()


if __name__ == "__main__":
    unittest.main()
