import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.single_call_translator as legacy_translator
from src.translator import extraction, lookup, pipeline, prompts


class TranslatorModuleImportSmokeTest(unittest.TestCase):
    def tearDown(self):
        legacy_translator._sync_compat_state()

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
        self.assertIs(
            legacy_translator.PROMPT_BUILDERS[("english", "mingrelian")],
            legacy_translator.construct_prompt_from_english_to_mingrelian,
        )

    def test_prompt_builder_routing_imports(self):
        self.assertIn(("mingrelian", "english"), prompts.PROMPT_BUILDERS)
        self.assertIn(("english", "mingrelian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("mingrelian", "georgian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("georgian", "mingrelian"), prompts.PROMPT_BUILDERS)

        self.assertIn(("mingrelian", "english"), legacy_translator.PROMPT_BUILDERS)
        self.assertIn(("english", "mingrelian"), legacy_translator.PROMPT_BUILDERS)
        self.assertIn(("mingrelian", "georgian"), legacy_translator.PROMPT_BUILDERS)
        self.assertIn(("georgian", "mingrelian"), legacy_translator.PROMPT_BUILDERS)

    def test_legacy_data_path_resolution_still_finds_repo_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "sentence_pairs.tsv").write_text(
                "Mingrelian\tEnglish\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"ARGO_DATA_DIR": str(data_dir)}, clear=False):
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

    def test_legacy_prompt_builders_sync_google_translator_monkeypatch(self):
        class CountingGoogleTranslator:
            calls = 0

            def __init__(self, *args, **kwargs):
                type(self).calls += 1

            def translate(self, text):
                return "translated"

        original_facade_translator = legacy_translator.GoogleTranslator
        original_lookup_translator = lookup.GoogleTranslator
        original_pipeline_translator = pipeline.GoogleTranslator
        empty_data_patches = [
            patch.object(legacy_translator, "_load_sentence_pairs_rows", return_value=[]),
            patch.object(legacy_translator, "_load_gal_rows", return_value=[]),
            patch.object(legacy_translator, "_load_kk_rows", return_value=[]),
            patch.object(legacy_translator, "_load_context_source_entries", return_value=[]),
            patch.object(legacy_translator, "_load_master_lexicon_rows", return_value=[]),
            patch.object(legacy_translator, "_load_grammar", return_value=""),
        ]

        try:
            for data_patch in empty_data_patches:
                data_patch.start()

            for builder, sentence in (
                (legacy_translator.construct_prompt_from_english_to_mingrelian, "the"),
                (legacy_translator.construct_prompt_from_georgian_to_mingrelian, "სატესტო"),
            ):
                with self.subTest(builder=builder.__name__):
                    CountingGoogleTranslator.calls = 0
                    lookup.GoogleTranslator = CountingGoogleTranslator
                    pipeline.GoogleTranslator = CountingGoogleTranslator
                    legacy_translator.GoogleTranslator = None

                    prompt = builder(sentence)

                    self.assertIn(sentence, prompt)
                    self.assertEqual(CountingGoogleTranslator.calls, 0)
                    self.assertIsNone(lookup.GoogleTranslator)
                    self.assertIsNone(pipeline.GoogleTranslator)

            CountingGoogleTranslator.calls = 0
            lookup.GoogleTranslator = CountingGoogleTranslator
            pipeline.GoogleTranslator = CountingGoogleTranslator
            legacy_translator.GoogleTranslator = None

            map_prompt = legacy_translator.PROMPT_BUILDERS[("english", "mingrelian")]("the")

            self.assertIn("the", map_prompt)
            self.assertEqual(CountingGoogleTranslator.calls, 0)
        finally:
            for data_patch in reversed(empty_data_patches):
                data_patch.stop()
            legacy_translator.GoogleTranslator = original_facade_translator
            lookup.GoogleTranslator = original_lookup_translator
            pipeline.GoogleTranslator = original_pipeline_translator

    def test_legacy_prompt_builders_observe_facade_lookup_monkeypatch(self):
        lookup_calls = []

        def patched_lookup(word):
            lookup_calls.append(word)
            return f"\npatched facade dictionary entry for {word}\n"

        with (
            patch.object(legacy_translator, "grep_search_from_mingrelian", side_effect=patched_lookup),
            patch.object(legacy_translator, "_load_grammar", return_value=""),
        ):
            prompt = legacy_translator.construct_prompt_from_mingrelian_to_english("patchedword")

        self.assertEqual(lookup_calls, ["patchedword"])
        self.assertIn("patched facade dictionary entry for patchedword", prompt)

    def test_legacy_prompt_builder_map_observes_facade_lookup_monkeypatch(self):
        def patched_lookup(word):
            return f"\nfacade map dictionary entry for {word}\n"

        with (
            patch.object(legacy_translator, "grep_search_from_mingrelian", side_effect=patched_lookup),
            patch.object(legacy_translator, "_load_grammar", return_value=""),
        ):
            prompt = legacy_translator.PROMPT_BUILDERS[("mingrelian", "english")]("mapword")

        self.assertIn("facade map dictionary entry for mapword", prompt)

    def test_legacy_translate_observes_facade_exact_match_monkeypatch(self):
        class FailingClient:
            provider = "test"
            model = "test"

            def complete(self, prompt):
                raise AssertionError("translate should return before the LLM call")

        def patched_candidates(input_text, source_lang, target_lang):
            self.assertEqual((input_text, source_lang, target_lang), ("hello", "english", "mingrelian"))
            return [{"source_name": "test", "target_text": "patched exact"}]

        with patch.object(legacy_translator, "collect_exact_match_candidates", side_effect=patched_candidates):
            result = legacy_translator.translate("hello", "english", "mingrelian", FailingClient())

        self.assertEqual(result["translation"], "patched exact")
        self.assertEqual(result["response_source"], "exact_lexicon")

    def test_legacy_translate_observes_facade_google_exact_monkeypatch(self):
        class FailingClient:
            provider = "test"
            model = "test"

            def complete(self, prompt):
                raise AssertionError("translate should return before the LLM call")

        with (
            patch.object(legacy_translator, "GoogleTranslator", None),
            patch.object(legacy_translator, "collect_exact_match_candidates", return_value=[]),
            patch.object(
                legacy_translator,
                "check_exact_match_with_google_translate",
                return_value="patched bridge exact",
            ),
        ):
            result = legacy_translator.translate("hello", "english", "mingrelian", FailingClient())

        self.assertEqual(result["translation"], "patched bridge exact")
        self.assertEqual(result["response_source"], "dictionary_google_bridge")

    def test_legacy_translate_observes_facade_prompt_builder_map_monkeypatch(self):
        class RecordingClient:
            provider = "test"
            model = "test"

            def __init__(self):
                self.prompt = None

            def complete(self, prompt):
                self.prompt = prompt
                return "notes\n<<<TRANSLATION>>>\npatched final\n<<<END_TRANSLATION>>>"

        client = RecordingClient()
        prompt_builders = {
            ("mingrelian", "english"): lambda sentence, **kwargs: f"patched prompt for {sentence}"
        }

        with (
            patch.object(legacy_translator, "GoogleTranslator", None),
            patch.object(legacy_translator, "collect_exact_match_candidates", return_value=[]),
            patch.object(legacy_translator, "check_exact_match_with_google_translate", return_value=None),
            patch.object(legacy_translator, "PROMPT_BUILDERS", prompt_builders),
        ):
            result = legacy_translator.translate("word", "mingrelian", "english", client)

        self.assertEqual(client.prompt, "patched prompt for word")
        self.assertEqual(result["translation"], "patched final")


if __name__ == "__main__":
    unittest.main()
