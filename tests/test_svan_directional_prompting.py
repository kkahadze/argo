import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import dictionary_store
from src.translator import data, pipeline, prompts


class SvanDirectionalPromptingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        svan = root / "svan"
        svan.mkdir()
        long_gloss = " ".join(["განმარტება"] * 1800)
        (svan / "kk.tsv").write_text(
            "word\tipa\trussian_def\tgeorgian_def\n"
            "მხოლოდსვანური\t\t\tსხვა მნიშვნელობა\n"
            "სხვა\t\t\tჩამოტეხილ მხოლოდ განმარტებაშია\n"
            "გემ\t\t\tგემო\n"
            f"ხმაურიერთი\t\t\t{long_gloss}\n"
            f"ხმაურიორი\t\t\t{long_gloss}\n"
            f"ხმაურისამი\t\t\t{long_gloss}\n",
            encoding="utf-8",
        )
        (svan / "sentence_pairs.tsv").write_text("svan\tenglish\n", encoding="utf-8")
        (svan / "gal.tsv").write_text("russian\tsvan\n", encoding="utf-8")
        (svan / "translation_overrides.tsv").write_text(
            "source_language\ttarget_language\tsource\ttranslation\n", encoding="utf-8"
        )
        (svan / "master-lexicon-mkhedruli.csv").write_text(
            "transcription,translation\n", encoding="utf-8"
        )
        (svan / "context_source.txt").write_text(
            "===== SOURCE: one =====\n"
            "Svan: წყარო აქ\n"
            "Georgian: სხვა ტექსტი\n\n"
            "===== SOURCE: two =====\n"
            "Svan: სხვა ტექსტი\n"
            "Georgian: წყარო აქ\n",
            encoding="utf-8",
        )
        (svan / "tuite.txt").write_text("SOME HUGE SVAN GRAMMAR", encoding="utf-8")
        (svan / "tuite_compact.txt").write_text("SOME HUGE SVAN GRAMMAR", encoding="utf-8")
        self.env_patch = patch.dict(os.environ, {"ARGO_DATA_DIR": str(root)}, clear=False)
        self.env_patch.start()
        self._clear_caches()

    def tearDown(self):
        self.env_patch.stop()
        self._clear_caches()
        self.tempdir.cleanup()

    @staticmethod
    def _clear_caches():
        dictionary_store._get_dictionary_store_cached.cache_clear()
        dictionary_store._compiled_word_pattern.cache_clear()
        data._load_grammar_cached.cache_clear()

    def test_svan_source_does_not_retrieve_georgian_definition_matches(self):
        prompt = prompts.construct_prompt_from_svan_to_georgian("ჩამოტეხილ")
        self.assertNotIn("Svan: სხვა", prompt)

    def test_svan_context_matching_only_uses_source_lines(self):
        prompt = prompts.construct_prompt_from_svan_to_georgian("წყარო")
        self.assertIn("SOURCE: one", prompt)
        self.assertNotIn("SOURCE: two", prompt)

    def test_svan_to_georgian_omits_grammar_and_bounds_dictionary_context(self):
        prompt = prompts.construct_prompt_from_svan_to_georgian(
            "ხმაურიერთი ხმაურიორი ხმაურისამი"
        )
        self.assertNotIn("SOME HUGE SVAN GRAMMAR", prompt)
        dict_entries = prompt.split(
            "Here are some various dictionary entries for word(s) in that phrase:\n\n",
            1,
        )[1].split("\n\nNow remember,", 1)[0]
        self.assertLessEqual(len(dict_entries), 8000)

    def test_svan_pipeline_defaults_to_no_grammar_payload(self):
        class RecordingClient:
            provider = "test"
            model = "test"
            prompt = ""

            def complete(self, prompt):
                self.prompt = prompt
                return "<<<TRANSLATION>>>\nთარგმანი\n<<<END_TRANSLATION>>>"

        client = RecordingClient()
        with (
            patch.object(pipeline, "collect_exact_match_candidates", return_value=[]),
            patch.object(pipeline, "check_exact_match_with_google_translate", return_value=None),
        ):
            pipeline.translate("გემ უცნობი", "svan", "georgian", client)
        self.assertNotIn("SOME HUGE SVAN GRAMMAR", client.prompt)

    def test_georgian_to_svan_keeps_existing_full_grammar_and_retrieval_path(self):
        prompt = prompts.construct_prompt_from_georgian_to_svan("მხოლოდსვანური")
        self.assertIn("SOME HUGE SVAN GRAMMAR", prompt)
        self.assertIn("Svan: მხოლოდსვანური", prompt)

    def test_svan_to_english_keeps_existing_full_grammar_default(self):
        prompt = prompts.construct_prompt_from_svan_to_english("გემ")
        self.assertIn("SOME HUGE SVAN GRAMMAR", prompt)


if __name__ == "__main__":
    unittest.main()
