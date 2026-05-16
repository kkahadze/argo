import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import single_call_translator as translator


class GrammarPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        (self.data_dir / "harris.txt").write_text(
            "FULL GRAMMAR SOURCE\n" * 20,
            encoding="utf-8",
        )
        (self.data_dir / "harris_compact.txt").write_text(
            "COMPACT GRAMMAR SOURCE",
            encoding="utf-8",
        )
        (self.data_dir / "sentence_pairs.tsv").write_text(
            "foo\tbar\n",
            encoding="utf-8",
        )
        (self.data_dir / "gal.tsv").write_text("", encoding="utf-8")
        (self.data_dir / "kk.tsv").write_text("", encoding="utf-8")
        (self.data_dir / "master-lexicon-mkhedruli.csv").write_text(
            "headword,headword_raw,translation\n",
            encoding="utf-8",
        )
        (self.data_dir / "context_source.txt").write_text("", encoding="utf-8")

        self.path_patch = mock.patch.object(
            translator,
            "_get_data_path",
            side_effect=lambda filename: str(self.data_dir / filename),
        )
        self.path_patch.start()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(translator._sync_compat_state)
        self.addCleanup(self.path_patch.stop)

        self._clear_caches()
        self.addCleanup(self._clear_caches)

    def _clear_caches(self) -> None:
        translator._load_master_lexicon_rows_cached.cache_clear()
        translator._load_sentence_pairs_rows_cached.cache_clear()
        translator._load_gal_rows_cached.cache_clear()
        translator._load_kk_rows_cached.cache_clear()
        translator._load_context_source_entries_cached.cache_clear()
        translator._load_grammar_cached.cache_clear()
        translator._compiled_word_pattern.cache_clear()

    def test_compact_policy_uses_compact_grammar(self) -> None:
        full_prompt = translator.construct_prompt_from_mingrelian_to_english(
            "foo",
            grammar_policy="full",
        )
        compact_prompt = translator.construct_prompt_from_mingrelian_to_english(
            "foo",
            grammar_policy="compact",
        )
        no_grammar_prompt = translator.construct_prompt_from_mingrelian_to_english(
            "foo",
            grammar_policy="none",
        )

        self.assertIn("FULL GRAMMAR SOURCE", full_prompt)
        self.assertNotIn("COMPACT GRAMMAR SOURCE", full_prompt)

        self.assertIn("COMPACT GRAMMAR SOURCE", compact_prompt)
        self.assertNotIn("FULL GRAMMAR SOURCE", compact_prompt)
        self.assertLess(len(compact_prompt), len(full_prompt))

        self.assertNotIn("Here is the Mingrelian grammar information:", no_grammar_prompt)
        self.assertEqual(
            translator._measure_prompt_sections(no_grammar_prompt)["grammar_chars"],
            0,
        )
        self.assertEqual(
            translator._measure_prompt_sections(no_grammar_prompt)["dict_entries_chars"],
            translator._measure_prompt_sections(compact_prompt)["dict_entries_chars"],
        )

    def test_env_policy_is_used_when_no_explicit_policy_is_passed(self) -> None:
        with mock.patch.dict(os.environ, {"ARGO_GRAMMAR_POLICY": "compact"}):
            prompt = translator.construct_prompt_from_mingrelian_to_english("foo")

        self.assertIn("COMPACT GRAMMAR SOURCE", prompt)


if __name__ == "__main__":
    unittest.main()
