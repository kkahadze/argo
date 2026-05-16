import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import single_call_translator as translator


class DictionaryLoaderHeaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        (self.data_dir / "gal.tsv").write_text(
            "Russian\tMingrelian\n"
            "Здравствуйте\tგომორძგუა\n",
            encoding="utf-8",
        )
        (self.data_dir / "kk.tsv").write_text(
            "word\tipa\trussian_def\tgeorgian_def\n"
            "აბა!\taba\tну-ка! да! ну!\tაბა!\n",
            encoding="utf-8",
        )
        (self.data_dir / "sentence_pairs.tsv").write_text("", encoding="utf-8")
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
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

        self._clear_loader_caches()
        self.addCleanup(self._clear_loader_caches)

    def _clear_loader_caches(self) -> None:
        translator._load_master_lexicon_rows_cached.cache_clear()
        translator._load_sentence_pairs_rows_cached.cache_clear()
        translator._load_gal_rows_cached.cache_clear()
        translator._load_kk_rows_cached.cache_clear()
        translator._load_context_source_entries_cached.cache_clear()
        translator._compiled_word_pattern.cache_clear()

    def test_gal_loader_excludes_header_from_rows_and_lookups(self) -> None:
        self.assertEqual(
            translator._load_gal_rows(),
            (("Здравствуйте", "გომორძგუა"),),
        )

        self.assertIsNone(
            translator.check_exact_match_simple("Russian", "russian", "mingrelian")
        )
        self.assertIsNone(translator.find_mingrelian_in_dicts("Mingrelian"))
        self.assertEqual(
            translator.check_exact_match_simple("Здравствуйте", "russian", "mingrelian"),
            "გომორძგუა",
        )

        output, has_standalone = translator.grep_search_gal("Russian")
        self.assertEqual(output, "")
        self.assertFalse(has_standalone)

    def test_kk_loader_excludes_header_from_rows_and_lookups(self) -> None:
        self.assertEqual(
            translator._load_kk_rows(),
            (("აბა!", "aba", "ну-ка! да! ну!", "აბა!"),),
        )

        self.assertIsNone(
            translator.check_exact_match_simple("word", "mingrelian", "georgian")
        )
        self.assertIsNone(translator.find_mingrelian_in_dicts("word"))
        self.assertEqual(
            translator.collect_exact_match_candidates("word", "mingrelian", "georgian"),
            [],
        )
        self.assertEqual(
            translator.check_exact_match_simple("აბა!", "mingrelian", "georgian"),
            "აბა!",
        )

        output, has_standalone = translator.grep_search_kk("word")
        self.assertEqual(output, "")
        self.assertFalse(has_standalone)


if __name__ == "__main__":
    unittest.main()
