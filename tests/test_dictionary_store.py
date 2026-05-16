import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import dictionary_store
from src.dictionary_store import get_dictionary_store
from src.single_call_translator import (
    check_exact_match_simple,
    grep_search_gal,
    grep_search_kk,
)


class DictionaryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self._write_test_data()

        self.env_patch = patch.dict(os.environ, {"ARGO_DATA_DIR": str(self.data_dir)}, clear=False)
        self.env_patch.start()
        self._clear_caches()

        self.addCleanup(self._clear_caches)
        self.addCleanup(self.env_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def _write_test_data(self):
        (self.data_dir / "sentence_pairs.tsv").write_text(
            "Mingrelian\tEnglish\n"
            "გომორძგუა\tHello\n",
            encoding="utf-8",
        )
        (self.data_dir / "gal.tsv").write_text(
            "Russian\tMingrelian\n"
            "Я\tმა\n",
            encoding="utf-8",
        )
        (self.data_dir / "kk.tsv").write_text(
            "word\tipa\trussian_def\tgeorgian_def\n"
            "აბაზი\tabazi\tдвугривенный, двадцать копеек\tაბაზი, ოცი კაპიკი\n",
            encoding="utf-8",
        )
        (self.data_dir / "kajaia_cleaned.txt").write_text("", encoding="utf-8")
        (self.data_dir / "harris.txt").write_text("", encoding="utf-8")

    def _clear_caches(self):
        dictionary_store._get_dictionary_store_cached.cache_clear()
        dictionary_store._compiled_word_pattern.cache_clear()

    def test_store_is_cached_and_skips_tsv_headers(self):
        first_store = get_dictionary_store()
        second_store = get_dictionary_store()

        self.assertIs(first_store, second_store)
        self.assertEqual(first_store.gal_entries[0].russian, "Я")
        self.assertNotEqual(first_store.kk_entries[0].mingrelian, "word")
        self.assertIsNone(check_exact_match_simple("Russian", "russian", "mingrelian"))

    def test_exact_lookup_uses_indexed_rows(self):
        self.assertEqual(check_exact_match_simple("Я", "russian", "mingrelian"), "მა")
        self.assertEqual(
            check_exact_match_simple("აბაზი", "mingrelian", "georgian"),
            "აბაზი, ოცი კაპიკი",
        )

    def test_grep_search_contract_is_preserved(self):
        gal_output, gal_has_standalone = grep_search_gal("მა")
        self.assertTrue(gal_has_standalone)
        self.assertIn("Mingrelian: მა", gal_output)
        self.assertIn("Russian: Я", gal_output)

        kk_output, kk_has_standalone = grep_search_kk("აბაზი")
        self.assertTrue(kk_has_standalone)
        self.assertIn("Mingrelian: აბაზი", kk_output)
        self.assertIn("Russian primary meaning: двугривенный, двадцать копеек", kk_output)


if __name__ == "__main__":
    unittest.main()
