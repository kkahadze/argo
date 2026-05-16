import unittest

from src.dictionary_store import get_dictionary_store
from src.single_call_translator import (
    check_exact_match_simple,
    grep_search_gal,
    grep_search_kk,
)


class DictionaryStoreTests(unittest.TestCase):
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
