import unittest

from src.language_packs import (
    SUPPORTED_TRANSLATION_PAIRS,
    get_language_pack,
    is_supported_translation_pair,
)


class LanguagePackTests(unittest.TestCase):
    def test_unified_registry_contains_all_low_resource_languages(self) -> None:
        mingrelian = get_language_pack("mingrelian")
        bats = get_language_pack("tsova_tush")
        svan = get_language_pack("svan")

        self.assertEqual(mingrelian.code, "mingrelian")
        self.assertEqual(mingrelian.display_name, "Mingrelian")
        self.assertEqual(bats.code, "tsova_tush")
        self.assertEqual(bats.display_name, "Bats")
        self.assertEqual(svan.code, "svan")
        self.assertEqual(svan.display_name, "Svan")

    def test_supported_pair_matrix_includes_existing_products_but_not_cross_low_resource(self) -> None:
        expected_pairs = {
            ("mingrelian", "english"),
            ("english", "mingrelian"),
            ("mingrelian", "georgian"),
            ("georgian", "mingrelian"),
            ("tsova_tush", "english"),
            ("english", "tsova_tush"),
            ("tsova_tush", "georgian"),
            ("georgian", "tsova_tush"),
            ("svan", "english"),
            ("english", "svan"),
            ("svan", "georgian"),
            ("georgian", "svan"),
            ("english", "georgian"),
            ("georgian", "english"),
        }

        self.assertEqual(SUPPORTED_TRANSLATION_PAIRS, expected_pairs)
        self.assertTrue(is_supported_translation_pair("english", "tsova_tush"))
        self.assertTrue(is_supported_translation_pair("english", "svan"))
        self.assertFalse(is_supported_translation_pair("mingrelian", "tsova_tush"))
        self.assertFalse(is_supported_translation_pair("tsova_tush", "svan"))

    def test_bats_pack_normalizes_reader_facing_output(self) -> None:
        bats = get_language_pack("tsova_tush")

        self.assertEqual(
            bats.normalize_output("daqˁoⁿ kʼuitʼ0"),
            "daqqoⁿ kʼuitʼ",
        )
        self.assertEqual(
            bats.canonicalize_lookup_target("დაჴჴოჼ კუიტ"),
            "daqqoⁿ kʼuitʼ",
        )


if __name__ == "__main__":
    unittest.main()
