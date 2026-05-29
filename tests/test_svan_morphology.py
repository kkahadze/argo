import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import dictionary_store
from src.morphology import get_morphology_analyzer
from src.translator import data, lookup, prompts


class SvanMorphologyRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._data_dir.cleanup)
        self._old_data_dir = os.environ.get("ARGO_DATA_DIR")
        os.environ["ARGO_DATA_DIR"] = self._data_dir.name
        self.addCleanup(self._restore_data_dir)

        svan_dir = Path(self._data_dir.name, "svan")
        svan_dir.mkdir()
        svan_dir.joinpath("kk.tsv").write_text(
            "word\tipa\trussian_def\tgeorgian_def\n"
            "ხოჩა\t\t\tკარგი\n"
            "ქორ\t\t\tსახლი\n",
            encoding="utf-8",
        )
        svan_dir.joinpath("sentence_pairs.tsv").write_text(
            "svan\tenglish\n", encoding="utf-8"
        )
        svan_dir.joinpath("gal.tsv").write_text("russian\tsvan\n", encoding="utf-8")
        svan_dir.joinpath("parallel_pairs.tsv").write_text(
            "low_resource\tgeorgian\tsource_id\tsource_family\tevidence_type\tdomain_type\n",
            encoding="utf-8",
        )
        svan_dir.joinpath("attested_variants.tsv").write_text(
            "query_form_raw\tquery_form_nfc\trelated_form_raw\trelated_form_nfc\t"
            "relation_type\tsource_id\tpage_start\tpage_end\tsource_line_start\t"
            "source_line_end\tconfidence\n"
            "ხოშა\tხოშა\tხოჩა\tხოჩა\tsame_as\ttopuria-kaldani:article-00003\t"
            "3\t3\t10\t10\thigh\n"
            "აჯაღ\tაჯაღ\tადაჲდ\tადაჲდ\tsame_as\ttopuria-kaldani:article-00004\t"
            "4\t4\t11\t11\thigh\n",
            encoding="utf-8",
        )
        svan_dir.joinpath("paradigm_forms.tsv").write_text(
            "form_raw\tform_nfc\tdialect\tlemma_raw\tparadigm_raw\tcolumn_raw\t"
            "person_slot\tsource_id\tpdf_pages\tprinted_pages\tconfidence\n"
            "ხუ̂აგემ\tხუ̂აგემ\tLenṭex\t'construire' ('to build')\t"
            "Series I - present subseries\tPresent\t1SG\t"
            "She 2024, PDF page 523 render (printed page 522)\t523\t522\thigh\n"
            "(ču)ləmšxēlxwäsd\t(ču)ləmšxēlxwäsd\tBal superior\t"
            "'etre noirci'\tSeries III - perfect subseries\tPluperfect\t1excl.\t"
            "She 2024, PDF pages 547-549 renders\t547-549\t546-548\tverified\n"
            "ču xwimešexolnold\tču xwimešexolnold\tLenṭex\t"
            "'etre noirci'\tSeries I - futur imparfait subseries\t"
            "Conditional imperfect\t1excl.\tShe 2024, PDF page 530 render\t"
            "530\t529\tverified\n",
            encoding="utf-8",
        )
        for filename in ("context_source.txt", "translation_overrides.tsv"):
            svan_dir.joinpath(filename).write_text("", encoding="utf-8")
        self._clear_caches()

    def tearDown(self) -> None:
        self._clear_caches()

    def _restore_data_dir(self) -> None:
        if self._old_data_dir is None:
            os.environ.pop("ARGO_DATA_DIR", None)
        else:
            os.environ["ARGO_DATA_DIR"] = self._old_data_dir

    @staticmethod
    def _clear_caches() -> None:
        dictionary_store._get_dictionary_store_cached.cache_clear()
        data._load_grammar_cached.cache_clear()
        try:
            from src.svan import morphology
        except ImportError:
            return
        morphology._get_runtime_index.cache_clear()

    def test_prompt_includes_attested_variant_evidence_and_lemma_entry(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_georgian("ხოშა")

        self.assertIn("Attested Topuria-Kaldani variant", prompt)
        self.assertIn("topuria-kaldani:article-00003", prompt)
        self.assertIn("ხოჩა", prompt)
        self.assertIn("კარგი", prompt)

    def test_prompt_includes_verified_she_paradigm_evidence(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_georgian("ხუ̂აგემ")

        self.assertIn("Verified She 2024 paradigm form", prompt)
        self.assertIn("Lenṭex", prompt)
        self.assertIn("1SG Present", prompt)

    def test_prompt_omits_variant_relation_without_lexical_evidence(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_georgian("აჯაღ")

        self.assertNotIn("topuria-kaldani:article-00004", prompt)

    def test_svan_to_georgian_prompt_includes_parenthesized_she_cell(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_georgian("(ču)ləmšxēlxwäsd")

        self.assertIn("Morphology evidence for (ču)ləmšxēlxwäsd", prompt)

    def test_svan_to_georgian_prompt_includes_multiword_she_cell(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_georgian("ču xwimešexolnold")

        self.assertIn("Morphology evidence for ču xwimešexolnold", prompt)

    def test_svan_to_english_prompt_includes_multiword_she_cell(self) -> None:
        prompt = prompts.construct_prompt_from_svan_to_english("ču xwimešexolnold")

        self.assertIn("Morphology evidence for ču xwimešexolnold", prompt)

    def test_morphology_can_be_disabled_for_control_evals(self) -> None:
        with patch.dict(os.environ, {"ARGO_ENABLE_SVAN_MORPHOLOGY": "0"}):
            prompt = prompts.construct_prompt_from_svan_to_georgian("ქორს")

        self.assertNotIn("Tuite 2023 noun suffix analysis", prompt)

    def test_svan_to_english_uses_tuite_noun_analysis_not_mingrelian_fallback(self) -> None:
        evidence = lookup.grep_search_from_svan("ქორს")

        self.assertIn("Tuite 2023 noun suffix analysis", evidence)
        self.assertIn("DAT", evidence)
        self.assertIn("ქორ", evidence)
        self.assertIn("სახლი", evidence)
        self.assertNotIn("Case-stripped fallback", evidence)

    def test_unknown_noun_suffix_analysis_is_not_returned_without_lexicon_match(
        self,
    ) -> None:
        evidence = lookup.grep_search_from_svan("უცნობის")

        self.assertNotIn("Tuite 2023 noun suffix analysis", evidence)

    def test_non_svan_packs_do_not_receive_svan_analyzer(self) -> None:
        self.assertIsNone(get_morphology_analyzer("mingrelian"))
        self.assertIsNone(get_morphology_analyzer("tsova_tush"))


if __name__ == "__main__":
    unittest.main()
