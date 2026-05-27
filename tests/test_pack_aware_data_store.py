import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import dictionary_store
from src.dictionary_store import get_dictionary_store
from src.translator import data


class PackAwareDataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.bats_dir = self.data_root / "tsova_tush"
        self.bats_dir.mkdir()

        self._write_pack_data(
            self.data_root,
            low_header="Mingrelian",
            low_word="გომორძგუა",
            english="hello",
            russian="привет",
            georgian="გამარჯობა",
            grammar="mingrelian grammar",
            compact_grammar="mingrelian compact grammar",
            context="mingrelian context",
            source_language="mingrelian",
        )
        self._write_pack_data(
            self.bats_dir,
            low_header="tsova_tush",
            low_word="eq:ar",
            english="jump",
            russian="прыгать",
            georgian="ხტომა",
            grammar="bats grammar",
            compact_grammar="bats compact grammar",
            context="bats context",
            source_language="tsova_tush",
        )

        self.env_patch = patch.dict(os.environ, {"ARGO_DATA_DIR": str(self.data_root)}, clear=False)
        self.env_patch.start()
        self._clear_caches()

        self.addCleanup(self._clear_caches)
        self.addCleanup(self.env_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def _write_pack_data(
        self,
        pack_dir: Path,
        *,
        low_header: str,
        low_word: str,
        english: str,
        russian: str,
        georgian: str,
        grammar: str,
        compact_grammar: str,
        context: str,
        source_language: str,
    ) -> None:
        pack_dir.mkdir(exist_ok=True)
        pack_dir.joinpath("master-lexicon-mkhedruli.csv").write_text(
            f"headword,headword_raw,translation\n{low_word},{low_word},{english}\n",
            encoding="utf-8",
        )
        pack_dir.joinpath("sentence_pairs.tsv").write_text(
            f"{low_header}\tEnglish\n{low_word}\t{english}\n",
            encoding="utf-8",
        )
        pack_dir.joinpath("gal.tsv").write_text(
            f"russian\t{low_header}\n{russian}\t{low_word}\n",
            encoding="utf-8",
        )
        pack_dir.joinpath("kk.tsv").write_text(
            f"word\tipa\trussian_def\tgeorgian_def\n{low_word}\t\t{russian}\t{georgian}\n",
            encoding="utf-8",
        )
        pack_dir.joinpath("translation_overrides.tsv").write_text(
            "source_language\ttarget_language\tsource_text\ttarget_text\n"
            f"english\t{source_language}\t{english}\t{low_word}\n",
            encoding="utf-8",
        )
        pack_dir.joinpath("context_source.txt").write_text(context, encoding="utf-8")
        pack_dir.joinpath("harris.txt").write_text(grammar, encoding="utf-8")
        pack_dir.joinpath("harris_compact.txt").write_text(compact_grammar, encoding="utf-8")

    def _clear_caches(self) -> None:
        data._load_master_lexicon_rows_cached.cache_clear()
        data._load_sentence_pairs_rows_cached.cache_clear()
        data._load_gal_rows_cached.cache_clear()
        data._load_kk_rows_cached.cache_clear()
        data._load_context_source_entries_cached.cache_clear()
        data._load_grammar_cached.cache_clear()
        dictionary_store._get_dictionary_store_cached.cache_clear()
        dictionary_store._compiled_word_pattern.cache_clear()

    def test_data_loaders_resolve_pack_specific_dir_before_legacy_root(self) -> None:
        self.assertEqual(
            data._get_data_path("sentence_pairs.tsv", pack_id="tsova_tush"),
            str(self.bats_dir / "sentence_pairs.tsv"),
        )
        self.assertEqual(
            data._get_data_path("sentence_pairs.tsv"),
            str(self.data_root / "sentence_pairs.tsv"),
        )

        self.assertEqual(data._load_master_lexicon_rows("tsova_tush"), (("eq:ar", "eq:ar", "jump"),))
        self.assertEqual(data._load_sentence_pairs_rows("tsova_tush"), (("eq:ar", "jump"),))
        self.assertEqual(data._load_gal_rows("tsova_tush"), (("прыгать", "eq:ar"),))
        self.assertEqual(data._load_kk_rows("tsova_tush"), (("eq:ar", "", "прыгать", "ხტომა"),))
        self.assertEqual(data._load_context_source_entries("tsova_tush"), ("bats context",))
        self.assertEqual(data._load_grammar(pack_id="tsova_tush"), "bats grammar")
        self.assertEqual(data._load_compact_grammar(pack_id="tsova_tush"), "bats compact grammar")

        self.assertEqual(data._load_sentence_pairs_rows(), (("გომორძგუა", "hello"),))
        self.assertEqual(data._load_grammar(), "mingrelian grammar")

    def test_dictionary_stores_are_pack_scoped_and_keep_compatibility_accessors(self) -> None:
        mingrelian = get_dictionary_store()
        mingrelian_again = get_dictionary_store("mingrelian")
        bats = get_dictionary_store("tsova_tush")

        self.assertIs(mingrelian, mingrelian_again)
        self.assertIsNot(mingrelian, bats)

        self.assertEqual(mingrelian.exact_sentence_low_resource("გომორძგუა")[0].english, "hello")
        self.assertEqual(bats.exact_sentence_low_resource("eq:ar")[0].english, "jump")
        self.assertEqual(bats.exact_sentence_tsova_tush("eq:ar")[0].tsova_tush, "eq:ar")
        self.assertEqual(bats.exact_sentence_tsova_tush("eq:ar")[0].mingrelian, "eq:ar")
        self.assertEqual(bats.exact_kk_low_resource("eq:ar")[0].georgian, "ხტომა")
        self.assertEqual(
            bats.exact_translation_overrides("english", "tsova_tush", "jump")[0].target_text,
            "eq:ar",
        )

        bats_result = bats.search_kk("eq:ar")
        mingrelian_result = mingrelian.search_kk("გომორძგუა")
        self.assertIn("Bats: eq:ar", bats_result.output)
        self.assertNotIn("Mingrelian: eq:ar", bats_result.output)
        self.assertIn("Mingrelian: გომორძგუა", mingrelian_result.output)
        self.assertEqual(mingrelian.search_sentence_pairs("eq:ar").output, "")


if __name__ == "__main__":
    unittest.main()
