import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.translator import prompts
from src.translator.lookup import check_exact_match_simple
from src.translator.pipeline import translate


class UnifiedTranslatorRoutingTests(unittest.TestCase):
    def test_prompt_builder_registry_includes_bats_directions(self) -> None:
        self.assertIn(("tsova_tush", "english"), prompts.PROMPT_BUILDERS)
        self.assertIn(("english", "tsova_tush"), prompts.PROMPT_BUILDERS)
        self.assertIn(("tsova_tush", "georgian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("georgian", "tsova_tush"), prompts.PROMPT_BUILDERS)
        self.assertIn(("svan", "english"), prompts.PROMPT_BUILDERS)
        self.assertIn(("english", "svan"), prompts.PROMPT_BUILDERS)
        self.assertIn(("svan", "georgian"), prompts.PROMPT_BUILDERS)
        self.assertIn(("georgian", "svan"), prompts.PROMPT_BUILDERS)

    def test_pipeline_normalizes_reader_facing_bats_llm_output(self) -> None:
        class RecordingClient:
            provider = "test"
            model = "test"

            def complete(self, prompt: str) -> str:
                return "<<<TRANSLATION>>>\ndaqˁoⁿ kʼuitʼ0\n<<<END_TRANSLATION>>>"

        with (
            patch("src.translator.pipeline.GoogleTranslator", None),
            patch("src.translator.pipeline.collect_exact_match_candidates", return_value=[]),
            patch("src.translator.pipeline.check_exact_match_with_google_translate", return_value=None),
            patch.dict(
                prompts.PROMPT_BUILDERS,
                {
                    ("english", "tsova_tush"): lambda sentence, **kwargs: f"prompt for {sentence}",
                },
                clear=False,
            ),
        ):
            result = translate("big cat", "english", "tsova_tush", RecordingClient())

        self.assertEqual(result["translation"], "daqqoⁿ kʼuitʼ")

    def test_bats_prompt_uses_bats_grammar_channel(self) -> None:
        with (
            patch("src.translator.prompts._load_grammar_for_policy", return_value="BATS GRAMMAR"),
            patch("src.single_call_translator._load_grammar_for_policy", return_value="BATS GRAMMAR"),
        ):
            prompt = prompts.construct_prompt_from_english_to_tsova_tush(
                "hello",
                exact_candidates_block="Candidate 1",
                grammar_policy="full",
            )

        self.assertIn("Here is the Bats grammar information:", prompt)
        self.assertIn("BATS GRAMMAR", prompt)

    def test_svan_prompt_uses_svan_grammar_channel(self) -> None:
        with (
            patch("src.translator.prompts._load_grammar_for_policy", return_value="SVAN GRAMMAR"),
            patch("src.single_call_translator._load_grammar_for_policy", return_value="SVAN GRAMMAR"),
        ):
            prompt = prompts.construct_prompt_from_english_to_svan(
                "hello",
                exact_candidates_block="Candidate 1",
                grammar_policy="full",
            )

        self.assertIn("Here is the Svan grammar information:", prompt)
        self.assertIn("SVAN GRAMMAR", prompt)

    def test_bats_exact_lookup_uses_bats_runtime_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            bats_dir = data_root / "tsova_tush"
            bats_dir.mkdir()
            (bats_dir / "sentence_pairs.tsv").write_text(
                "tsova_tush\tenglish\n"
                "სტაკ ვა\tI am\n",
                encoding="utf-8",
            )
            (bats_dir / "gal.tsv").write_text("Russian\tTsova_Tush\n", encoding="utf-8")
            (bats_dir / "kk.tsv").write_text(
                "word\tipa\trussian_def\tgeorgian_def\n",
                encoding="utf-8",
            )
            (bats_dir / "translation_overrides.tsv").write_text(
                "source_language\ttarget_language\tsource_text\ttarget_text\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"ARGO_DATA_DIR": str(data_root)}, clear=False):
                self.assertEqual(
                    check_exact_match_simple("სტაკ ვა", "tsova_tush", "english"),
                    "I am",
                )

    def test_svan_exact_lookup_uses_svan_runtime_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            svan_dir = data_root / "svan"
            svan_dir.mkdir()
            (svan_dir / "sentence_pairs.tsv").write_text("svan\tenglish\n", encoding="utf-8")
            (svan_dir / "gal.tsv").write_text("russian\tsvan\n", encoding="utf-8")
            (svan_dir / "kk.tsv").write_text(
                "word\tipa\trussian_def\tgeorgian_def\n"
                "ქა\t\t\tსახელი\n",
                encoding="utf-8",
            )
            (svan_dir / "translation_overrides.tsv").write_text(
                "source_language\ttarget_language\tsource_text\ttarget_text\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"ARGO_DATA_DIR": str(data_root)}, clear=False):
                self.assertEqual(
                    check_exact_match_simple("ქა", "svan", "georgian"),
                    "სახელი",
                )
                self.assertEqual(
                    check_exact_match_simple("სახელი", "georgian", "svan"),
                    "ქა",
                )


if __name__ == "__main__":
    unittest.main()
