import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.translator import data
from scripts.build_svan_runtime_data import build_runtime_data


class SvanRuntimeDataTests(unittest.TestCase):
    def test_builder_materializes_dictionary_context_and_grammar_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            private_data_dir = root / "private_data" / "svan"
            ready_dir.mkdir(parents=True)

            ready_dir.joinpath("liparteliani_dictionary_ready.tsv").write_text(
                "source_id\theadword_svan\tgeorgian_gloss\tseparator\tstatus_reason\tsource_line\n"
                "liparteliani:line-1\tქა\tსახელი\t-\tstable_dash_entry_shape\t1\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("titus_svan_georgian_pairs_high_confidence.tsv").write_text(
                "source_id\tsource_name\tpair_type\tsvan_text\tgeorgian_translation\tenglish_translation\tconfidence\tnotes\tsvan_source_url\tgeorgian_source_url\n"
                "titus:1\ttitus\tblock\tქა სვანურად\tეს არის სახელი\t\thigh\tpaired\t\t\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("tuite_svan_grammar_2023.txt").write_text(
                "SVAN GRAMMAR",
                encoding="utf-8",
            )

            counts = build_runtime_data(ready_dir=ready_dir, private_data_dir=private_data_dir)

            self.assertEqual(counts["kk_rows"], 1)
            self.assertEqual(counts["context_blocks"], 1)

            with private_data_dir.joinpath("kk.tsv").open("r", encoding="utf-8", newline="") as file:
                kk_rows = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(kk_rows[0]["word"], "ქა")
            self.assertEqual(kk_rows[0]["georgian_def"], "სახელი")

            context = private_data_dir.joinpath("context_source.txt").read_text(encoding="utf-8")
            self.assertIn("Svan: ქა სვანურად", context)
            self.assertIn("Georgian: ეს არის სახელი", context)

            self.assertEqual(
                private_data_dir.joinpath("tuite.txt").read_text(encoding="utf-8"),
                "SVAN GRAMMAR",
            )
            self.assertEqual(
                private_data_dir.joinpath("tuite_compact.txt").read_text(encoding="utf-8"),
                "SVAN GRAMMAR",
            )
            self.assertFalse(private_data_dir.joinpath("harris.txt").exists())
            self.assertFalse(private_data_dir.joinpath("harris_compact.txt").exists())

            data._load_grammar_cached.cache_clear()
            with patch.dict(os.environ, {"ARGO_DATA_DIR": str(root / "private_data")}, clear=False):
                self.assertEqual(data._load_grammar(pack_id="svan"), "SVAN GRAMMAR")
                self.assertEqual(data._load_compact_grammar(pack_id="svan"), "SVAN GRAMMAR")


if __name__ == "__main__":
    unittest.main()
