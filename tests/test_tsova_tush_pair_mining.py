import csv
import tempfile
import unittest
from pathlib import Path

from scripts.mine_tsova_tush_translation_pairs import (
    PairMiningInputs,
    mine_translation_pairs,
)


class TsovaTushPairMiningTests(unittest.TestCase):
    def test_mines_titus_pairs_and_writes_confidence_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            output_dir = root / "pairs"
            ready_dir.mkdir()

            (ready_dir / "titus_examples.tsv").write_text(
                "lemma\tlemma_number\texample_number\tbatsbi_text\tbatsbi_text_tokenized\tgeorgian_translation\tsource_url\n"
                "d-a\t1\t1\tსტაკ ვა\tსტაკ ვა\tკაცი არის\thttps://example.test/1\n"
                "d-a\t1\t1\tსტაკ ვა\tსტაკ ვა\tკაცი არის\thttps://example.test/1\n",
                encoding="utf-8",
            )

            manifest = mine_translation_pairs(
                PairMiningInputs(
                    ready_dir=ready_dir,
                    output_dir=output_dir,
                )
            )

            self.assertEqual(manifest["counts"]["high_confidence_pairs"], 1)
            self.assertEqual(manifest["counts"]["review_pairs"], 0)

            with (output_dir / "pairs_high_confidence.tsv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pair_type"], "batsbi_georgian")
            self.assertEqual(rows[0]["batsbi_text"], "სტაკ ვა")
            self.assertEqual(rows[0]["georgian_translation"], "კაცი არის")
            self.assertEqual(rows[0]["confidence"], "high")

    def test_includes_numbered_text_book_pairs_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            output_dir = root / "pairs"
            ready_dir.mkdir()

            (ready_dir / "titus_examples.tsv").write_text(
                "lemma\tlemma_number\texample_number\tbatsbi_text\tbatsbi_text_tokenized\tgeorgian_translation\tsource_url\n",
                encoding="utf-8",
            )
            (ready_dir / "tsovatush_texts_part1_2009.txt").write_text(
                "10. ბაცბური წინადადება.\n"
                "11. კიდევ ერთი ბაცბური.\n\n"
                "10. ქართული თარგმანი.\n"
                "11. კიდევ ერთი ქართული თარგმანი.\n",
                encoding="utf-8",
            )
            (ready_dir / "tsovatush_texts_part4_2017.txt").write_text("", encoding="utf-8")

            manifest = mine_translation_pairs(
                PairMiningInputs(
                    ready_dir=ready_dir,
                    output_dir=output_dir,
                )
            )

            self.assertEqual(manifest["counts"]["high_confidence_pairs"], 2)

            with (output_dir / "pairs_high_confidence.tsv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file, delimiter="\t"))

            self.assertEqual(rows[0]["source_name"], "tsovatush_texts_part1_2009")
            self.assertEqual(rows[0]["pair_type"], "batsbi_georgian")
            self.assertEqual(rows[0]["batsbi_text"], "ბაცბური წინადადება.")
            self.assertEqual(rows[0]["georgian_translation"], "ქართული თარგმანი.")

    def test_includes_grammar_example_pairs_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            output_dir = root / "pairs"
            ready_dir.mkdir()

            (ready_dir / "titus_examples.tsv").write_text(
                "lemma\tlemma_number\texample_number\tbatsbi_text\tbatsbi_text_tokenized\tgeorgian_translation\tsource_url\n",
                encoding="utf-8",
            )
            (ready_dir / "tsovatush_texts_part1_2009.txt").write_text("", encoding="utf-8")
            (ready_dir / "tsovatush_texts_part4_2017.txt").write_text("", encoding="utf-8")
            (ready_dir / "holisky_gagua_1994.txt").write_text(
                "eq:ar\n‘jumpʼ\n",
                encoding="utf-8",
            )

            manifest = mine_translation_pairs(
                PairMiningInputs(
                    ready_dir=ready_dir,
                    output_dir=output_dir,
                )
            )

            self.assertEqual(manifest["counts"]["high_confidence_pairs"], 1)

            with (output_dir / "pairs_high_confidence.tsv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file, delimiter="\t"))

            self.assertEqual(rows[0]["source_name"], "holisky_gagua_1994")
            self.assertEqual(rows[0]["pair_type"], "batsbi_english")
            self.assertEqual(rows[0]["batsbi_text"], "eq:ar")
            self.assertEqual(rows[0]["english_translation"], "jump")

    def test_includes_part4_story_level_review_pairs_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            output_dir = root / "pairs"
            ready_dir.mkdir()

            (ready_dir / "titus_examples.tsv").write_text(
                "lemma\tlemma_number\texample_number\tbatsbi_text\tbatsbi_text_tokenized\tgeorgian_translation\tsource_url\n",
                encoding="utf-8",
            )
            (ready_dir / "tsovatush_texts_part1_2009.txt").write_text("", encoding="utf-8")
            (ready_dir / "tsovatush_texts_part4_2017.txt").write_text(
                "batsbi title\n"
                "befcu×nç: oeõ\n"
                "batsbi story.\n\n"
                "31\n\n"
                "georgian title\n"
                "mTxr.: igive\n"
                "georgian story.\n\n"
                "33\n\n"
                "English Title\n"
                "Narrator: the same\n"
                "english story.\n\n"
                "35\n\n"
                "sar­Cev : sar­Ce­vi : Contents\n"
                "batsbi title\n"
                "31\n"
                "georgian title\n"
                "33\n"
                "English Title\n"
                "35\n",
                encoding="utf-8",
            )
            (ready_dir / "holisky_gagua_1994.txt").write_text("", encoding="utf-8")

            manifest = mine_translation_pairs(
                PairMiningInputs(
                    ready_dir=ready_dir,
                    output_dir=output_dir,
                )
            )

            self.assertEqual(manifest["counts"]["high_confidence_pairs"], 0)
            self.assertEqual(manifest["counts"]["review_pairs"], 1)

            with (output_dir / "pairs_review.tsv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file, delimiter="\t"))

            self.assertEqual(rows[0]["source_name"], "tsovatush_texts_part4_2017")
            self.assertEqual(rows[0]["pair_type"], "batsbi_georgian_english_story")
            self.assertEqual(rows[0]["confidence"], "review")


if __name__ == "__main__":
    unittest.main()
