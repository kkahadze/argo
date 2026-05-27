import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_tsova_tush_assets import (
    AssetBuildInputs,
    build_assets,
    fetch_titus_pages,
    parse_titus_dictionary_page,
)


class TsovaTushAssetBuilderTests(unittest.TestCase):
    def test_fetch_titus_pages_stops_on_soft_index_fallback(self) -> None:
        class FakeResponse:
            def __init__(self, payload: str) -> None:
                self.payload = payload.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def read(self) -> bytes:
                return self.payload

        with patch(
            "scripts.build_tsova_tush_assets.urlopen",
            side_effect=[
                FakeResponse("<title>Batsbi-Georgian-Russian Dictionary</title>"),
                FakeResponse("<title>TITUS index</title>"),
            ],
        ):
            pages = fetch_titus_pages(base_url="https://example.test", max_pages=2)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][0], "https://example.test/tt_di001.htm")

    def test_parse_titus_dictionary_page_extracts_entries_and_examples(self) -> None:
        html = """
        <span id="n16"><span id="h3">Lemma: d-a</span></span>
        <span id="n16"><span id="h4">Number: 1</span></span>
        <span id="emttb22">დ-ა</span>
        <span id="ectt22">d-a</span>
        <span id="mxngb16">არის</span>
        <span id="slrub16">есть</span>
        <span id="n16"><span id="h5">Example: 1</span></span>
        <span id="emtt16">ღაზე<sup>ნ</sup> წა და</span>
        <span id="mxng16">მას კარგი სახლი აქვს</span>
        """

        entries, examples = parse_titus_dictionary_page(
            html,
            source_url="https://example.test/tt_di001.htm",
        )

        self.assertEqual(
            entries,
            [
                {
                    "lemma": "d-a",
                    "lemma_number": "1",
                    "batsbi_mkhedruli": "დ-ა",
                    "batsbi_transcription": "d-a",
                    "georgian_gloss": "არის",
                    "russian_gloss": "есть",
                    "source_url": "https://example.test/tt_di001.htm",
                }
            ],
        )
        self.assertEqual(
            examples,
            [
                {
                    "lemma": "d-a",
                    "lemma_number": "1",
                    "example_number": "1",
                    "batsbi_text": "ღაზენ წა და",
                    "batsbi_text_tokenized": "ღაზე{sup:ნ} წა და",
                    "georgian_translation": "მას კარგი სახლი აქვს",
                    "source_url": "https://example.test/tt_di001.htm",
                }
            ],
        )

    def test_build_assets_writes_ready_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "ready"
            source_dir.mkdir()

            ids_tsv = source_dir / "ids.tab"
            ids_tsv.write_text(
                "chapter_id\tentry_id\tmeaning\tTsova-Tush_CyrillTrans\tTsova-Tush_Phonemic\tcomment\n"
                "1\t100\tworld\tдуниа\tdunia\t\n"
                "1\t210\tearth, land\tкве̄къаᴴ; мохкІ\tkwēqʼaⁿ; moxkʼ\t\n",
                encoding="utf-8",
            )
            grammar_full = source_dir / "full.txt"
            grammar_full.write_text("  Full grammar  \n", encoding="utf-8")
            grammar_compact = source_dir / "compact.txt"
            grammar_compact.write_text("Compact grammar\n", encoding="utf-8")
            grammar_classic = source_dir / "classic.txt"
            grammar_classic.write_text("Classic grammar\n", encoding="utf-8")
            texts_part1 = source_dir / "part1.txt"
            texts_part1.write_text("ა\x03ბ\n", encoding="utf-8")
            texts_part4 = source_dir / "part4.txt"
            texts_part4.write_text("Part four prose\n", encoding="utf-8")

            manifest = build_assets(
                AssetBuildInputs(
                    ids_tsv=ids_tsv,
                    grammar_full=grammar_full,
                    grammar_compact=grammar_compact,
                    grammar_classic=grammar_classic,
                    texts_part1=texts_part1,
                    texts_part4=texts_part4,
                    output_dir=output_dir,
                )
            )

            self.assertEqual(manifest["counts"]["ids_lexicon_rows"], 3)
            self.assertTrue((output_dir / "ids_lexicon.csv").exists())
            self.assertTrue((output_dir / "ids_exact_overrides.tsv").exists())
            self.assertTrue((output_dir / "context_source.txt").exists())
            self.assertTrue((output_dir / "build_manifest.json").exists())

            with (output_dir / "ids_lexicon.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[0]["canonical_practical"], "dunia")
            self.assertEqual(rows[1]["canonical_practical"], "kwēqʼaⁿ")
            self.assertEqual(rows[2]["canonical_practical"], "moxkʼ")

            overrides = (output_dir / "ids_exact_overrides.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("english\ttsova_tush\tworld\tdunia", overrides)
            self.assertIn("tsova_tush\tenglish\tmoxkʼ\tearth, land", overrides)

            cleaned_part1 = (output_dir / "tsovatush_texts_part1_2009.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(cleaned_part1.strip(), "ა - ბ")

            loaded_manifest = json.loads(
                (output_dir / "build_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded_manifest["counts"]["ids_lexicon_rows"], 3)


if __name__ == "__main__":
    unittest.main()
