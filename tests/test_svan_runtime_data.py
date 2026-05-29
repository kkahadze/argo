import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.translator import data
from scripts.build_svan_runtime_data import _build_parallel_overrides, build_runtime_data


class SvanRuntimeDataTests(unittest.TestCase):
    def test_parallel_overrides_skip_ambiguous_source_forms(self) -> None:
        rows = _build_parallel_overrides(
            [
                {"svan_text": "ერთი ფორმა", "georgian_translation": "პირველი აზრი"},
                {"svan_text": "ერთი ფორმა", "georgian_translation": "მეორე აზრი"},
                {"svan_text": "სტაბილური", "georgian_translation": "ზუსტი"},
            ]
        )

        self.assertNotIn(
            {
                "source_language": "svan",
                "target_language": "georgian",
                "source_text": "ერთი ფორმა",
                "target_text": "პირველი აზრი",
            },
            rows,
        )
        self.assertIn(
            {
                "source_language": "svan",
                "target_language": "georgian",
                "source_text": "სტაბილური",
                "target_text": "ზუსტი",
            },
            rows,
        )

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
            ready_dir.joinpath("topuria_kaldani_dictionary_ready.tsv").write_text(
                "source_id\tarticle_index\theadword_svan_raw\theadword_svan_nfc\thead_region_raw\tgeorgian_definition\tsense_no\tdialect_labels\tpart_of_speech_labels\tpage_start\tpage_end\tsource_line_start\tsource_line_end\tconfidence\tstatus_reason\tarticle_raw\n"
                "topuria-kaldani:article-00001\t1\tქა\tქა\tქა\tსახელი\t\t\t\t1\t1\t1\t1\thigh\tpromoted\t\n"
                "topuria-kaldani:article-00002\t2\tლახუ\tლახუ\tლახუ\tხელი\t\t\t\t1\t1\t2\t2\thigh\tpromoted\t\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_cross_references.tsv").write_text(
                "source_id\tarticle_index\tsource_headword_raw\trelation_type\t"
                "target_headword_raw\tpage_start\tpage_end\tsource_line_start\t"
                "source_line_end\tarticle_raw\tconfidence\n"
                "topuria-kaldani:article-00003\t3\tხოშა\tsame_as\tხოჩა\t3\t3\t"
                "10\t10\tხოშა იგივეა, რაც ხოჩა\thigh\n"
                "topuria-kaldani:article-00004\t4\tხოშა\tsee\tხოჩა\t4\t4\t"
                "11\t11\tიხ. ხოჩა\thigh\n"
                "topuria-kaldani:article-00005\t5\tხოჩა\tsame_as\tხოჩა\t5\t5\t"
                "12\t12\tხოჩა იგივეა, რაც ხოჩა\thigh\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("titus_svan_georgian_pairs_high_confidence.tsv").write_text(
                "source_id\tsource_name\tpair_type\tsvan_text\tgeorgian_translation\tenglish_translation\tconfidence\tnotes\tsvan_source_url\tgeorgian_source_url\n"
                "titus:1\ttitus\tblock\tქა სვანურად\tეს არის სახელი\t\thigh\tpaired\t\t\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_example_pairs_ready.tsv").write_text(
                "source_id\tarticle_index\theadword_svan_raw\tsvan_text\tgeorgian_translation\tcitation_label\tcitation_page\tdictionary_page_start\tdictionary_page_end\tsource_line_start\tsource_line_end\tconfidence\tnotes\tarticle_raw\n"
                "topuria-kaldani:article-00002:example-01\t2\tლახუ\tლახუ ვოხუ̂ე\tხელი დავიბანე\tბზ.\t1\t1\t1\t2\t2\thigh\tpromoted\t\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_example_pairs_domain.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\tdomain_type\n"
                "topuria-kaldani:domain-1\tგი̄მ მუ̂ეხუ̂ე\tგული მღერის\tpoetry\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_svan_georgian_conversation_pairs.tsv").write_text(
                "source_id\tcategory\tsvan_text\tsvan_variants\tgeorgian_translation\tsource_url\tevidence\n"
                "topuria:greeting\tgreeting\tიმჟი ხა̈რი?\tიმჟი ხა̈რი?|||იმჟი ხარი?\tროგორ ხარ?\turl\tpaired\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("quizlet_svan_georgian_conversation_pairs.tsv").write_text(
                "source_id\tcategory\tsvan_text\tsvan_variants\tgeorgian_translation\tsource_url\tevidence\tconfidence\n"
                "quizlet:question\tlocation_question\tსი იმე ხარი\tსი იმე ხარი\tშენ სად ხარ?\turl\tdirect pair\tusable_prompt_evidence\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("tuite_svan_grammar_2023.txt").write_text(
                "SVAN GRAMMAR",
                encoding="utf-8",
            )
            ready_dir.joinpath("she_2024_morphology_compact_verified.txt").write_text(
                "===== SOURCE: She 2024, PDF page 523 render (printed page 522) =====\n"
                "DIALECT: Lenṭex\n"
                "LEMMA/GLOSS: 'construire' ('to build')\n"
                "PARADIGM: Series I - present subseries\n"
                "\n"
                "Person\tPresent\tImperfect\n"
                "1SG\txûagem\txûagema\n",
                encoding="utf-8",
            )

            counts = build_runtime_data(ready_dir=ready_dir, private_data_dir=private_data_dir)

            self.assertEqual(counts["kk_rows"], 2)
            self.assertEqual(counts["topuria_dictionary_rows"], 1)
            self.assertEqual(counts["parallel_pair_rows"], 2)
            self.assertEqual(counts["topuria_domain_pair_rows"], 1)
            self.assertEqual(counts["context_blocks"], 1)
            self.assertEqual(counts["quizlet_context_blocks"], 1)
            self.assertEqual(counts["override_rows"], 4)
            self.assertEqual(counts["attested_variant_rows"], 2)
            self.assertEqual(counts["paradigm_form_rows"], 2)

            with private_data_dir.joinpath("kk.tsv").open("r", encoding="utf-8", newline="") as file:
                kk_rows = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(kk_rows[0]["word"], "ქა")
            self.assertEqual(kk_rows[0]["georgian_def"], "სახელი")
            self.assertEqual(kk_rows[1]["word"], "ლახუ")
            self.assertEqual(kk_rows[1]["georgian_def"], "ხელი")

            with private_data_dir.joinpath("parallel_pairs.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                parallel_pairs = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(
                parallel_pairs,
                [
                    {
                        "low_resource": "ლახუ ვოხუ̂ე",
                        "georgian": "ხელი დავიბანე",
                        "source_id": "topuria-kaldani:article-00002:example-01",
                        "source_family": "topuria-kaldani",
                        "evidence_type": "ordinary_example",
                        "domain_type": "",
                    },
                    {
                        "low_resource": "გი̄მ მუ̂ეხუ̂ე",
                        "georgian": "გული მღერის",
                        "source_id": "topuria-kaldani:domain-1",
                        "source_family": "topuria-kaldani",
                        "evidence_type": "domain_example",
                        "domain_type": "poetry",
                    },
                ],
            )

            with private_data_dir.joinpath("attested_variants.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                variants = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(
                {(row["query_form_raw"], row["related_form_raw"]) for row in variants},
                {("ხოშა", "ხოჩა"), ("ხოჩა", "ხოშა")},
            )
            self.assertTrue(all(row["relation_type"] == "same_as" for row in variants))

            with private_data_dir.joinpath("paradigm_forms.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                paradigm_forms = list(csv.DictReader(file, delimiter="\t"))
            self.assertEqual(
                {(row["form_raw"], row["person_slot"], row["column_raw"]) for row in paradigm_forms},
                {("xûagem", "1SG", "Present"), ("xûagema", "1SG", "Imperfect")},
            )

            context = private_data_dir.joinpath("context_source.txt").read_text(encoding="utf-8")
            self.assertIn("Svan: ქა სვანურად", context)
            self.assertIn("Georgian: ეს არის სახელი", context)
            self.assertIn("Svan: სი იმე ხარი", context)
            self.assertIn("Georgian: შენ სად ხარ?", context)
            self.assertIn("prompt evidence only", context)
            self.assertNotIn("როგორ ხარ?", context)
            self.assertNotIn("ხელი დავიბანე", context)

            with private_data_dir.joinpath("translation_overrides.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                overrides = list(csv.DictReader(file, delimiter="\t"))
            self.assertNotIn(
                {
                    "source_language": "georgian",
                    "target_language": "svan",
                    "source_text": "როგორ ხარ?",
                    "target_text": "იმჟი ხა̈რი?",
                },
                overrides,
            )
            self.assertIn(
                {
                    "source_language": "georgian",
                    "target_language": "svan",
                    "source_text": "შენ სად ხარ?",
                    "target_text": "სი იმე ხარი",
                },
                overrides,
            )
            self.assertIn(
                {
                    "source_language": "svan",
                    "target_language": "georgian",
                    "source_text": "სი იმე ხარი",
                    "target_text": "შენ სად ხარ?",
                },
                overrides,
            )
            self.assertIn(
                {
                    "source_language": "svan",
                    "target_language": "georgian",
                    "source_text": "ქა სვანურად",
                    "target_text": "ეს არის სახელი",
                },
                overrides,
            )
            self.assertNotIn(
                {
                    "source_language": "svan",
                    "target_language": "georgian",
                    "source_text": "იმჟი ხარი?",
                    "target_text": "როგორ ხარ?",
                },
                overrides,
            )
            self.assertNotIn(
                {
                    "source_language": "svan",
                    "target_language": "georgian",
                    "source_text": "ლახუ ვოხუ̂ე",
                    "target_text": "ხელი დავიბანე",
                },
                overrides,
            )

            self.assertEqual(
                private_data_dir.joinpath("tuite.txt").read_text(encoding="utf-8"),
                "SVAN GRAMMAR",
            )
            self.assertEqual(
                private_data_dir.joinpath("tuite_compact.txt").read_text(encoding="utf-8"),
                "SVAN GRAMMAR",
            )
            self.assertEqual(
                private_data_dir.joinpath("morphology_support.txt").read_text(encoding="utf-8"),
                "===== SOURCE: She 2024, PDF page 523 render (printed page 522) =====\n"
                "DIALECT: Lenṭex\n"
                "LEMMA/GLOSS: 'construire' ('to build')\n"
                "PARADIGM: Series I - present subseries\n"
                "\n"
                "Person\tPresent\tImperfect\n"
                "1SG\txûagem\txûagema\n",
            )
            self.assertFalse(private_data_dir.joinpath("harris.txt").exists())
            self.assertFalse(private_data_dir.joinpath("harris_compact.txt").exists())

            data._load_grammar_cached.cache_clear()
            with patch.dict(os.environ, {"ARGO_DATA_DIR": str(root / "private_data")}, clear=False):
                self.assertEqual(data._load_grammar(pack_id="svan"), "SVAN GRAMMAR")
                self.assertEqual(data._load_compact_grammar(pack_id="svan"), "SVAN GRAMMAR")

    def test_builder_loads_audited_supplemental_runtime_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready_dir = root / "ready"
            private_data_dir = root / "private_data" / "svan"
            ready_dir.mkdir(parents=True)

            ready_dir.joinpath("liparteliani_dictionary_ready.tsv").write_text(
                "source_id\theadword_svan\tgeorgian_gloss\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_dictionary_ready.tsv").write_text(
                "source_id\theadword_svan_raw\tgeorgian_definition\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("titus_svan_georgian_pairs_high_confidence.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_example_pairs_ready.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("topuria_kaldani_example_pairs_domain.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\tdomain_type\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("quizlet_svan_georgian_conversation_pairs.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("tuite_svan_grammar_2023.txt").write_text("", encoding="utf-8")
            ready_dir.joinpath("she_2024_morphology_compact_verified.txt").write_text("", encoding="utf-8")

            ready_dir.joinpath("supplemental_russian_svan_lexicon_ready.tsv").write_text(
                "source_id\trussian\tsvan\tsource_scheme\tconfidence\n"
                "test:ru-1\tвода\tლიც\tmkhedruli\taudited\n"
                "test:ru-dup\tвода\tლიც\tmkhedruli\taudited\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("supplemental_svan_georgian_lexicon_ready.tsv").write_text(
                "source_id\tsvan\tgeorgian\tconfidence\n"
                "test:ka-1\tლიც\tწყალი\taudited\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("supplemental_svan_georgian_pairs_ready.tsv").write_text(
                "source_id\tsvan_text\tgeorgian_translation\tdomain\tconfidence\n"
                "test:pair-1\tლიც მესგუ̂ა\tწყალი დავლიე\tproverb\taudited\n",
                encoding="utf-8",
            )
            ready_dir.joinpath("supplemental_svan_english_lexicon_ready.tsv").write_text(
                "source_id\tenglish\tsvan\tconfidence\n"
                "test:en-1\twolf\tთხე̄რე\taudited\n"
                "test:en-dup\twolf\tთხე̄რე\taudited\n",
                encoding="utf-8",
            )

            counts = build_runtime_data(ready_dir=ready_dir, private_data_dir=private_data_dir)

            self.assertEqual(counts["supplemental_russian_rows"], 1)
            self.assertEqual(counts["supplemental_georgian_lexicon_rows"], 1)
            self.assertEqual(counts["supplemental_parallel_pair_rows"], 1)
            self.assertEqual(counts["supplemental_english_rows"], 1)

            with private_data_dir.joinpath("gal.tsv").open("r", encoding="utf-8", newline="") as file:
                self.assertEqual(
                    list(csv.DictReader(file, delimiter="\t")),
                    [{"russian": "вода", "svan": "ლიც"}],
                )
            with private_data_dir.joinpath("sentence_pairs.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as file:
                self.assertEqual(
                    list(csv.DictReader(file, delimiter="\t")),
                    [{"svan": "თხე̄რე", "english": "wolf"}],
                )
            with private_data_dir.joinpath("kk.tsv").open("r", encoding="utf-8", newline="") as file:
                self.assertEqual(
                    list(csv.DictReader(file, delimiter="\t")),
                    [{"word": "ლიც", "ipa": "", "russian_def": "", "georgian_def": "წყალი"}],
                )
            with private_data_dir.joinpath("parallel_pairs.tsv").open("r", encoding="utf-8", newline="") as file:
                self.assertEqual(
                    list(csv.DictReader(file, delimiter="\t")),
                    [{
                        "low_resource": "ლიც მესგუ̂ა",
                        "georgian": "წყალი დავლიე",
                        "source_id": "test:pair-1",
                        "source_family": "supplemental",
                        "evidence_type": "domain_example",
                        "domain_type": "proverb",
                    }],
                )
            with private_data_dir.joinpath("translation_overrides.tsv").open("r", encoding="utf-8", newline="") as file:
                self.assertEqual(list(csv.DictReader(file, delimiter="\t")), [])


if __name__ == "__main__":
    unittest.main()
