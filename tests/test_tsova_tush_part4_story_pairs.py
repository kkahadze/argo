import tempfile
import unittest
from pathlib import Path

from src.tsova_tush.part4_story_pair_extraction import (
    extract_part4_story_translation_pairs,
)


class TsovaTushPart4StoryPairExtractionTests(unittest.TestCase):
    def test_extracts_story_level_batsbi_georgian_english_triples(self) -> None:
        rows = extract_part4_story_translation_pairs(
            "batsbi title one\n"
            "befcu×nç: oeõ\n"
            "batsbi line one.\n"
            "batsbi line two.\n\n"
            "31\n\n"
            "georgian title one\n"
            "mTxr.: igive\n"
            "georgian line one.\n\n"
            "33\n\n"
            "English Title One\n"
            "Narrator: the same\n"
            "english line one.\n"
            "english line two.\n\n"
            "35\n\n"
            "batsbi title two\n"
            "befcu×nç: oeõ\n"
            "batsbi second story.\n\n"
            "36\n\n"
            "georgian title two\n"
            "mTxr.: igive\n"
            "georgian second story.\n\n"
            "39\n\n"
            "English Title Two\n"
            "Narrator: the same\n"
            "english second story.\n\n"
            "42\n\n"
            "sar­Cev : sar­Ce­vi : Contents\n"
            "batsbi title one\n"
            "31\n"
            "georgian title one\n"
            "33\n"
            "English Title One\n"
            "35\n"
            "batsbi title two\n"
            "36\n"
            "georgian title two\n"
            "39\n"
            "English Title Two\n"
            "42\n",
            source_name="tsovatush_texts_part4_2017",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_id"], "tsovatush_texts_part4_2017:story-001")
        self.assertEqual(rows[0]["pair_type"], "batsbi_georgian_english_story")
        self.assertEqual(rows[0]["batsbi_text"], "batsbi line one. batsbi line two.")
        self.assertEqual(rows[0]["georgian_translation"], "georgian line one.")
        self.assertEqual(
            rows[0]["english_translation"],
            "english line one. english line two.",
        )
        self.assertEqual(rows[0]["confidence"], "review")
        self.assertIn("story-level", rows[0]["notes"])

        self.assertEqual(rows[1]["batsbi_text"], "batsbi second story.")
        self.assertEqual(rows[1]["georgian_translation"], "georgian second story.")
        self.assertEqual(rows[1]["english_translation"], "english second story.")

    def test_accepts_path_input_and_skips_incomplete_triples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "part4.txt"
            source_path.write_text(
                "batsbi title one\n"
                "befcu×nç: oeõ\n"
                "batsbi text.\n\n"
                "31\n\n"
                "georgian title one\n"
                "mTxr.: igive\n"
                "georgian text.\n\n"
                "33\n\n"
                "English Title One\n"
                "Narrator: the same\n"
                "english text.\n\n"
                "35\n\n"
                "sar­Cev : sar­Ce­vi : Contents\n"
                "batsbi title one\n"
                "31\n"
                "georgian title one\n"
                "33\n"
                "English Title One\n"
                "35\n"
                "orphan batsbi title\n"
                "40\n",
                encoding="utf-8",
            )

            rows = extract_part4_story_translation_pairs(
                source_path,
                source_name="tsovatush_texts_part4_2017",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["english_translation"], "english text.")

    def test_ignores_bare_narrator_words_inside_story_text(self) -> None:
        rows = extract_part4_story_translation_pairs(
            "batsbi title\n"
            "befcu×nç: oeõ\n"
            "story text mentioning Narrator without colon.\n\n"
            "georgian title\n"
            "mTxrobeli: arsen berTlani\n"
            "georgian text.\n\n"
            "English Title\n"
            "Narrator: Arsen Bertlani\n"
            "english text.\n\n"
            "sar­Cev : sar­Ce­vi : Contents\n"
            "batsbi title\n"
            "31\n"
            "georgian title\n"
            "33\n"
            "English Title\n"
            "35\n",
            source_name="tsovatush_texts_part4_2017",
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("Narrator without colon", rows[0]["batsbi_text"])

    def test_uses_narrator_anchors_even_without_contents_block(self) -> None:
        rows = extract_part4_story_translation_pairs(
            "batsbi title\n"
            "×efcu×nç: oeõ\n"
            "batsbi text.\n\n"
            "georgian title\n"
            "mTxr.: igive\n"
            "georgian text.\n\n"
            "English Title\n"
            "Narrator: the same\n"
            "english text.\n",
            source_name="tsovatush_texts_part4_2017",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["batsbi_text"], "batsbi text.")
        self.assertEqual(rows[0]["georgian_translation"], "georgian text.")
        self.assertEqual(rows[0]["english_translation"], "english text.")


if __name__ == "__main__":
    unittest.main()
