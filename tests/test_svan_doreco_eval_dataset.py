import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.build_svan_doreco_eval_dataset import (
    DorecoSegment,
    build_dataset,
    select_balanced_segments,
)


class SvanDorecoEvalDatasetTests(unittest.TestCase):
    def test_builder_uses_clean_mkhedruli_svan_georgian_eaf_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "doreco.zip"
            output_path = root / "heldout.csv"
            eaf_path = root / "doreco_svan1243_story.eaf"
            eaf_path.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
  <TIER TIER_ID="ref@SPK">
    <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"><ANNOTATION_VALUE>0001_DoReCo_story</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
    <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2"><ANNOTATION_VALUE>0002_DoReCo_story</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
    <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a3"><ANNOTATION_VALUE>0003_DoReCo_story</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  </TIER>
  <TIER TIER_ID="tr1@SPK-" PARENT_REF="ref@SPK">
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a4" ANNOTATION_REF="a1"><ANNOTATION_VALUE>სვანური წინადადება ტესტისთვის.</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a5" ANNOTATION_REF="a2"><ANNOTATION_VALUE>&lt;p:&gt;</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a6" ANNOTATION_REF="a3"><ANNOTATION_VALUE>_ ფრაგმენტი...</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
  </TIER>
  <TIER TIER_ID="fg@SPK-" PARENT_REF="ref@SPK">
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a7" ANNOTATION_REF="a1"><ANNOTATION_VALUE>ქართული თარგმანი ტესტისთვის.</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a8" ANNOTATION_REF="a2"><ANNOTATION_VALUE>&lt;p:&gt;</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
    <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="a9" ANNOTATION_REF="a3"><ANNOTATION_VALUE>_ ფრაგმენტი...</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION>
  </TIER>
</ANNOTATION_DOCUMENT>
""",
                encoding="utf-8",
            )

            with ZipFile(zip_path, "w") as archive:
                archive.write(eaf_path, arcname=f"bundle/{eaf_path.name}")

            count = build_dataset(zip_path=zip_path, output_path=output_path, target_per_bucket=None)

            self.assertEqual(count, 1)
            with output_path.open("r", encoding="utf-8", newline="") as file:
                written = list(csv.DictReader(file))

            self.assertEqual(written[0]["source_id"], "doreco:doreco_svan1243_story:SPK-:0001_DoReCo_story")
            self.assertEqual(written[0]["svan"], "სვანური წინადადება ტესტისთვის.")
            self.assertEqual(written[0]["georgian"], "ქართული თარგმანი ტესტისთვის.")

    def test_sampler_keeps_sparse_length_buckets(self) -> None:
        segments = [
            DorecoSegment("short", "recording", "SPK", "short-ref", "შ" * 40, "Short."),
            DorecoSegment("medium", "recording", "SPK", "medium-ref", "მ" * 120, "Medium."),
            DorecoSegment("long", "recording", "SPK", "long-ref", "ლ" * 240, "Long."),
        ]

        selected = select_balanced_segments(segments, target_per_bucket=16)

        self.assertEqual([segment.source_id for segment in selected], ["short", "medium", "long"])


if __name__ == "__main__":
    unittest.main()
