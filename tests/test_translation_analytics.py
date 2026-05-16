import unittest

from src.translation_analytics import build_translation_event, derive_translation_path_metrics


class TranslationAnalyticsTests(unittest.TestCase):
    def test_llm_with_evidence_bundle_gets_queryable_path_fields(self):
        event = build_translation_event(
            source_text="source",
            target_text="target",
            source_language="mingrelian",
            target_language="english",
            provider="gemini",
            model="gemini-test",
            duration_ms=1234,
            response_source="llm",
            used_user_api_key=False,
            prompt_metrics={
                "used_llm": True,
                "has_dictionary_entries": True,
                "used_grammar": True,
                "used_exact_candidate_shortlist": False,
                "exact_candidate_count": 2,
                "prompt_characters": 1000,
                "dict_entries_chars": 250,
                "grammar_chars": 500,
                "llm_call_ms": 900,
            },
        )

        self.assertEqual(event["translation_path"], "llm_evidence_bundle")
        self.assertTrue(event["used_llm"])
        self.assertTrue(event["used_evidence_bundle"])
        self.assertTrue(event["used_dictionary_entries"])
        self.assertTrue(event["used_grammar"])
        self.assertFalse(event["used_exact_candidate_shortlist"])
        self.assertEqual(event["exact_candidate_count"], 2)
        self.assertEqual(event["prompt_characters"], 1000)
        self.assertEqual(event["dictionary_entries_characters"], 250)
        self.assertEqual(event["grammar_characters"], 500)
        self.assertEqual(event["llm_call_ms"], 900)

    def test_llm_without_evidence_is_direct(self):
        metrics = derive_translation_path_metrics(
            "llm",
            {
                "used_llm": True,
                "has_dictionary_entries": False,
                "used_grammar": False,
                "used_exact_candidate_shortlist": False,
            },
        )

        self.assertEqual(metrics["translation_path"], "llm_direct")
        self.assertTrue(metrics["used_llm"])
        self.assertFalse(metrics["used_evidence_bundle"])

    def test_non_llm_paths_keep_response_source_as_path(self):
        event = build_translation_event(
            source_text="source",
            target_text="target",
            source_language="english",
            target_language="mingrelian",
            provider="gemini",
            model="gemini-test",
            duration_ms=15,
            response_source="dictionary_google_bridge",
            used_user_api_key=False,
            prompt_metrics={
                "used_llm": False,
                "method": "dictionary+google_translate",
            },
        )

        self.assertEqual(event["translation_path"], "dictionary_google_bridge")
        self.assertFalse(event["used_llm"])
        self.assertFalse(event["used_evidence_bundle"])


if __name__ == "__main__":
    unittest.main()
