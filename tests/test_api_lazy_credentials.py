import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import fastapi_app.api as api
from src import dictionary_store
import src.single_call_translator as translator


class LazyCredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self._write_test_data()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = TestClient(api.app)

    def _write_test_data(self):
        (self.data_dir / "sentence_pairs.tsv").write_text(
            "Mingrelian\tEnglish\n"
            "ჭიფანა კაკალეფ უღუუ.\tIt has smaller grapes.\n",
            encoding="utf-8",
        )
        (self.data_dir / "gal.tsv").write_text("Russian\tMingrelian\n", encoding="utf-8")
        (self.data_dir / "kk.tsv").write_text(
            "word\tipa\trussian_def\tgeorgian_def\n"
            "აბანა\tabana\tбаня\tაბანო 2. სამკურნალო წყლები\n",
            encoding="utf-8",
        )
        (self.data_dir / "translation_overrides.tsv").write_text(
            "source_language\ttarget_language\tsource_text\ttarget_text\n",
            encoding="utf-8",
        )
        (self.data_dir / "context_source.txt").write_text("", encoding="utf-8")
        (self.data_dir / "harris.txt").write_text("", encoding="utf-8")

    def _clear_caches(self):
        dictionary_store._get_dictionary_store_cached.cache_clear()
        translator._load_master_lexicon_rows_cached.cache_clear()
        translator._load_sentence_pairs_rows_cached.cache_clear()
        translator._load_gal_rows_cached.cache_clear()
        translator._load_kk_rows_cached.cache_clear()
        translator._load_context_source_entries_cached.cache_clear()
        translator._load_grammar_cached.cache_clear()

    def _post_events(self, payload):
        self._clear_caches()
        with patch.dict(
            os.environ,
            {
                "ARGO_DATA_DIR": str(self.data_dir),
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "GEMINI_API_KEY": "",
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "",
                "SUPABASE_LOGGING_ENABLED": "false",
            },
            clear=False,
        ):
            response = self.client.post("/chat", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(events, response.text)
        return events

    def test_exact_lexicon_match_does_not_require_credentials(self):
        with patch.object(api, "LLMClient", side_effect=AssertionError("LLMClient should not be constructed")):
            events = self._post_events(
                {
                    "prompt": "ჭიფანა კაკალეფ უღუუ.",
                    "source_language": "mingrelian",
                    "target_language": "english",
                    "provider": "openai",
                    "model": "not-on-server-allowlist",
                }
            )

        result = events[-1]["result"]
        self.assertEqual(result["target_text"], "It has smaller grapes.")
        self.assertEqual(result["english"], "It has smaller grapes.")

    def test_dictionary_google_bridge_does_not_require_credentials(self):
        fake_google = self._fake_google_translator(
            {
                ("en", "ka", "bathhouse"): "აბანო 2. სამკურნალო წყლები",
                ("en", "ru", "bathhouse"): "no matching russian gloss",
            }
        )

        with (
            patch.object(translator, "GoogleTranslator", fake_google),
            patch.object(api, "LLMClient", side_effect=AssertionError("LLMClient should not be constructed")),
        ):
            events = self._post_events(
                {
                    "prompt": "bathhouse",
                    "source_language": "english",
                    "target_language": "mingrelian",
                    "provider": "openai",
                    "model": "not-on-server-allowlist",
                }
            )

        result = events[-1]["result"]
        self.assertEqual(result["target_text"], "აბანა")
        self.assertEqual(result["mingrelian_mkhedruli"], "აბანა")

    def test_direct_google_translate_does_not_require_credentials(self):
        fake_google = self._fake_google_translator(
            {
                ("en", "ka", "hello"): "გამარჯობა",
            }
        )

        with (
            patch.object(translator, "GoogleTranslator", fake_google),
            patch.object(api, "LLMClient", side_effect=AssertionError("LLMClient should not be constructed")),
        ):
            events = self._post_events(
                {
                    "prompt": "hello",
                    "source_language": "english",
                    "target_language": "georgian",
                    "provider": "openai",
                    "model": "not-on-server-allowlist",
                }
            )

        result = events[-1]["result"]
        self.assertEqual(result["target_text"], "გამარჯობა")
        self.assertEqual(result["georgian"], "გამარჯობა")

    def test_llm_fallback_without_credentials_returns_sse_error(self):
        with patch.object(api, "LLMClient", side_effect=AssertionError("LLMClient should not be constructed")):
            events = self._post_events(
                {
                    "prompt": "zzzzzz unmatched phrase",
                    "source_language": "mingrelian",
                    "target_language": "english",
                    "provider": "openai",
                    "model": "not-on-server-allowlist",
                }
            )

        self.assertEqual(
            events[-1],
            {"error": "Model 'not-on-server-allowlist' requires a user-provided API key"},
        )

    @staticmethod
    def _fake_google_translator(mapping):
        class FakeGoogleTranslator:
            def __init__(self, source, target):
                self.source = source
                self.target = target

            def translate(self, text):
                return mapping[(self.source, self.target, text)]

        return FakeGoogleTranslator


if __name__ == "__main__":
    unittest.main()
