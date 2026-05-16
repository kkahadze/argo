import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import fastapi_app.api as api
import src.single_call_translator as translator


class LazyCredentialTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def _post_events(self, payload):
        with patch.dict(
            os.environ,
            {
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
