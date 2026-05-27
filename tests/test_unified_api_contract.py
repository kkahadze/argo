import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import fastapi_app.api as api


class UnifiedApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_chat_rejects_unsupported_cross_low_resource_pair(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "prompt": "test",
                "source_language": "mingrelian",
                "target_language": "tsova_tush",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not supported", response.json()["detail"].lower())

    def test_streamed_result_exposes_shared_response_shape_for_bats(self) -> None:
        with patch.object(
            api,
            "single_call_translate",
            return_value={
                "translation": "daqˁoⁿ kʼuitʼ0",
                "full_response": "daqˁoⁿ kʼuitʼ0",
                "response_source": "llm",
                "prompt_metrics": {},
            },
        ):
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "openai",
                    "LLM_MODEL": "gpt-5.5",
                    "SUPABASE_LOGGING_ENABLED": "false",
                },
                clear=False,
            ):
                response = self.client.post(
                    "/chat",
                    json={
                        "prompt": "big cat",
                        "source_language": "english",
                        "target_language": "tsova_tush",
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(events, response.text)

        result = events[-1]["result"]
        self.assertEqual(result["source_language"], "english")
        self.assertEqual(result["target_language"], "tsova_tush")
        self.assertEqual(result["translated_text"], "daqqoⁿ kʼuitʼ")
        self.assertEqual(result["romanized_text"], "daqqoⁿ kʼuitʼ")
        self.assertEqual(result["target_text"], "daqqoⁿ kʼuitʼ")
        self.assertEqual(result["tsova_tush_latinized"], "daqqoⁿ kʼuitʼ")

    def test_streamed_result_exposes_svan_target_without_georgian_fallback(self) -> None:
        with patch.object(
            api,
            "single_call_translate",
            return_value={
                "translation": "ლაშხ",
                "full_response": "ლაშხ",
                "response_source": "llm",
                "prompt_metrics": {},
            },
        ):
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "openai",
                    "LLM_MODEL": "gpt-5.5",
                    "SUPABASE_LOGGING_ENABLED": "false",
                },
                clear=False,
            ):
                response = self.client.post(
                    "/chat",
                    json={
                        "prompt": "name",
                        "source_language": "english",
                        "target_language": "svan",
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        result = events[-1]["result"]
        self.assertEqual(result["source_language"], "english")
        self.assertEqual(result["target_language"], "svan")
        self.assertEqual(result["translated_text"], "ლაშხ")
        self.assertEqual(result["target_text"], "ლაშხ")
        self.assertEqual(result["svan"], "ლაშხ")
        self.assertEqual(result["georgian"], "")


if __name__ == "__main__":
    unittest.main()
