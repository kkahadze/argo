#!/usr/bin/env python3
"""Focused checks for canonical provider configuration."""
import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import provider_config


class ProviderConfigTests(unittest.TestCase):
    def _fake_dotenv_module(self):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda *args, **kwargs: None
        return fake_dotenv

    def _import_api_with_lightweight_deps(self):
        class FakeFastAPI:
            def __init__(self, *args, **kwargs):
                self.middleware = []

            def add_middleware(self, *args, **kwargs):
                self.middleware.append((args, kwargs))

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

        class FakeHTTPException(Exception):
            def __init__(self, status_code, detail):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FakeBaseModel:
            def __init__(self, **data):
                for name in self.__class__.__annotations__:
                    if name in data:
                        value = data.pop(name)
                    elif name in self.__class__.__dict__:
                        value = getattr(self.__class__, name)
                    else:
                        raise TypeError(f"Missing required field: {name}")
                    setattr(self, name, value)
                for name, value in data.items():
                    setattr(self, name, value)

        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = FakeFastAPI
        fake_fastapi.HTTPException = FakeHTTPException
        fake_fastapi.Request = type("Request", (), {})

        fake_cors = types.ModuleType("fastapi.middleware.cors")
        fake_cors.CORSMiddleware = type("CORSMiddleware", (), {})

        fake_responses = types.ModuleType("fastapi.responses")
        fake_responses.StreamingResponse = type(
            "StreamingResponse",
            (),
            {"__init__": lambda self, *args, **kwargs: None},
        )

        fake_pydantic = types.ModuleType("pydantic")
        fake_pydantic.BaseModel = FakeBaseModel

        fake_requests = types.ModuleType("requests")

        modules = {
            "dotenv": self._fake_dotenv_module(),
            "fastapi": fake_fastapi,
            "fastapi.middleware": types.ModuleType("fastapi.middleware"),
            "fastapi.middleware.cors": fake_cors,
            "fastapi.responses": fake_responses,
            "pydantic": fake_pydantic,
            "requests": fake_requests,
        }

        previous_api = sys.modules.pop("fastapi_app.api", None)
        try:
            with patch.dict(sys.modules, modules):
                return importlib.import_module("fastapi_app.api")
        finally:
            sys.modules.pop("fastapi_app.api", None)
            if previous_api is not None:
                sys.modules["fastapi_app.api"] = previous_api

    def _load_eval_provider_with_fake_dotenv(self):
        module_name = "eval_provider_under_test"
        previous_module = sys.modules.pop(module_name, None)
        provider_path = PROJECT_ROOT / "eval" / "provider.py"
        spec = importlib.util.spec_from_file_location(module_name, provider_path)
        module = importlib.util.module_from_spec(spec)
        try:
            with patch.dict(sys.modules, {"dotenv": self._fake_dotenv_module()}):
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            return module
        finally:
            if previous_module is not None:
                sys.modules[module_name] = previous_module

    def test_public_defaults_are_unchanged(self):
        self.assertEqual(provider_config.DEFAULT_PROVIDER, "openai")
        self.assertEqual(
            provider_config.SUPPORTED_PROVIDERS,
            ("openai", "anthropic", "gemini"),
        )
        self.assertEqual(
            provider_config.VALID_LANGUAGES,
            ("mingrelian", "georgian", "english"),
        )
        self.assertEqual(provider_config.DEFAULT_SOURCE_LANGUAGE, "mingrelian")
        self.assertEqual(provider_config.DEFAULT_TARGET_LANGUAGE, "english")
        self.assertEqual(
            provider_config.DEFAULT_MODEL_BY_PROVIDER,
            {
                "openai": "gpt-5.5",
                "anthropic": "claude-sonnet-4-5-20250929",
                "gemini": "gemini-3.1-flash-lite-preview",
            },
        )
        self.assertEqual(
            provider_config.PROVIDER_API_KEY_ENV,
            {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gemini": "GEMINI_API_KEY",
            },
        )
        self.assertEqual(
            provider_config.DEFAULT_REASONING_EFFORT_BY_MODEL,
            {"gpt-5.5": "none"},
        )

    def test_server_key_allowlist_is_centralized(self):
        self.assertEqual(
            provider_config.SERVER_KEY_MODELS,
            {
                "openai": frozenset({"gpt-5.5", "gpt-5.4-nano"}),
                "gemini": frozenset({"gemini-3.1-flash-lite-preview"}),
            },
        )

        for provider, models in provider_config.SERVER_KEY_MODELS.items():
            self.assertIn(provider, provider_config.SUPPORTED_PROVIDERS)
            for model in models:
                self.assertTrue(
                    provider_config.is_server_key_model_allowed(provider, model)
                )

        self.assertFalse(
            provider_config.is_server_key_model_allowed(
                "anthropic",
                provider_config.DEFAULT_MODEL_BY_PROVIDER["anthropic"],
            )
        )

    def test_helpers_read_from_canonical_maps(self):
        self.assertIn(
            provider_config.DEFAULT_SOURCE_LANGUAGE,
            provider_config.VALID_LANGUAGES,
        )
        self.assertIn(
            provider_config.DEFAULT_TARGET_LANGUAGE,
            provider_config.VALID_LANGUAGES,
        )

        for provider in provider_config.SUPPORTED_PROVIDERS:
            self.assertEqual(
                provider_config.get_default_model_for_provider(provider),
                provider_config.DEFAULT_MODEL_BY_PROVIDER[provider],
            )
            self.assertEqual(
                provider_config.get_api_key_env_var(provider),
                provider_config.PROVIDER_API_KEY_ENV[provider],
            )

        self.assertIsNone(provider_config.get_default_model_for_provider("unknown"))
        self.assertIsNone(provider_config.get_api_key_env_var("unknown"))
        self.assertEqual(
            provider_config.get_default_reasoning_effort_for_model("gpt-5.5"),
            "none",
        )
        self.assertIsNone(
            provider_config.get_default_reasoning_effort_for_model("gpt-5.4-nano")
        )
        self.assertIsNone(provider_config.get_default_reasoning_effort_for_model(None))

    def test_llm_client_uses_canonical_config(self):
        with patch.dict(sys.modules, {"dotenv": self._fake_dotenv_module()}):
            from src import llm_client

        self.assertIs(
            llm_client.DEFAULT_MODEL_BY_PROVIDER,
            provider_config.DEFAULT_MODEL_BY_PROVIDER,
        )
        self.assertIs(llm_client.LLMProvider, provider_config.LLMProvider)
        self.assertIs(
            llm_client.get_api_key_env_var,
            provider_config.get_api_key_env_var,
        )

    def test_llm_client_reads_api_keys_from_canonical_env_mapping(self):
        fake_openai = types.ModuleType("openai")

        fake_anthropic = types.ModuleType("anthropic")
        fake_anthropic.Anthropic = lambda api_key: types.SimpleNamespace(
            api_key=api_key
        )

        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = lambda api_key: types.SimpleNamespace(api_key=api_key)
        fake_google.genai = fake_genai

        modules = {
            "dotenv": self._fake_dotenv_module(),
            "openai": fake_openai,
            "anthropic": fake_anthropic,
            "google": fake_google,
            "google.genai": fake_genai,
        }

        with patch.dict(sys.modules, modules):
            from src import llm_client

            for provider in provider_config.SUPPORTED_PROVIDERS:
                env_var = provider_config.get_api_key_env_var(provider)
                with patch.dict(
                    os.environ,
                    {env_var: f"{provider}-test-key"},
                    clear=True,
                ):
                    client = llm_client.LLMClient(provider=provider)

                self.assertEqual(client.api_key, f"{provider}-test-key")
                self.assertEqual(
                    client.model,
                    provider_config.DEFAULT_MODEL_BY_PROVIDER[provider],
                )
                self.assertEqual(
                    client.reasoning_effort,
                    provider_config.get_default_reasoning_effort_for_model(client.model),
                )

    def test_api_uses_canonical_defaults_and_key_mapping(self):
        api = self._import_api_with_lightweight_deps()

        request_data = api.PromptIn(prompt="მა")
        self.assertIsNone(request_data.reasoning_effort)
        self.assertEqual(
            request_data.source_language,
            provider_config.DEFAULT_SOURCE_LANGUAGE,
        )
        self.assertEqual(
            request_data.target_language,
            provider_config.DEFAULT_TARGET_LANGUAGE,
        )
        self.assertEqual(
            api.stream_translation.__defaults__[:2],
            (
                provider_config.DEFAULT_SOURCE_LANGUAGE,
                provider_config.DEFAULT_TARGET_LANGUAGE,
            ),
        )

        env_var = provider_config.get_api_key_env_var("gemini")
        with patch.dict(os.environ, {env_var: "gemini-test-key"}, clear=True):
            self.assertEqual(api.get_server_api_key("gemini"), "gemini-test-key")
        self.assertIsNone(api.get_server_api_key("unknown"))

    def test_api_passes_reasoning_effort_to_lazy_llm_client(self):
        api = self._import_api_with_lightweight_deps()
        captured = {}

        class FakeLLMClient:
            def __init__(self, provider, model, api_key, reasoning_effort=None):
                captured["provider"] = provider
                captured["model"] = model
                captured["api_key"] = api_key
                captured["reasoning_effort"] = reasoning_effort
                self.model = model

            def complete(self, prompt, system_prompt=None):
                captured["prompt"] = prompt
                captured["system_prompt"] = system_prompt
                return "ok"

        with patch.object(api, "resolve_api_key_for_llm", return_value="server-key"), patch.object(
            api, "LLMClient", FakeLLMClient
        ):
            client = api.LazyLLMClient(
                provider="openai",
                model="gpt-5.5",
                api_key=None,
                reasoning_effort="none",
            )
            self.assertEqual(client.complete("hello", system_prompt="sys"), "ok")

        self.assertEqual(captured["provider"], "openai")
        self.assertEqual(captured["model"], "gpt-5.5")
        self.assertEqual(captured["api_key"], "server-key")
        self.assertEqual(captured["reasoning_effort"], "none")
        self.assertEqual(captured["prompt"], "hello")
        self.assertEqual(captured["system_prompt"], "sys")

    def test_eval_provider_uses_provider_default_model_for_provider_only_config(self):
        eval_provider = self._load_eval_provider_with_fake_dotenv()
        captured = {}

        class FakeLLMClient:
            def __init__(
                self,
                provider,
                api_key,
                model,
                temperature,
                max_tokens,
                reasoning_effort=None,
            ):
                captured["provider"] = provider
                captured["api_key"] = api_key
                captured["model"] = model
                captured["temperature"] = temperature
                captured["max_tokens"] = max_tokens
                captured["reasoning_effort"] = reasoning_effort
                self.model = model

        def fake_translate(input_text, source_lang, target_lang, llm_client, grammar_policy=None):
            captured["input_text"] = input_text
            captured["source_language"] = source_lang
            captured["target_language"] = target_lang
            captured["grammar_policy"] = grammar_policy
            return {
                "translation": "ok",
                "full_response": "full",
                "response_source": "test",
                "prompt_metrics": {"method": "test"},
            }

        env_var = provider_config.get_api_key_env_var("anthropic")
        with patch.object(eval_provider, "LLMClient", FakeLLMClient), patch.object(
            eval_provider, "translate", fake_translate
        ), patch.dict(os.environ, {env_var: "anthropic-test-key"}, clear=True):
            result = eval_provider.call_api("hello", {"provider": "anthropic"}, {})

        self.assertEqual(result["output"], "ok")
        self.assertEqual(captured["provider"], "anthropic")
        self.assertEqual(captured["api_key"], "anthropic-test-key")
        self.assertEqual(
            captured["model"],
            provider_config.DEFAULT_MODEL_BY_PROVIDER["anthropic"],
        )
        self.assertEqual(
            captured["source_language"],
            provider_config.DEFAULT_SOURCE_LANGUAGE,
        )
        self.assertEqual(
            captured["target_language"],
            provider_config.DEFAULT_TARGET_LANGUAGE,
        )
        self.assertIsNone(captured["reasoning_effort"])
        self.assertIsNone(captured["grammar_policy"])


if __name__ == "__main__":
    unittest.main()
