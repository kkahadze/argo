#!/usr/bin/env python3
"""
Helpers for writing translation analytics events to Supabase without
blocking the translation response path.
"""
import asyncio
import os
from typing import Any, Optional
from urllib.parse import quote

import requests

from src.logger import setup_logger


logger = setup_logger("translation_analytics")


def _coerce_bool(value: Any) -> bool:
    """Coerce JSON-ish metric values into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce JSON-ish metric values into integers when possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a truthy/falsey environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_supabase_api_key() -> Optional[str]:
    """Prefer server credentials, but allow a publishable key for append-only logging."""
    return (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
    )


def analytics_enabled() -> bool:
    """Return whether analytics are fully configured."""
    return bool(
        _env_flag("SUPABASE_LOGGING_ENABLED")
        and os.getenv("SUPABASE_URL")
        and _get_supabase_api_key()
    )


def _analytics_url() -> Optional[str]:
    """Build the PostgREST endpoint for the configured analytics table."""
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    table = os.getenv("SUPABASE_LOGGING_TABLE", "translation_events").strip()
    if not supabase_url or not table:
        return None
    return f"{supabase_url}/rest/v1/{quote(table, safe='')}"


def infer_response_source(result: Optional[dict[str, Any]] = None) -> str:
    """Classify how a translation was produced for later analysis."""
    if not result:
        return "unknown"

    response_source = (result.get("response_source") or "").strip()
    if response_source:
        return response_source

    prompt_metrics = result.get("prompt_metrics") or {}
    if prompt_metrics.get("method") == "exact_lexicon":
        return "exact_lexicon"
    if prompt_metrics.get("method") in {"dictionary+google_translate", "dictionary_google_bridge"}:
        return "dictionary_google_bridge"
    if prompt_metrics.get("reason") == "google_translate_direct":
        return "google_translate_direct"
    if prompt_metrics.get("reason") == "instant_lookup":
        return "dictionary_google_bridge"

    full_response = (result.get("full_response") or "").strip()
    if full_response.startswith("Exact lexicon match:"):
        return "exact_lexicon"
    if full_response.startswith("Dictionary match (via Google Translate bridge):"):
        return "dictionary_google_bridge"
    if full_response.startswith("Translation (via Google Translate):"):
        return "google_translate_direct"
    if result.get("translation"):
        return "llm"
    return "unknown"


def derive_translation_path_metrics(
    response_source: Optional[str],
    prompt_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Derive query-friendly path fields from response_source and prompt metrics."""
    metrics = prompt_metrics or {}
    source = (response_source or "unknown").strip() or "unknown"

    used_llm = _coerce_bool(metrics.get("used_llm")) or source == "llm"
    used_dictionary_entries = _coerce_bool(metrics.get("has_dictionary_entries"))
    used_grammar = _coerce_bool(metrics.get("used_grammar") or metrics.get("grammar_included"))
    used_exact_candidate_shortlist = _coerce_bool(metrics.get("used_exact_candidate_shortlist"))
    used_evidence_bundle = bool(
        used_llm
        and (
            used_dictionary_entries
            or used_grammar
            or used_exact_candidate_shortlist
        )
    )

    if source == "llm":
        if used_evidence_bundle:
            translation_path = "llm_evidence_bundle"
        else:
            translation_path = "llm_direct"
    else:
        translation_path = source

    return {
        "translation_path": translation_path,
        "used_llm": used_llm,
        "used_evidence_bundle": used_evidence_bundle,
        "used_dictionary_entries": used_dictionary_entries,
        "used_grammar": used_grammar,
        "used_exact_candidate_shortlist": used_exact_candidate_shortlist,
        "exact_candidate_count": _coerce_int(metrics.get("exact_candidate_count")),
        "prompt_characters": _coerce_int(
            metrics.get("prompt_characters") or metrics.get("prompt_chars")
        ),
        "dictionary_entries_characters": _coerce_int(metrics.get("dict_entries_chars")),
        "grammar_characters": _coerce_int(metrics.get("grammar_chars")),
        "llm_call_ms": _coerce_int(metrics.get("llm_call_ms")),
    }


def build_translation_event(
    *,
    source_text: str,
    target_text: Optional[str],
    source_language: str,
    target_language: str,
    provider: Optional[str],
    model: Optional[str],
    duration_ms: Optional[int],
    response_source: str,
    used_user_api_key: bool,
    prompt_metrics: Optional[dict[str, Any]] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    app_origin: Optional[str] = None,
    referer: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> dict[str, Any]:
    """Build a JSON-safe analytics payload for Supabase."""
    cleaned_source = source_text or ""
    cleaned_target = target_text or None
    cleaned_response_source = response_source or "unknown"
    path_metrics = derive_translation_path_metrics(cleaned_response_source, prompt_metrics)
    return {
        "status": status,
        "source_text": cleaned_source,
        "target_text": cleaned_target,
        "source_language": source_language,
        "target_language": target_language,
        "provider": provider,
        "model": model,
        "response_source": cleaned_response_source,
        **path_metrics,
        "duration_ms": duration_ms,
        "source_text_length": len(cleaned_source),
        "target_text_length": len(cleaned_target) if cleaned_target is not None else None,
        "used_user_api_key": used_user_api_key,
        "prompt_metrics": prompt_metrics or {},
        "error_message": error_message,
        "app_origin": app_origin,
        "referer": referer,
        "user_agent": user_agent,
    }


def write_translation_event(payload: dict[str, Any]) -> None:
    """Send a single analytics event to Supabase."""
    if not analytics_enabled():
        return

    endpoint = _analytics_url()
    api_key = _get_supabase_api_key()
    timeout_seconds = float(os.getenv("SUPABASE_LOGGING_TIMEOUT_SECONDS", "2.5"))

    if not endpoint or not api_key:
        return

    headers = {
        "Content-Type": "application/json",
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Prefer": "return=minimal",
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to write translation analytics event: %s", exc)


def _log_background_task_result(task: "asyncio.Task[None]") -> None:
    """Report unexpected background task failures without surfacing them to users."""
    try:
        task.result()
    except Exception as exc:
        logger.warning("Translation analytics background task failed: %s", exc)


def schedule_translation_event(payload: dict[str, Any]) -> None:
    """
    Schedule analytics delivery off the hot path.

    If there is no running event loop, fall back to a direct write.
    """
    if not analytics_enabled():
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        write_translation_event(payload)
        return

    task = loop.create_task(asyncio.to_thread(write_translation_event, payload))
    task.add_done_callback(_log_background_task_result)
