#!/usr/bin/env python3
"""
Compare prompt grammar policies for the lesson-note translation evals.

Use --measure-only for a no-LLM prompt-size pass. Without --measure-only this
runs promptfoo for each selected direction/policy and summarizes quality plus
prompt-size metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "eval" / "results" / "grammar-policy"
PROMPTFOO_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
DATASET = PROJECT_ROOT / "eval" / "datasets" / "notion-mingrelian-lesson-notes-triples.csv"

CONFIGS = {
    "mingrelian_to_english": "eval/promptfooconfig.lesson-notes.mingrelian-to-english.yaml",
    "english_to_mingrelian": "eval/promptfooconfig.lesson-notes.english-to-mingrelian.yaml",
    "mingrelian_to_georgian": "eval/promptfooconfig.lesson-notes.mingrelian-to-georgian.yaml",
    "georgian_to_mingrelian": "eval/promptfooconfig.lesson-notes.georgian-to-mingrelian.yaml",
}

PROMPT_BUILDERS = {
    "mingrelian_to_english": ("mingrelian", "construct_prompt_from_mingrelian_to_english"),
    "english_to_mingrelian": ("english", "construct_prompt_from_english_to_mingrelian"),
    "mingrelian_to_georgian": ("mingrelian", "construct_prompt_from_mingrelian_to_georgian"),
    "georgian_to_mingrelian": ("georgian", "construct_prompt_from_georgian_to_mingrelian"),
}

POLICIES = ("full", "compact", "none")


def _parse_csv_list(value: str | None, allowed: set[str], label: str) -> list[str]:
    if not value:
        return sorted(allowed)
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [part for part in parsed if part not in allowed]
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return parsed


def _median(values: list[int]) -> int | None:
    if not values:
        return None
    return int(statistics.median(values))


def _metric(payload: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    return response.get("metadata") or row.get("metadata") or {}


def _summarize_prompt_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars = [
        value
        for item in metrics
        if (value := _metric(item, "prompt_chars", "prompt_characters")) is not None
    ]
    dict_chars = [
        value
        for item in metrics
        if (value := _metric(item, "dict_entries_chars")) is not None
    ]
    grammar_chars = [
        value
        for item in metrics
        if (value := _metric(item, "grammar_chars")) is not None
    ]
    return {
        "prompt_chars_min": min(prompt_chars) if prompt_chars else None,
        "prompt_chars_median": _median(prompt_chars),
        "prompt_chars_max": max(prompt_chars) if prompt_chars else None,
        "dict_entries_chars_median": _median(dict_chars),
        "dict_entries_chars_max": max(dict_chars) if dict_chars else None,
        "grammar_chars_median": _median(grammar_chars),
        "grammar_chars_max": max(grammar_chars) if grammar_chars else None,
    }


def measure_prompts(directions: list[str], policies: list[str], *, allow_google_translate: bool) -> dict[str, Any]:
    """Build prompts without LLM calls and summarize prompt section sizes."""
    sys.path.insert(0, str(PROJECT_ROOT))
    import src.single_call_translator as translator

    if not allow_google_translate:
        translator.GoogleTranslator = None

    rows = list(csv.DictReader(DATASET.open("r", encoding="utf-8")))
    summary: dict[str, Any] = {}

    for direction in directions:
        source_column, builder_name = PROMPT_BUILDERS[direction]
        builder = getattr(translator, builder_name)
        summary[direction] = {}
        for policy in policies:
            metrics = []
            for row in rows:
                prompt = builder(row[source_column], grammar_policy=policy)
                metrics.append(translator._measure_prompt_sections(prompt))
            summary[direction][policy] = {
                "cases": len(metrics),
                **_summarize_prompt_metrics(metrics),
            }

    return summary


def summarize_promptfoo_output(result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload["results"]["results"]

    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        provider_label = (row.get("provider") or {}).get("label") or "unknown"
        by_provider[provider_label].append(row)

    provider_summary: dict[str, Any] = {}
    for provider_label, provider_rows in by_provider.items():
        successes = sum(1 for row in provider_rows if row.get("success"))
        prompt_metrics = [
            _row_metadata(row).get("prompt_metrics") or {}
            for row in provider_rows
        ]
        provider_summary[provider_label] = {
            "passed": successes,
            "total": len(provider_rows),
            "pass_rate": (successes / len(provider_rows)) if provider_rows else 0.0,
            **_summarize_prompt_metrics(prompt_metrics),
        }

    return {
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "stats": payload["results"].get("stats", {}),
        "providers": provider_summary,
    }


def run_eval(
    *,
    direction: str,
    config_path: str,
    policy: str,
    output_dir: Path,
    repeat: int,
) -> dict[str, Any]:
    output_path = output_dir / f"{direction}.{policy}.json"
    log_path = output_dir / f"{direction}.{policy}.log"

    if output_path.exists():
        print(f"[reuse] {direction} / {policy}")
        return summarize_promptfoo_output(output_path)

    env = os.environ.copy()
    env["PROMPTFOO_PYTHON"] = str(PROMPTFOO_PYTHON)
    env["ARGO_GRAMMAR_POLICY"] = policy

    cmd = [
        "promptfoo",
        "eval",
        "-c",
        config_path,
        "--env-path",
        ".env",
        "--repeat",
        str(repeat),
        "--no-cache",
        "--no-share",
        "--no-progress-bar",
        "--no-table",
        "--output",
        str(output_path),
        "--description",
        f"{direction} / grammar_policy={policy}",
    ]

    print(f"[run] {direction} / {policy}")
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(cmd),
                "",
                "STDOUT:",
                completed.stdout,
                "",
                "STDERR:",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Promptfoo run failed for {direction} / {policy}. "
            f"See {log_path.relative_to(PROJECT_ROOT)}"
        )

    if completed.returncode != 0:
        print(f"[warn] promptfoo exited with code {completed.returncode}; output JSON was written")

    return summarize_promptfoo_output(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", help="Comma-separated direction keys; default is all")
    parser.add_argument("--policies", help="Comma-separated policies: full,compact,none; default is all")
    parser.add_argument("--repeat", type=int, default=1, help="Promptfoo repeat count")
    parser.add_argument("--output-dir", help="Where to write/reuse results")
    parser.add_argument("--measure-only", action="store_true", help="Only build prompts; do not call LLMs")
    parser.add_argument(
        "--allow-google-translate",
        action="store_true",
        help="Allow deep-translator calls during --measure-only prompt construction",
    )
    args = parser.parse_args()

    directions = _parse_csv_list(args.directions, set(CONFIGS), "directions")
    policies = _parse_csv_list(args.policies, set(POLICIES), "policies")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = RESULTS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.measure_only:
        summary = {
            "created_at": datetime.now().isoformat(),
            "mode": "measure_only",
            "allow_google_translate": args.allow_google_translate,
            "directions": measure_prompts(
                directions,
                policies,
                allow_google_translate=args.allow_google_translate,
            ),
        }
    else:
        summary = {
            "created_at": datetime.now().isoformat(),
            "mode": "promptfoo",
            "repeat": args.repeat,
            "runs": {},
        }
        for direction in directions:
            summary["runs"][direction] = {}
            for policy in policies:
                summary["runs"][direction][policy] = run_eval(
                    direction=direction,
                    config_path=CONFIGS[direction],
                    policy=policy,
                    output_dir=output_dir,
                    repeat=args.repeat,
                )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] wrote {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
