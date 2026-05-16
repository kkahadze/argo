#!/usr/bin/env python3
"""
Run a lesson-notes ablation that compares the translation pipeline with and
without the master lexicon across all four directions and both translation
models already defined in the promptfoo configs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "eval" / "results" / "master-lexicon-ablation"
PROMPTFOO_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

CONFIGS = {
    "mingrelian_to_english": "eval/promptfooconfig.lesson-notes.mingrelian-to-english.yaml",
    "english_to_mingrelian": "eval/promptfooconfig.lesson-notes.english-to-mingrelian.yaml",
    "mingrelian_to_georgian": "eval/promptfooconfig.lesson-notes.mingrelian-to-georgian.yaml",
    "georgian_to_mingrelian": "eval/promptfooconfig.lesson-notes.georgian-to-mingrelian.yaml",
}

CONDITIONS = {
    "master_on": "true",
    "master_off": "false",
}


def _case_key(result: dict[str, Any]) -> str:
    """Build a stable case key so repeated runs can be compared."""
    vars_payload = result.get("vars") or {}
    return json.dumps(
        {
            "prompt": ((result.get("prompt") or {}).get("raw")),
            "vars": vars_payload,
            "provider": (result.get("provider") or {}).get("label"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def summarize_output(result_path: Path) -> dict[str, Any]:
    """Summarize a single promptfoo JSON output by provider."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows = payload["results"]["results"]

    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        provider_label = (row.get("provider") or {}).get("label") or "unknown"
        by_provider[provider_label].append(row)

    provider_summary: dict[str, Any] = {}
    for provider_label, provider_rows in by_provider.items():
        successes = sum(1 for row in provider_rows if row.get("success"))
        totals = len(provider_rows)

        metric_totals: dict[str, float] = defaultdict(float)
        metric_counts: dict[str, int] = defaultdict(int)
        repeat_buckets: dict[str, list[bool]] = defaultdict(list)

        for row in provider_rows:
            for metric, value in (row.get("namedScores") or {}).items():
                if isinstance(value, (int, float)):
                    metric_totals[metric] += float(value)
                    metric_counts[metric] += 1

            repeat_buckets[_case_key(row)].append(bool(row.get("success")))

        repeat_consistent = sum(
            1
            for outcomes in repeat_buckets.values()
            if len(outcomes) > 1 and len(set(outcomes)) == 1
        )
        repeat_inconsistent = sum(
            1
            for outcomes in repeat_buckets.values()
            if len(outcomes) > 1 and len(set(outcomes)) > 1
        )

        provider_summary[provider_label] = {
            "passed": successes,
            "total": totals,
            "pass_rate": (successes / totals) if totals else 0.0,
            "named_metric_means": {
                metric: metric_totals[metric] / metric_counts[metric]
                for metric in sorted(metric_totals)
                if metric_counts[metric]
            },
            "repeat_consistent_cases": repeat_consistent,
            "repeat_inconsistent_cases": repeat_inconsistent,
            "unique_cases": len(repeat_buckets),
        }

    return {
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "stats": payload["results"]["stats"],
        "providers": provider_summary,
    }


def run_eval(config_key: str, config_path: str, condition_key: str, enabled_value: str, output_dir: Path) -> dict[str, Any]:
    """Run one promptfoo eval and return its summarized output."""
    output_path = output_dir / f"{config_key}.{condition_key}.json"
    log_path = output_dir / f"{config_key}.{condition_key}.log"

    if output_path.exists():
        print(f"[reuse] {config_key} / {condition_key}")
        return summarize_output(output_path)

    env = os.environ.copy()
    env["PROMPTFOO_PYTHON"] = str(PROMPTFOO_PYTHON)
    env["ARGO_ENABLE_MASTER_LEXICON"] = enabled_value

    cmd = [
        "promptfoo",
        "eval",
        "-c",
        config_path,
        "--env-path",
        ".env",
        "--repeat",
        "2",
        "--no-cache",
        "--no-share",
        "--no-progress-bar",
        "--no-table",
        "--output",
        str(output_path),
        "--description",
        f"{config_key} / {condition_key}",
    ]

    print(f"[run] {config_key} / {condition_key}")
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
            f"Promptfoo run failed for {config_key} / {condition_key}. "
            f"See {log_path.relative_to(PROJECT_ROOT)}"
        )

    if completed.returncode != 0:
        print(
            f"[warn] promptfoo exited with code {completed.returncode} for "
            f"{config_key} / {condition_key}, but output JSON was written"
        )

    return summarize_output(output_path)


def build_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare master-on vs master-off pass rates for each direction/provider pair."""
    comparisons: dict[str, Any] = {}
    for direction_key, direction_payload in summary["runs"].items():
        comparisons[direction_key] = {}
        on_providers = direction_payload["master_on"]["providers"]
        off_providers = direction_payload["master_off"]["providers"]
        for provider_label in sorted(set(on_providers) | set(off_providers)):
            on_stats = on_providers.get(provider_label, {})
            off_stats = off_providers.get(provider_label, {})
            on_rate = on_stats.get("pass_rate")
            off_rate = off_stats.get("pass_rate")
            if on_rate is None or off_rate is None:
                delta = None
            else:
                delta = on_rate - off_rate
            comparisons[direction_key][provider_label] = {
                "master_on_pass_rate": on_rate,
                "master_off_pass_rate": off_rate,
                "pass_rate_delta": delta,
                "master_on_passed": on_stats.get("passed"),
                "master_off_passed": off_stats.get("passed"),
                "master_on_total": on_stats.get("total"),
                "master_off_total": off_stats.get("total"),
            }
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="Reuse an existing output directory or specify where new results should go",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "repeat_count": 2,
        "dataset_rows": 27,
        "runs": {},
    }

    for config_key, config_path in CONFIGS.items():
        summary["runs"][config_key] = {}
        for condition_key, enabled_value in CONDITIONS.items():
            summary["runs"][config_key][condition_key] = run_eval(
                config_key,
                config_path,
                condition_key,
                enabled_value,
                output_dir,
            )

    summary["comparisons"] = build_comparison(summary)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {summary_path.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary["comparisons"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
