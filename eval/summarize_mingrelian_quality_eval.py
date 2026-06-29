#!/usr/bin/env python3
"""Summarize continuous Mingrelian Promptfoo quality results by direction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


DIRECTIONS = (
    "english-to-mingrelian",
    "georgian-to-mingrelian",
    "mingrelian-to-english",
    "mingrelian-to-georgian",
)


def _results(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if isinstance(results, dict):
        rows = results.get("results", [])
    else:
        rows = []
    if not isinstance(rows, list):
        raise ValueError(f"Promptfoo results missing from {path}")
    return rows


def _metric(row: dict, name: str) -> float:
    named = row.get("gradingResult", {}).get("namedScores", {})
    return float(named.get(name, 0) or 0)


def _row_key(row: dict) -> str:
    vars_ = row.get("vars", {})
    return str(vars_.get("case_id") or f"{vars_.get('direction')}:{vars_.get('input_text')}")


def _summary(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("vars", {}).get("direction", "unknown"))].append(row)

    summary: dict[str, dict] = {}
    for direction in DIRECTIONS:
        direction_rows = groups.get(direction, [])
        scores = [float(row.get("gradingResult", {}).get("score", 0) or 0) for row in direction_rows]
        summary[direction] = {
            "rows": len(direction_rows),
            "mean": mean(scores) if scores else 0,
            "passes": sum(bool(row.get("gradingResult", {}).get("pass")) for row in direction_rows),
            "script_rate": mean(
                [_metric(row, "target_script_and_format") for row in direction_rows]
            ) if direction_rows else 0,
            "reference_rate": mean(
                [_metric(row, "reference_form_diagnostic") for row in direction_rows]
            ) if direction_rows else 0,
            "token_coverage": mean(
                [_metric(row, "expected_token_coverage") for row in direction_rows]
            ) if direction_rows else 0,
        }
    return summary


def _print_summary(label: str, rows: list[dict]) -> None:
    print(f"## {label}")
    print()
    print("| Direction | Rows | Mean quality | Full passes | Expected-token coverage | Script/format | Reference form |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for direction, values in _summary(rows).items():
        print(
            f"| {direction} | {values['rows']} | {values['mean']:.3f} | "
            f"{values['passes']}/{values['rows']} | {values['token_coverage']:.3f} | "
            f"{values['script_rate']:.3f} | "
            f"{values['reference_rate']:.3f} |"
        )
    print()


def _print_comparison(baseline: list[dict], candidate: list[dict]) -> None:
    baseline_by_key = {_row_key(row): row for row in baseline}
    candidate_by_key = {_row_key(row): row for row in candidate}
    print("## Paired Delta")
    print()
    print("| Direction | Shared rows | Mean delta | Improved | Worse | Tie |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for direction in DIRECTIONS:
        deltas = []
        for key, candidate_row in candidate_by_key.items():
            if candidate_row.get("vars", {}).get("direction") != direction:
                continue
            baseline_row = baseline_by_key.get(key)
            if not baseline_row:
                continue
            candidate_score = float(candidate_row.get("gradingResult", {}).get("score", 0) or 0)
            baseline_score = float(baseline_row.get("gradingResult", {}).get("score", 0) or 0)
            deltas.append(candidate_score - baseline_score)
        improved = sum(delta > 0.025 for delta in deltas)
        worse = sum(delta < -0.025 for delta in deltas)
        ties = len(deltas) - improved - worse
        print(
            f"| {direction} | {len(deltas)} | "
            f"{(mean(deltas) if deltas else 0):+.3f} | {improved} | {worse} | {ties} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path, help="Promptfoo quality result JSON")
    parser.add_argument("--baseline", type=Path, help="Optional paired baseline JSON")
    args = parser.parse_args()

    candidate = _results(args.candidate)
    if args.baseline:
        baseline = _results(args.baseline)
        _print_summary("Baseline", baseline)
        _print_summary("Candidate", candidate)
        _print_comparison(baseline, candidate)
    else:
        _print_summary("Quality Eval", candidate)


if __name__ == "__main__":
    main()
