#!/usr/bin/env python3
"""
Run a grouped master-lexicon evaluation against the live translator.

This script samples unique inputs from the master lexicon in both directions:
- Mingrelian -> English
- English -> Mingrelian

For duplicated lexicon rows, it treats all grouped values as acceptable answers.
That means:
- a Mingrelian headword may map to multiple acceptable English glosses
- an English gloss may map to multiple acceptable Mingrelian headwords/headword_raw forms
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient
from src.provider_config import DEFAULT_MODEL_BY_PROVIDER, DEFAULT_PROVIDER
from src.single_call_translator import translate


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def load_grouped_cases(dataset_path: Path):
    rows = list(csv.DictReader(dataset_path.open("r", encoding="utf-8")))

    mingrelian_to_english: dict[str, set[str]] = defaultdict(set)
    english_to_mingrelian: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        headword = (row.get("headword") or "").strip()
        headword_raw = (row.get("headword_raw") or "").strip()
        translation = (row.get("translation") or "").strip()

        if headword and translation:
            mingrelian_to_english[headword].add(translation)

        if translation and headword:
            english_to_mingrelian[translation].add(headword)
            if headword_raw:
                english_to_mingrelian[translation].add(headword_raw)

    return mingrelian_to_english, english_to_mingrelian


def sample_cases(grouped: dict[str, set[str]], sample_size: int, seed: int):
    keys = sorted(grouped.keys())
    rng = random.Random(seed)
    if sample_size >= len(keys):
        selected = keys
    else:
        selected = rng.sample(keys, sample_size)
    return [(key, grouped[key]) for key in selected]


def matches_any(output: str, expected_values: Iterable[str]) -> bool:
    normalized_output = normalize_text(output)
    normalized_expected = {normalize_text(value) for value in expected_values if value.strip()}
    return normalized_output in normalized_expected


def evaluate_direction(
    *,
    cases: list[tuple[str, set[str]]],
    source_lang: str,
    target_lang: str,
    llm_client: LLMClient,
):
    results = []

    for input_text, acceptable_outputs in cases:
        result = translate(
            input_text=input_text,
            source_lang=source_lang,
            target_lang=target_lang,
            llm_client=llm_client,
        )
        output = result["translation"]
        passed = matches_any(output, acceptable_outputs)
        results.append(
            {
                "input": input_text,
                "output": output,
                "accepted_outputs": sorted(acceptable_outputs),
                "passed": passed,
            }
        )

    passed_count = sum(1 for item in results if item["passed"])
    total = len(results)
    return {
        "passed": passed_count,
        "total": total,
        "pass_rate": (passed_count / total) if total else 0.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="fastapi_app/data/master-lexicon-mkhedruli.csv",
        help="Path to the master lexicon CSV",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size per direction")
    parser.add_argument("--seed", type=int, default=20260412, help="Random seed")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER],
        help="Model name",
    )
    parser.add_argument(
        "--output",
        default="eval/results/master-lexicon-sampled-eval.json",
        help="Path for JSON results",
    )
    args = parser.parse_args()

    load_dotenv()

    dataset_path = Path(args.dataset)
    mingrelian_to_english, english_to_mingrelian = load_grouped_cases(dataset_path)

    m2e_cases = sample_cases(mingrelian_to_english, args.sample_size, args.seed)
    e2m_cases = sample_cases(english_to_mingrelian, args.sample_size, args.seed + 1)

    llm_client = LLMClient(provider=args.provider, model=args.model)

    mingrelian_to_english_eval = evaluate_direction(
        cases=m2e_cases,
        source_lang="mingrelian",
        target_lang="english",
        llm_client=llm_client,
    )
    english_to_mingrelian_eval = evaluate_direction(
        cases=e2m_cases,
        source_lang="english",
        target_lang="mingrelian",
        llm_client=llm_client,
    )

    overall_passed = (
        mingrelian_to_english_eval["passed"] + english_to_mingrelian_eval["passed"]
    )
    overall_total = mingrelian_to_english_eval["total"] + english_to_mingrelian_eval["total"]

    payload = {
        "dataset": str(dataset_path),
        "sample_size_per_direction": args.sample_size,
        "seed": args.seed,
        "provider": args.provider,
        "model": args.model,
        "mingrelian_to_english": mingrelian_to_english_eval,
        "english_to_mingrelian": english_to_mingrelian_eval,
        "overall": {
            "passed": overall_passed,
            "total": overall_total,
            "pass_rate": (overall_passed / overall_total) if overall_total else 0.0,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload["overall"], ensure_ascii=False))
    print(
        json.dumps(
            {
                "mingrelian_to_english": {
                    "passed": mingrelian_to_english_eval["passed"],
                    "total": mingrelian_to_english_eval["total"],
                    "pass_rate": mingrelian_to_english_eval["pass_rate"],
                },
                "english_to_mingrelian": {
                    "passed": english_to_mingrelian_eval["passed"],
                    "total": english_to_mingrelian_eval["total"],
                    "pass_rate": english_to_mingrelian_eval["pass_rate"],
                },
            },
            ensure_ascii=False,
        )
    )
    print(f"Saved detailed results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
