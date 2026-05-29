#!/usr/bin/env python3
"""Promptfoo provider that emits a supplied candidate for grader calibration."""


def call_api(prompt, options, context):
    return {"output": prompt}
