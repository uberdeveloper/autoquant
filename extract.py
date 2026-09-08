#!/usr/bin/env python3
"""Stage [4] EXTRACT -- convert one triaged article into a strategy spec.

Runs extract_prompt.md over every triaged article via the `claude` CLI in
headless mode (-p), reusing your Claude Code auth. A valid reply is published
to specs/<slug>.yaml. An UNTESTABLE reply writes specs/<slug>.untestable; an
invalid reply writes specs/<slug>.error. Nothing is deleted.

Independent and resumable: articles.jsonl is READ-ONLY here -- the spec file
itself is the state. Re-running skips every slug that already has an outcome.

    python3 extract.py --limit 5    # calibrate against your own reading
    python3 extract.py              # the whole triaged queue
    python3 extract.py --respec     # also retry slugs with a .error marker
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path

import yaml

from triage import load, strip_body

ROOT = Path(__file__).resolve().parent
ARTICLES = ROOT / "data" / "articles.jsonl"
SPECS = ROOT / "specs"
PROMPT = ROOT / "extract_prompt.md"

CLI_TIMEOUT = 600
MAX_ARTICLE_CHARS = 40_000  # same budget triage uses
REQUIRED_SECTIONS = ("meta", "data", "signal", "rules", "costs",
                     "validation", "verdict", "ambiguities")


def parse_spec(text: str) -> dict:
    """The rubric asks for bare YAML; tolerate a fenced reply."""
    text = text.strip()
    fence = re.search(r"```(?:yaml)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError("reply is not a YAML mapping")
    return doc


def validate_spec(doc: dict) -> list[str]:
    """Structural checks every extracted spec must pass. [] = good."""
    errors: list[str] = []

    verdict = doc.get("verdict") or {}
    if verdict.get("status") == "UNTESTABLE":
        if not verdict.get("reason"):
            errors.append("UNTESTABLE verdict needs a reason")
        return errors

    for section in REQUIRED_SECTIONS:
        if section not in doc:
            errors.append(f"missing section: {section}")
    if errors:
        return errors

    meta, signal, costs = doc["meta"], doc["signal"], doc["costs"]
    for key in ("slug", "title", "url", "posted", "claim", "author_evidence"):
        if not meta.get(key):
            errors.append(f"meta.{key} is empty")
    if int(signal.get("lag_bars") or 0) < 1:
        errors.append("signal.lag_bars must be >= 1")
    if (costs.get("commission_bps") or 0) + (costs.get("slippage_bps") or 0) <= 0:
        errors.append("costs must be nonzero")
    if not doc["ambiguities"]:
        errors.append("ambiguities must not be empty")
    if doc["validation"].get("oos_start") != meta.get("posted"):
        errors.append("validation.oos_start must equal meta.posted")
    return errors


def build_prompt(rubric: str, row: dict, body: str) -> str:
    if len(body) > MAX_ARTICLE_CHARS:
        body = body[:MAX_ARTICLE_CHARS] + "\n\n[...truncated for length...]"
    payload = {"title": row["title"], "source": row["source"], "url": row["url"],
               "posted": row["posted"], "text": body}
    return (
        f"{rubric}\n\n"
        "---\n\nHere is the article to extract. Output YAML only -- no "
        "markdown fence, no prose.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
