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


def extract_one(row: dict, rubric: str, model: str | None) -> dict:
    """One claude -p call -> {"url", "spec"} or {"url", "error"}."""
    page = ROOT / row["page"]
    cmd = ["claude", "-p"] + (["--model", model] if model else [])
    prompt = build_prompt(rubric, row, strip_body(page))

    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"url": row["url"], "error": f"claude CLI timed out after {CLI_TIMEOUT}s"}
    if proc.returncode != 0:
        return {"url": row["url"], "error": f"claude exited {proc.returncode}: {proc.stderr[:200]}"}

    try:
        spec = parse_spec(proc.stdout)
    except (ValueError, yaml.YAMLError) as exc:
        return {"url": row["url"], "error": f"unparseable YAML: {exc}"[:300]}

    errors = validate_spec(spec)
    if errors:
        return {"url": row["url"], "error": "invalid spec: " + "; ".join(errors)}
    return {"url": row["url"], "spec": spec}


def _write_atomic(path: Path, text: str) -> None:
    """Publish atomically -- downstream stages never observe a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def publish(row: dict, result: dict, specs_dir: Path) -> str:
    """Record one extract outcome under specs/. The artifact is the state --
    articles.jsonl is never rewritten here."""
    slug = row["slug"]
    if "error" in result:
        _write_atomic(specs_dir / f"{slug}.error",
                      json.dumps({"url": row["url"], "error": result["error"]}, indent=2))
        return "extract_failed"

    spec = result["spec"]
    if spec.get("verdict", {}).get("status") == "UNTESTABLE":
        _write_atomic(specs_dir / f"{slug}.untestable",
                      json.dumps({"url": row["url"],
                                  "reason": spec["verdict"].get("reason", "")}, indent=2))
        return "untestable"

    _write_atomic(specs_dir / f"{slug}.yaml", yaml.safe_dump(spec, sort_keys=False))
    return "spec"


def pending(rows: list[dict], specs_dir: Path, retry: bool) -> list[dict]:
    """Triaged articles with no extract outcome yet. A .error marker counts
    as an outcome unless retry is set; .untestable is always final."""
    done = {p.stem for p in specs_dir.glob("*.yaml")}
    done |= {p.stem for p in specs_dir.glob("*.untestable")}
    if not retry:
        done |= {p.stem for p in specs_dir.glob("*.error")}
    return [r for r in rows
            if r.get("stage") == "triaged" and r["slug"] not in done]


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent claude calls")
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--respec", action="store_true",
                    help="also retry slugs with a specs/<slug>.error marker")
    args = ap.parse_args(argv)

    rubric = PROMPT.read_text()
    rows = load(ARTICLES)  # read-only -- publish() never rewrites this file
    todo = pending(rows, SPECS, args.respec)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to extract -- run triage.py first, or pass --respec")
        return 0

    print(f"extracting {len(todo)} articles ({args.jobs} at a time)\n")
    by_url = {r["url"]: r for r in todo}
    stages: dict[str, int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(extract_one, r, rubric, args.model): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row = by_url[fut.result()["url"]]
            stage = publish(row, fut.result(), SPECS)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "spec" else "FAIL" if stage == "extract_failed" else "SKIP"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {row['title'][:50]}")

    print("\nstages:", ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    print(f"specs -> {SPECS.relative_to(ROOT)}/")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
