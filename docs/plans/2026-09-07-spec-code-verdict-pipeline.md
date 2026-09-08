# Spec → Code → Verdict Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the autoquant pipeline's remaining stages: turn each triaged article into a spec (`extract.py`), turn each spec into runnable strategy code (`codegen.py`), and backtest every spec — with the three stages fully independent, safely parallel, and resumable at any point.

**Architecture:** Three independent CLI drivers. **The artifact is the state**: every stage owns exactly one output directory, writes its outputs atomically (tmp file + rename), and derives its queue from what already exists on disk. `data/articles.jsonl` is read-only from extract onward — no stage rewrites it, so no two stages can ever clobber each other's bookkeeping. A spec file existing means *extracted*; a strategy file existing means *coded*; a leaderboard row existing means *backtested*. Re-running any stage skips completed work — resume is just "run it again." Failures are recorded as `.error` marker files next to the output, never deleted. `backtest.py` gains one small function that appends a row to `results/leaderboard.jsonl` per completed run.

**Tech Stack:** Python 3.11+, uv, pytest, PyYAML, pandas/numpy (existing deps — nothing new). LLM calls go through the `claude` CLI in `-p` (headless) mode, same as triage.py.

**Repo under change:** `~/Desktop/autoquant` (branch off `main`). As part of the PR, this plan document is committed into the autoquant repo at `docs/plans/2026-09-07-spec-code-verdict-pipeline.md` so it is versioned next to the code it changes (its working copy lives in `~/Desktop/quantocracy-lab/plan/`).

---

## Stage independence, parallelism, and resume

Core rule: **the artifact is the state.** Each stage reads upstream artifacts read-only, owns exactly one output directory, and publishes atomically — a half-written file is never visible to a downstream stage.

| Stage | Reads (read-only) | Owns & writes | Done marker | Failure marker |
|---|---|---|---|---|
| `extract.py` | `articles.jsonl` (triaged rows), `data/pages/` | `specs/` | `specs/<slug>.yaml` | `specs/<slug>.error`, `specs/<slug>.untestable` |
| `codegen.py` | `specs/*.yaml` | `strategies/` | `strategies/<slug>.py` | `strategies/<slug>.error` |
| `backtest_batch.py` | `specs/*.yaml`, `strategies/` | `reports/`, `results/` | row in `results/leaderboard.jsonl` | row in `results/failed.jsonl` |

**Guarantees:**

- **Parallel-safe:** output directories are disjoint, so `extract.py` and `codegen.py` can run simultaneously. `codegen.py` picks up each spec the moment it appears; atomic rename means it never reads a half-written spec.
- **Resumable:** every stage is idempotent — re-run skips anything with a done marker. `.error`/`.untestable` markers are outcomes too (skipped unless the retry flag is passed). A crashed or interrupted run resumes by simply re-running the same command.
- **No shared mutable queue:** nothing to lock, nothing to clobber. This is why `articles.jsonl` is never rewritten downstream — a shared rewritten queue file is the one design that breaks parallel stages.
- **No stage blocks indefinitely:** extract/codegen wrap every CLI call in a timeout; the batch runner caps each spec in its own subprocess.

## Pipeline context (where this fits)

```
data/articles.jsonl (read-only from extract onward)
   └─ stage: triaged (~25)                     ← written by harvest/fetch/triage (exist today)
        │
        ▼  extract.py  (parallel, timeout per call)          ← Tasks 1–3
specs/<slug>.yaml          (+ <slug>.error / <slug>.untestable markers)
        │
        ▼  codegen.py  (parallel, timeout per call)          ← Tasks 4–5
strategies/<slug>.py       (+ <slug>.error markers)
        │
        ▼  backtest_batch.py  (one isolated subprocess per spec, hard timeout)  ← Task 6
reports/<slug>.md  +  results/leaderboard.jsonl row                      ← Task 7
                           (+ results/failed.jsonl rows)
```

All three stages can run at the same time: extract publishes spec #9 while codegen is on spec #3 and the batch runner is backtesting spec #1. Each command is also the resume command.

Storage layout (what ends up in git vs. ignored):

```
autoquant/
├── specs/
│   ├── <slug>.yaml            # git — the plan/contract per testable article
│   ├── <slug>.untestable      # git — recorded, never deleted (json: url, reason)
│   └── <slug>.error           # git — recorded, never deleted (json: url, error)
├── strategies/
│   ├── <slug>.py              # git — the signal implementation (smoke-tested)
│   └── <slug>.error           # git — recorded, never deleted
├── results/
│   ├── leaderboard.jsonl      # git — one verdict row per completed run
│   └── failed.jsonl           # git — batch failures (spec path + error)
├── reports/                   # gitignored (already) — regenerable full reports
└── data/                      # gitignored (already) — articles.jsonl, pages/, prices/
```

---

### Task 1: `extract.py` — spec reply parsing and validation

**Files:**
- Create: `extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_extract.py
"""Tests for extract.py — spec parsing, validation, publishing."""
from __future__ import annotations

import pytest
import yaml

import extract


VALID_SPEC = {
    "meta": {
        "slug": "test-slug", "title": "T", "url": "https://a.com/x",
        "source": "S", "posted": "2026-08-09", "claim": "C",
        "author_evidence": {"sample": "s", "headline_metrics": "m",
                            "costs_included": False},
    },
    "data": {"universe": ["SPY"], "start": "1993-01-01"},
    "signal": {"definition": "close > sma200", "lag_bars": 1},
    "rules": {"direction": "long_only"},
    "costs": {"commission_bps": 1.0, "slippage_bps": 5.0},
    "validation": {"oos_start": "2026-08-09", "cost_sweep_bps": [0, 5]},
    "ambiguities": [{"id": "a1", "question": "q", "default": "d",
                     "alternatives": ["x"], "material": False}],
    "verdict": {"status": "PENDING", "reason": None},
}


class TestParseSpec:
    def test_bare_yaml(self):
        doc = extract.parse_spec("meta:\n  slug: x\n")
        assert doc == {"meta": {"slug": "x"}}

    def test_fenced_yaml(self):
        doc = extract.parse_spec("```yaml\nmeta:\n  slug: x\n```")
        assert doc == {"meta": {"slug": "x"}}

    def test_prose_around_yaml_raises(self):
        with pytest.raises((ValueError, yaml.YAMLError)):
            extract.parse_spec("here is the spec, hope it helps!")

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="not a YAML mapping"):
            extract.parse_spec("- just\n- a\n- list\n")


class TestValidateSpec:
    def test_valid_spec_has_no_errors(self):
        assert extract.validate_spec(VALID_SPEC) == []

    def test_missing_section_reported(self):
        doc = {k: v for k, v in VALID_SPEC.items() if k != "costs"}
        errors = extract.validate_spec(doc)
        assert any("costs" in e for e in errors)

    def test_zero_costs_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["costs"] = {"commission_bps": 0, "slippage_bps": 0}
        assert any("costs" in e for e in extract.validate_spec(doc))

    def test_zero_lag_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["signal"]["lag_bars"] = 0
        assert any("lag_bars" in e for e in extract.validate_spec(doc))

    def test_empty_ambiguities_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["ambiguities"] = []
        assert any("ambiguities" in e for e in extract.validate_spec(doc))

    def test_oos_start_must_equal_posted(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["validation"]["oos_start"] = "2020-01-01"
        assert any("oos_start" in e for e in extract.validate_spec(doc))

    def test_untestable_needs_only_reason(self):
        doc = {"meta": {"slug": "x", "url": "u", "posted": "2026-08-09"},
               "verdict": {"status": "UNTESTABLE", "reason": "no deterministic rule"}}
        assert extract.validate_spec(doc) == []

    def test_untestable_without_reason_rejected(self):
        doc = {"meta": {"slug": "x"}, "verdict": {"status": "UNTESTABLE"}}
        assert any("reason" in e for e in extract.validate_spec(doc))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -q`
Expected: FAIL / collection error — `No module named 'extract'`

- [ ] **Step 3: Write `extract.py` with parsing and validation**

```python
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
```

Stop here in this task — `extract_one`, `publish`, `pending`, and `main` come in Tasks 2/3. The module must still be importable, so do not add a `main()` call yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: spec reply parsing and validation (stage 4 core)"
```

---

### Task 2: `extract.py` — `extract_one` (the CLI call)

**Files:**
- Modify: `extract.py` (append below `build_prompt`)
- Test: `tests/test_extract.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestExtractOne:
    ROW = {"url": "https://a.com/x", "title": "T", "source": "S",
           "posted": "2026-08-09", "slug": "test-slug", "page": "p.md"}

    def test_valid_reply_returns_spec(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\narticle body\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        reply = yaml.safe_dump(VALID_SPEC)
        recorded = {}

        def fake_run(cmd, input, capture_output, text, timeout):
            recorded["cmd"] = cmd
            recorded["input"] = input

            class Proc:
                returncode = 0
                stdout = reply
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert result["spec"]["meta"]["slug"] == "test-slug"
        assert "claude" in recorded["cmd"]
        assert "RUBRIC" in recorded["input"]

    def test_invalid_yaml_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = "not yaml at all: ["
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "error" in result

    def test_spec_failing_validation_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        bad = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        bad["signal"]["lag_bars"] = 0

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(bad)
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "invalid spec" in result["error"]
        assert "lag_bars" in result["error"]

    def test_model_flag_passed_through(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        seen = {}

        def fake_run(cmd, **k):
            seen["cmd"] = cmd

            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(VALID_SPEC)
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        extract.extract_one(self.ROW, "RUBRIC", "claude-sonnet-4-6")
        assert "--model" in seen["cmd"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -q -k ExtractOne`
Expected: FAIL — `AttributeError: module 'extract' has no attribute 'extract_one'`

- [ ] **Step 3: Implement `extract_one`**

Append to `extract.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -q`
Expected: PASS (all, including Task 1 tests)

- [ ] **Step 5: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: one claude CLI call per triaged article"
```

---

### Task 3: `extract.py` — publishing, queue derivation, `main()`

The state layer. `publish()` writes outcomes under `specs/` atomically; `pending()` derives the work queue from what already exists; `main()` never writes `articles.jsonl`.

**Files:**
- Modify: `extract.py` (append)
- Test: `tests/test_extract.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestPublish:
    def test_success_writes_spec_atomically(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "test-slug"}
        result = {"url": row["url"], "spec": yaml.safe_load(yaml.safe_dump(VALID_SPEC))}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "spec"
        assert (tmp_path / "specs" / "test-slug.yaml").exists()
        assert not (tmp_path / "specs" / "test-slug.yaml.tmp").exists()

    def test_untestable_writes_marker_with_reason(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "vix-thing"}
        result = {"url": row["url"], "spec": {
            "meta": {"slug": "vix-thing", "url": "u", "posted": "2026-08-09"},
            "verdict": {"status": "UNTESTABLE", "reason": "discretionary"}}}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "untestable"
        text = (tmp_path / "specs" / "vix-thing.untestable").read_text()
        assert "discretionary" in text
        assert not (tmp_path / "specs" / "vix-thing.yaml").exists()

    def test_error_writes_marker(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "vix-thing"}
        result = {"url": row["url"], "error": "claude exited 1: boom"}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "extract_failed"
        assert "claude exited 1" in (tmp_path / "specs" / "vix-thing.error").read_text()


class TestPending:
    def rows(self):
        return [
            {"url": "https://a.com/a", "slug": "done", "stage": "triaged"},
            {"url": "https://a.com/b", "slug": "failed", "stage": "triaged"},
            {"url": "https://a.com/c", "slug": "fresh", "stage": "triaged"},
            {"url": "https://a.com/d", "slug": "fetched-already", "stage": "fetched"},
        ]

    def setup(self, specs):
        specs.mkdir(parents=True)
        (specs / "done.yaml").write_text("meta:\n  slug: done\n")
        (specs / "failed.error").write_text("{}\n")

    def test_skips_specd_and_errored(self, tmp_path):
        specs = tmp_path / "specs"
        self.setup(specs)
        todo = extract.pending(self.rows(), specs, retry=False)
        assert [r["slug"] for r in todo] == ["fresh"]

    def test_retry_reincludes_errored(self, tmp_path):
        specs = tmp_path / "specs"
        self.setup(specs)
        todo = extract.pending(self.rows(), specs, retry=True)
        assert [r["slug"] for r in todo] == ["failed", "fresh"]

    def test_untestable_is_final_even_with_retry(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir(parents=True)
        (specs / "done.untestable").write_text("{}\n")
        rows = [{"url": "u", "slug": "done", "stage": "triaged"}]
        assert extract.pending(rows, specs, retry=True) == []


class TestMain:
    def test_publishes_without_touching_articles_jsonl(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "data").mkdir()
        articles = tmp_path / "data" / "articles.jsonl"
        articles.write_text(extract.json.dumps({
            "url": "https://a.com/x", "title": "T", "source": "S",
            "posted": "2026-08-09", "slug": "test-slug",
            "page": "data/pages/p.md", "stage": "triaged",
        }) + "\n")
        page = tmp_path / "data" / "pages" / "p.md"
        page.parent.mkdir(parents=True)
        page.write_text("---\n\nbody\n")
        before = articles.read_text()

        monkeypatch.setattr(extract, "ROOT", tmp_path)
        monkeypatch.setattr(extract, "ARTICLES", articles)
        monkeypatch.setattr(extract, "SPECS", tmp_path / "specs")
        monkeypatch.setattr(extract, "PROMPT", tmp_path / "prompt.md")
        (tmp_path / "prompt.md").write_text("RUBRIC")

        calls = []

        def fake_run(*a, **k):
            calls.append(1)

            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(VALID_SPEC)
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)

        assert extract.main_with(["--jobs", "1"]) == 0
        assert (tmp_path / "specs" / "test-slug.yaml").exists()
        assert articles.read_text() == before  # read-only: never rewritten

        # resume: a second run finds nothing to do and makes zero CLI calls
        assert extract.main_with(["--jobs", "1"]) == 0
        assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -q -k "Publish or Pending or TestMain"`
Expected: FAIL — `no attribute 'publish'`

- [ ] **Step 3: Implement `publish`, `pending`, and `main`**

Append to `extract.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -q`
Expected: PASS (all)

- [ ] **Step 5: Run the whole suite (guard against regressions)**

Run: `uv run pytest tests/ -q`
Expected: PASS (71 existing + new extract tests)

- [ ] **Step 6: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: artifact-as-state publishing and resumable queue"
```

---

### Task 4: `codegen.py` — smoke test and prompt helpers

**Files:**
- Create: `codegen.py`
- Test: `tests/test_codegen.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_codegen.py
"""Tests for codegen.py — smoke testing generated strategy modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import codegen


class TestSmokeFrame:
    def test_shape_and_columns(self):
        df = codegen.smoke_frame()
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 30
        assert df.index.is_monotonic_increasing


class TestSmokeTest:
    def write_module(self, tmp_path, source, name="m"):
        path = tmp_path / "strategies"
        path.mkdir(exist_ok=True)
        f = path / f"{name}.py"
        f.write_text(source)
        return f

    def test_valid_module_passes(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return (df.close > df.close.rolling(5).mean()).astype(float)\n"
        ))
        assert codegen.smoke_test(f, {}) is None

    def test_missing_signal_fails(self, tmp_path):
        f = self.write_module(tmp_path, "x = 1\n")
        err = codegen.smoke_test(f, {})
        assert err is not None and "signal" in err

    def test_crashing_signal_fails_with_exception(self, tmp_path):
        f = self.write_module(tmp_path, (
            "def signal(df, **params):\n"
            "    raise ValueError('boom')\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "ValueError" in err

    def test_wrong_index_fails(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return pd.Series([1.0])\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "index" in err

    def test_nan_output_fails(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def signal(df, **params):\n"
            "    return pd.Series(np.nan, index=df.index)\n"
        ))
        err = codegen.smoke_test(f, {})
        assert err is not None and "non-finite" in err

    def test_non_series_output_fails(self, tmp_path):
        f = self.write_module(tmp_path, "def signal(df, **params):\n    return 1.0\n")
        err = codegen.smoke_test(f, {})
        assert err is not None and "Series" in err

    def test_params_are_forwarded(self, tmp_path):
        f = self.write_module(tmp_path, (
            "import pandas as pd\n"
            "def signal(df, lookback=5):\n"
            "    return (df.close > df.close.rolling(lookback).mean()).astype(float)\n"
        ))
        assert codegen.smoke_test(f, {"lookback": 10}) is None


class TestStripFence:
    def test_bare_source(self):
        assert codegen.strip_fence("x = 1") == "x = 1"

    def test_python_fence(self):
        assert codegen.strip_fence("```python\nx = 1\n```") == "x = 1"

    def test_bare_fence(self):
        assert codegen.strip_fence("```\nx = 1\n```") == "x = 1"


class TestBuildPrompt:
    def test_prompt_embeds_spec_and_contract(self):
        prompt = codegen.build_prompt({"meta": {"slug": "s"}, "signal": {"definition": "d"}})
        assert "def signal" in prompt
        assert "slug: s" in prompt
        assert "do NOT shift" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codegen.py -q`
Expected: collection error — `No module named 'codegen'`

- [ ] **Step 3: Write `codegen.py` helpers (no main yet)**

```python
#!/usr/bin/env python3
"""Stage [5] CODEGEN -- write strategies/<slug>.py from specs/<slug>.yaml.

One claude -p call per spec, then an OFFLINE smoke test: the module must
import and signal(df, **params) must return a finite, index-aligned series
on synthetic data. The backtest harness lags the signal -- generated code
must not shift it, and the prompt says so explicitly.

Independent and resumable: reads only specs/, writes only strategies/.
The strategy file itself is the state -- re-running skips coded slugs.

    python3 codegen.py --limit 5     # calibrate
    python3 codegen.py               # every spec without a strategy yet
    python3 codegen.py --retry-failed
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SPECS = ROOT / "specs"
STRATEGIES = ROOT / "strategies"

CLI_TIMEOUT = 600

CONTRACT = """\
Write a Python module implementing the strategy spec below.

Contract:
- define exactly: def signal(df: pd.DataFrame, **params) -> pd.Series
- `df` has lowercase open/high/low/close/volume columns and a DatetimeIndex.
- Return target weights (float series, or boolean treated as 0/1) with the
  SAME index as df.
- The harness lags the signal and applies costs -- do NOT shift, and do NOT
  compute returns or costs.
- Vectorised pandas/numpy only; no network, no file I/O, no prints.
- Every parameter the spec's signal section uses must be a keyword argument
  with a default.
- Output ONLY Python source. No markdown fences, no prose.
"""


def strip_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    return (fence.group(1) if fence else text).strip()


def build_prompt(spec: dict) -> str:
    return (f"{CONTRACT}\n---\n\nStrategy spec:\n\n```yaml\n"
            f"{yaml.safe_dump(spec, sort_keys=False)}```\n")


def smoke_frame(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + np.linspace(-0.01, 0.01, n)), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(99.0), "high": close * 1.01,
         "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def smoke_test(path: Path, params: dict) -> str | None:
    """Import the module and call signal on synthetic data. None = OK."""
    spec_ = importlib.util.spec_from_file_location(f"strategies.{path.stem}", path)
    mod = importlib.util.module_from_spec(spec_)
    try:
        spec_.loader.exec_module(mod)
        if not hasattr(mod, "signal"):
            return "defines no signal(df, **params)"
        df = smoke_frame()
        out = mod.signal(df, **params)
    except Exception as exc:  # generated code -- any failure is a codegen failure
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(out, pd.Series):
        return "signal did not return a pd.Series"
    if not out.index.equals(df.index):
        return "signal index does not match df.index"
    if not np.isfinite(pd.to_numeric(out, errors="coerce")).all():
        return "signal returned non-finite values"
    return None
```

Note: codegen does not import from `triage` at all — it never touches `articles.jsonl`. Its only input is `specs/`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codegen.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add codegen.py tests/test_codegen.py
git commit -m "codegen: smoke test and prompt helpers (stage 5 core)"
```

---

### Task 5: `codegen.py` — `generate_one`, publishing, `main()`

**Files:**
- Modify: `codegen.py` (append)
- Test: `tests/test_codegen.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
VALID_MODULE = (
    "import pandas as pd\n"
    "def signal(df, **params):\n"
    "    return (df.close > df.close.rolling(5).mean()).astype(float)\n"
)


def cli_stub(stdout, returncode=0):
    def fake_run(*a, **k):
        class Proc:
            pass
        Proc.returncode = returncode
        Proc.stdout = stdout
        Proc.stderr = ""
        return Proc()
    return fake_run


def write_spec(tmp_path, slug="test-slug"):
    p = tmp_path / "specs" / f"{slug}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("meta:\n"
                 f"  slug: {slug}\n"
                 "signal:\n"
                 "  definition: d\n"
                 "  lag_bars: 1\n")
    return p


class TestGenerateOne:
    def test_valid_reply_returns_source(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub(VALID_MODULE))
        result = codegen.generate_one(spec_path, None)
        assert result["slug"] == "test-slug"
        assert "def signal" in result["source"]

    def test_cli_failure_returns_error(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)
        monkeypatch.setattr(codegen.subprocess, "run", cli_stub("", returncode=1))
        result = codegen.generate_one(spec_path, None)
        assert result["slug"] == "test-slug"
        assert "error" in result


class TestPublish:
    def test_success_smoke_passes_and_publishes(self, tmp_path):
        spec = {"meta": {"slug": "test-slug"},
                "signal": {"definition": "d", "lag_bars": 1}}
        result = {"slug": "test-slug", "source": VALID_MODULE}
        stage = codegen.publish(result, spec, tmp_path / "strategies")
        assert stage == "coded"
        assert (tmp_path / "strategies" / "test-slug.py").exists()
        assert not (tmp_path / "strategies" / "test-slug.py.tmp").exists()

    def test_smoke_failure_removes_module_and_writes_marker(self, tmp_path):
        spec = {"meta": {"slug": "test-slug"},
                "signal": {"definition": "d", "lag_bars": 1}}
        result = {"slug": "test-slug", "source": "x = 1\n"}  # no signal()
        stage = codegen.publish(result, spec, tmp_path / "strategies")
        assert stage == "codegen_failed"
        assert "signal" in (tmp_path / "strategies" / "test-slug.error").read_text()
        # a module that failed smoke is never left in place
        assert not (tmp_path / "strategies" / "test-slug.py").exists()

    def test_cli_error_writes_marker_without_module(self, tmp_path):
        result = {"slug": "x", "error": "claude exited 1"}
        stage = codegen.publish(result, {"meta": {"slug": "x"}}, tmp_path / "strategies")
        assert stage == "codegen_failed"
        assert (tmp_path / "strategies" / "x.error").exists()


class TestPendingSpecs:
    def test_skips_coded_slugs(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text("meta:\n  slug: a\n")
        (specs / "b.yaml").write_text("meta:\n  slug: b\n")
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "a.py").write_text("# done\n")
        todo = codegen.pending_specs(specs, strategies, retry=False)
        assert [p.stem for p in todo] == ["b"]

    def test_error_marker_blocks_retry_unless_flag(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text("meta:\n  slug: a\n")
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "a.error").write_text("{}\n")
        assert codegen.pending_specs(specs, strategies, retry=False) == []
        assert [p.stem for p in codegen.pending_specs(specs, strategies, retry=True)] == ["a"]


class TestMain:
    def test_runs_and_resumes_without_articles_jsonl(self, tmp_path, monkeypatch, capsys):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "a.yaml").write_text(
            "meta:\n  slug: a\nsignal:\n  definition: d\n  lag_bars: 1\n")
        strategies = tmp_path / "strategies"

        monkeypatch.setattr(codegen, "ROOT", tmp_path)
        monkeypatch.setattr(codegen, "SPECS", specs)
        monkeypatch.setattr(codegen, "STRATEGIES", strategies)

        calls = []

        def fake_run(*a, **k):
            calls.append(1)

            class Proc:
                returncode = 0
                stdout = VALID_MODULE
                stderr = ""

            return Proc()

        monkeypatch.setattr(codegen.subprocess, "run", fake_run)

        assert codegen.main_with(["--jobs", "1"]) == 0
        assert (strategies / "a.py").exists()
        assert not (tmp_path / "data").exists()  # codegen never touches articles

        # resume: a second run finds nothing to do and makes zero CLI calls
        assert codegen.main_with(["--jobs", "1"]) == 0
        assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codegen.py -q -k "GenerateOne or Publish or PendingSpecs or TestMain"`
Expected: FAIL — `no attribute 'generate_one'`

- [ ] **Step 3: Implement the remaining functions**

Append to `codegen.py`:

```python
def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def pending_specs(specs_dir: Path, strategies_dir: Path, retry: bool) -> list[Path]:
    """Specs with no codegen outcome yet. A .error marker counts as an
    outcome unless retry is set."""
    done = {p.stem for p in strategies_dir.glob("*.py")}
    if not retry:
        done |= {p.stem for p in strategies_dir.glob("*.error")}
    return [p for p in sorted(specs_dir.glob("*.yaml")) if p.stem not in done]


def generate_one(spec_path: Path, model: str | None) -> dict:
    """One claude -p call -> {"slug", "source"} or {"slug", "error"}."""
    spec = yaml.safe_load(spec_path.read_text())
    slug = spec["meta"]["slug"]
    cmd = ["claude", "-p"] + (["--model", model] if model else [])

    try:
        proc = subprocess.run(cmd, input=build_prompt(spec), capture_output=True,
                              text=True, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"slug": slug, "error": f"claude CLI timed out after {CLI_TIMEOUT}s"}
    if proc.returncode != 0:
        return {"slug": slug, "error": f"claude exited {proc.returncode}: {proc.stderr[:200]}"}
    source = strip_fence(proc.stdout)
    if not source:
        return {"slug": slug, "error": "empty reply"}
    return {"slug": slug, "source": source}


def publish(result: dict, spec: dict, strategies_dir: Path) -> str:
    """Smoke and publish one generated module. The artifact is the state."""
    slug = result["slug"]
    if "error" in result:
        _write_atomic(strategies_dir / f"{slug}.error",
                      json.dumps({"error": result["error"]}, indent=2))
        return "codegen_failed"

    path = strategies_dir / f"{slug}.py"
    _write_atomic(path, result["source"])  # never visible half-written

    params = {k: v for k, v in spec.get("signal", {}).items()
              if k not in ("definition", "lag_bars")}
    err = smoke_test(path, params)
    if err:
        path.unlink()  # a module that fails smoke is removed, not left trusted
        _write_atomic(strategies_dir / f"{slug}.error",
                      json.dumps({"error": err}, indent=2))
        return "codegen_failed"
    return "coded"


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent claude calls")
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry specs with a strategies/<slug>.error marker")
    args = ap.parse_args(argv)

    todo = pending_specs(SPECS, STRATEGIES, args.retry_failed)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to codegen -- run extract.py first, or pass --retry-failed")
        return 0

    print(f"codegen {len(todo)} specs ({args.jobs} at a time)\n")
    stages: dict[str, int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate_one, p, args.model): p for p in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            spec_path = futures[fut]
            spec = yaml.safe_load(spec_path.read_text())
            stage = publish(fut.result(), spec, STRATEGIES)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "coded" else "FAIL"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {spec_path.stem}")

    print("\nstages:", ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    print(f"strategies -> {STRATEGIES.relative_to(ROOT)}/")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codegen.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add codegen.py tests/test_codegen.py
git commit -m "codegen: artifact-as-state publishing and resumable queue"
```

---

### Task 6: `backtest_batch.py` — run every spec without blocking, skip completed

**Why:** a serial `for spec in specs/*.yaml` loop blocks on the slowest spec. The one genuine hang risk is the first-time yfinance download inside `load_prices` (no explicit timeout there); compute itself is seconds per spec. This runner puts each spec in its own subprocess with a hard timeout, so no single spec can ever stall the batch, runs specs concurrently, and **resumes by skipping any slug that already has a leaderboard row**.

**Design note:** the pool is a `ThreadPoolExecutor`, not processes — isolation comes from the `subprocess` boundary, and threads keep `run_one` monkeypatchable in tests.

**Files:**
- Create: `backtest_batch.py`
- Test: `tests/test_backtest_batch.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest_batch.py
"""Tests for backtest_batch.py — isolated, timeout-capped, resumable batch runs."""
from __future__ import annotations

import json

import pytest

import backtest_batch


class TestRunOne:
    def test_success_returns_output(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = "s: Sharpe 0.4 net\n"
                stderr = ""
            return Proc()

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "output" in result
        assert result["spec"].endswith("s.yaml")

    def test_timeout_is_recorded_not_raised(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            raise backtest_batch.subprocess.TimeoutExpired(cmd="x", timeout=10)

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "timed out" in result["error"]

    def test_nonzero_exit_records_stderr_tail(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            class Proc:
                returncode = 1
                stdout = ""
                stderr = "no price data for SPY"
            return Proc()

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "exit 1" in result["error"]
        assert "no price data" in result["error"]


class TestAlreadyRun:
    def test_reads_slugs(self, tmp_path):
        lb = tmp_path / "leaderboard.jsonl"
        lb.write_text('{"slug": "a"}\n\n{"slug": "b"}\n')
        assert backtest_batch.already_run(lb) == {"a", "b"}

    def test_missing_file_is_empty(self, tmp_path):
        assert backtest_batch.already_run(tmp_path / "nope.jsonl") == set()


class TestRecordFailure:
    def test_appends_jsonl(self, tmp_path):
        path = tmp_path / "results" / "failed.jsonl"
        backtest_batch.record_failure({"spec": "s.yaml", "error": "boom"}, path)
        backtest_batch.record_failure({"spec": "t.yaml", "error": "bang"}, path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["spec"] for r in rows] == ["s.yaml", "t.yaml"]


class TestMain:
    def _patch(self, tmp_path, monkeypatch, ran):
        specs = tmp_path / "specs"
        specs.mkdir(exist_ok=True)
        for slug in ("a", "b"):
            (specs / f"{slug}.yaml").write_text(f"meta:\n  slug: {slug}\n")
        monkeypatch.setattr(backtest_batch, "ROOT", tmp_path)
        monkeypatch.setattr(backtest_batch, "SPECS", specs)
        monkeypatch.setattr(backtest_batch, "LEADERBOARD", tmp_path / "results" / "leaderboard.jsonl")
        monkeypatch.setattr(backtest_batch, "FAILED", tmp_path / "results" / "failed.jsonl")

        def fake_run_one(spec, timeout):
            ran.append(spec.name)
            if spec.name == "a.yaml":
                return {"spec": str(spec), "output": "a: Sharpe 0.4 net\n"}
            return {"spec": str(spec), "error": "timed out after 600s"}

        monkeypatch.setattr(backtest_batch, "run_one", fake_run_one)
        return specs

    def test_batch_continues_after_failure(self, tmp_path, monkeypatch, capsys):
        ran = []
        self._patch(tmp_path, monkeypatch, ran)
        assert backtest_batch.main_with(["--jobs", "2"]) == 0
        failed = (tmp_path / "results" / "failed.jsonl").read_text()
        assert "b.yaml" in failed
        assert "1 ok, 1 failed" in capsys.readouterr().out
        assert sorted(ran) == ["a.yaml", "b.yaml"]

    def test_resume_skips_specs_already_in_leaderboard(self, tmp_path, monkeypatch):
        ran = []
        self._patch(tmp_path, monkeypatch, ran)
        lb = tmp_path / "results" / "leaderboard.jsonl"
        lb.parent.mkdir(parents=True)
        lb.write_text('{"slug": "a"}\n')  # a is already done
        assert backtest_batch.main_with(["--jobs", "1"]) == 0
        assert ran == ["b.yaml"]  # only the un-run spec executed

    def test_no_specs_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest_batch, "SPECS", tmp_path / "empty")
        assert backtest_batch.main_with([]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest_batch.py -q`
Expected: collection error — `No module named 'backtest_batch'`

- [ ] **Step 3: Implement `backtest_batch.py`**

```python
#!/usr/bin/env python3
"""Batch backtest runner -- every spec, isolated process, hard timeout.

Each spec runs as its own `backtest.py` subprocess with a timeout cap, so a
hung data download or a pathological spec can never block the batch. A slug
with a leaderboard row is skipped, so re-running resumes where the last run
stopped. Failures are appended to results/failed.jsonl; the batch moves on.

    python3 backtest_batch.py                 # every specs/*.yaml not yet run
    python3 backtest_batch.py --jobs 4
    python3 backtest_batch.py --timeout 300
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SPECS = ROOT / "specs"
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FAILED = ROOT / "results" / "failed.jsonl"
DEFAULT_TIMEOUT = 600


def run_one(spec: Path, timeout: int) -> dict:
    """One spec in an isolated process. Never raises."""
    cmd = ["uv", "run", "python", "backtest.py", str(spec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"spec": str(spec), "error": f"timed out after {timeout}s"}
    if proc.returncode != 0:
        return {"spec": str(spec), "error": f"exit {proc.returncode}: {proc.stderr[-300:]}"}
    return {"spec": str(spec), "output": proc.stdout}


def already_run(leaderboard: Path) -> set[str]:
    """Slugs with a leaderboard row -- running them again is a no-op."""
    if not leaderboard.exists():
        return set()
    return {json.loads(line)["slug"]
            for line in leaderboard.read_text().splitlines() if line.strip()}


def record_failure(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(result) + "\n")


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="concurrent specs")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="per-spec subprocess cap in seconds")
    ap.add_argument("--specs", type=Path, default=SPECS)
    args = ap.parse_args(argv)

    done = already_run(LEADERBOARD)
    specs = []
    for path in sorted(args.specs.glob("*.yaml")):
        slug = yaml.safe_load(path.read_text())["meta"]["slug"]
        if slug not in done:
            specs.append(path)
    if not specs:
        print(f"nothing to backtest in {args.specs} -- all specs have leaderboard rows")
        return 1

    print(f"backtesting {len(specs)} specs "
          f"({args.jobs} at a time, {args.timeout}s cap each)\n")

    ok = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, s, args.timeout): s for s in specs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            if "error" in result:
                failed += 1
                record_failure(result, FAILED)
                print(f"[{i}/{len(specs)}] FAIL  {result['error'][:60]:<60}  {result['spec']}")
            else:
                ok += 1
                lines = result["output"].strip().splitlines()
                print(f"[{i}/{len(specs)}] OK    {lines[0] if lines else ''}")

    print(f"\n{ok} ok, {failed} failed (see {FAILED.relative_to(ROOT)})")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest_batch.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add backtest_batch.py tests/test_backtest_batch.py
git commit -m "backtest_batch: timeout-capped concurrent runs, resume via leaderboard"
```

---

### Task 7: `backtest.py` — leaderboard append

**Files:**
- Modify: `backtest.py` (add `append_leaderboard`, call from `main()`)
- Test: `tests/test_backtest.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
class TestAppendLeaderboard:
    def make_spec_and_out(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)
        spec = {"verdict": {"status": "WEAKER", "reason": "r"}}
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        out = {
            "slug": "always-in",
            "full": backtest.metrics(res),
            "null_test": {"beats_null_95": True},
            "deflated_sharpe": 0.9,
            "auto_flags": ["WEAK: does not beat buy-and-hold after costs"],
        }
        return spec, out

    def test_appends_one_json_row(self, tmp_path, monkeypatch):
        spec, out = self.make_spec_and_out(tmp_path, monkeypatch)
        path = tmp_path / "results" / "leaderboard.jsonl"
        backtest.append_leaderboard(spec, out, path)
        rows = backtest.json.loads(path.read_text().splitlines()[0])
        assert rows["slug"] == "always-in"
        assert rows["verdict"] == "WEAKER"
        assert rows["net_sharpe"] == out["full"]["sharpe"]
        assert rows["deflated_sharpe"] == 0.9
        assert rows["beats_null_95"] is True
        assert rows["flags"] == out["auto_flags"]
        assert "run_at" in rows

    def test_appends_not_overwrites(self, tmp_path, monkeypatch):
        spec, out = self.make_spec_and_out(tmp_path, monkeypatch)
        path = tmp_path / "results" / "leaderboard.jsonl"
        backtest.append_leaderboard(spec, out, path)
        backtest.append_leaderboard(spec, out, path)
        assert len(path.read_text().splitlines()) == 2

    def test_creates_results_dir(self, tmp_path, monkeypatch):
        spec, out = self.make_spec_and_out(tmp_path, monkeypatch)
        path = backtest.append_leaderboard(spec, out, tmp_path / "results" / "leaderboard.jsonl")
        assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py::TestAppendLeaderboard -q`
Expected: FAIL — `no attribute 'append_leaderboard'`

- [ ] **Step 3: Implement `append_leaderboard` and wire it into `main()`**

In `backtest.py`, add to the imports at the top:

```python
import subprocess
from datetime import datetime, timezone
```

Add the function (above `main()`):

```python
def append_leaderboard(spec: dict, out: dict, path: Path | None = None) -> Path:
    """One row per completed run -- the leaderboard that feeds the
    deflated-Sharpe n_tested_so_far count. Append-only; backtest_batch.py
    treats an existing row as "already done"."""
    path = path or (ROOT / "results" / "leaderboard.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, cwd=ROOT).stdout.strip()
    row = {
        "slug": out["slug"],
        "verdict": spec["verdict"]["status"],
        "net_sharpe": out["full"]["sharpe"],
        "deflated_sharpe": out["deflated_sharpe"],
        "beats_null_95": out["null_test"].get("beats_null_95"),
        "commit": commit,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flags": out["auto_flags"],
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return path
```

In `main()`, after `report = write_report(args.spec, out)` add:

```python
    append_leaderboard(yaml.safe_load(args.spec.read_text()), out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -q`
Expected: PASS (all, including existing backtest tests)

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (71 existing + extract + codegen + batch + leaderboard tests)

- [ ] **Step 6: Commit**

```bash
git add backtest.py tests/test_backtest.py
git commit -m "backtest: append run verdicts to results/leaderboard.jsonl"
```

---

### Task 8: Update README with the new stages

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the stage commands**

In `README.md`, after the existing pipeline description, add:

```markdown
## Stages

```bash
python3 harvest.py                # [1] latest Quantocracy links -> data/articles.jsonl
python3 fetch.py                  # [2] article text -> data/pages/
python3 triage.py                 # [3] backtestability scores -> data/triage.jsonl
python3 extract.py                # [4] triaged article -> specs/<slug>.yaml
python3 codegen.py                # [5] spec -> strategies/<slug>.py (smoke-tested)
python3 backtest_batch.py         # [6-8] every spec, isolated process, timeout-capped
uv run python backtest.py specs/<slug>.yaml   # [6-8] single spec + report + leaderboard row
```

Every stage is independent and resumable: re-running any command skips work
that already has its output artifact. Specs, strategy code, and
`results/leaderboard.jsonl` are committed; page caches, price caches, and
full reports stay out of Git.
```

- [ ] **Step 2: Verify the commands render sensibly**

Run: `uv run python extract.py --help && uv run python codegen.py --help && uv run python backtest_batch.py --help`
Expected: all three print usage without error (argparse help; no network)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: document extract/codegen/batch stages and committed artifacts"
```

---

## Known limitations (accepted, documented)

- **`load_prices` (backtest.py:44) has no explicit timeout on its yfinance cache-miss download.** The batch runner's per-spec subprocess timeout (Task 6) caps the damage at 600s per spec, but a standalone `backtest.py` invocation can still hang on a flaky network. Fixing `load_prices` itself is out of scope for this plan — it changes existing behavior and deserves its own change + review.
- **Single-asset specs only** — `run()` exits for multi-ticker universes; cross-sectional support is future work.
- **LLM stages are imperfect on first pass** — both `extract.py` and `codegen.py` ship retry flags (`--respec`, `--retry-failed`) by design; expect a few failures per batch.
- **Slug collisions** — `slugify` truncation can in principle collide two article titles onto one slug; extract's pending() would then treat the second as done. Accept at ~25-item scale; revisit if the collection grows.

## Out of scope (deliberately)

- Rewriting triage/fetch to use the shared helpers — they already work
- DuckDB/Parquet store — revisit when the collection outgrows jsonl scanning
- Verdict auto-upgrade logic — `verdict.status` stays as the template defines it; the human decides REPLICATED/WEAKER/FAILED from the report
- CI workflow — the repo has none today; adding one is a separate decision
- A `load_prices` fetch timeout — see Known limitations

## Self-review notes

- Spec coverage: spec generation (Tasks 1–3), code generation (Tasks 4–5), batch runner (Task 6), verdict recording (Task 7), docs (Task 8) — all design sections have tasks
- Independence: `codegen.py` reads only `specs/`; `extract.py` reads `articles.jsonl` but never writes it; `backtest_batch.py` reads only `specs/` + `results/`. Output dirs are disjoint → the three stages are parallel-safe by construction, and every main() has an explicit test asserting resume makes zero redundant calls (or zero CLI calls)
- Type consistency: `extract_one` returns `{"url", "spec"|"error"}`; `codegen.generate_one` returns `{"slug", "source"|"error"}`; `backtest_batch.run_one` returns `{"spec", "output"|"error"}`; `publish` functions return the stage string — consistent across tasks and tests
- All tests are offline; every `claude` CLI call is stubbed via `monkeypatch.setattr(subprocess, "run", ...)`; the batch runner uses a thread pool precisely so `run_one` stays monkeypatchable in tests
- Blocking concern resolved: no stage can block indefinitely — extract/codegen have CLI timeouts, the batch runner has per-spec subprocess timeouts
