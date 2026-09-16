# Provider-Agnostic LLM Adapter (Issue #6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the hardcoded LLM CLI invocation (`opencode run`, previously `claude -p`) from `triage.py`, `extract.py`, and `codegen.py` into a shared, provider-agnostic `llm.py` adapter with env-var provider swap and argument passthrough.

**Architecture:** New module `llm.py` exposes `complete(prompt, model, extra, timeout) -> str` (single-shot CLI completion, raises `LLMError`) plus `preflight()` and `build_argv()`. The CLI template comes from `AUTOQUANT_LLM_CMD` (default `opencode run`); extra args flow in globally via `AUTOQUANT_LLM_EXTRA_ARGS` and per-invocation via a repeatable `--llm-arg` flag on each stage. Each stage keeps its own prompt building and reply parsing; only the `cmd`/`subprocess.run`/error plumbing changes. `llm.py` is also where the `shutil.which` preflight and `FileNotFoundError` handling live (the failure-handling half of issue #5).

**Tech Stack:** Python 3.11 stdlib (`subprocess`, `shlex`, `shutil`, `os`), pytest via `uv run pytest`. Repo conventions: follow `AGENTS.md` (smallest change, honest type annotations, TDD, `uv` for all runs).

**Repo:** `/Users/nehapriya/Desktop/autoquant` (fork remote `fork`, upstream `origin` = `uberdeveloper/autoquant`).

**Key existing facts the implementer must know:**

- `CLI_TIMEOUT` differs per stage: `triage.py:37` = 300, `extract.py:34` = 600, `codegen.py:35` = 600. The stages keep their own constant and pass it to `complete()`.
- Error-row message formats that tests assert on: `"opencode CLI timed out after {N}s"`, `"opencode exited {code}: {stderr[:200]}"`. The adapter preserves these exact shapes (with `argv[0]` in place of the literal `opencode`).
- Existing stage tests (e.g. `tests/test_triage.py:78-96`) monkeypatch `stage.subprocess.run`, which patches the *global* `subprocess` module that `llm.py` also uses — they keep passing after the refactor with no edits.
- `extract.py` and `codegen.py` have `main_with(argv)`; `triage.py` only has `main()`. This plan gives `triage.py` the same `main_with(argv)` wrapper so preflight is testable.
- Commit style: lowercase `area: summary` (e.g. `codegen: multi-asset contract and smoke test`).

---

### Task 1: `llm.py` adapter module

**Files:**
- Create: `llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm.py`:

```python
"""Tests for llm.py — provider-agnostic single-shot CLI adapter."""
from __future__ import annotations

import subprocess

import pytest

import llm


def clear_env(monkeypatch):
    monkeypatch.delenv("AUTOQUANT_LLM_CMD", raising=False)
    monkeypatch.delenv("AUTOQUANT_LLM_EXTRA_ARGS", raising=False)


class TestBuildArgv:
    def test_default_cmd(self, monkeypatch):
        clear_env(monkeypatch)
        assert llm.build_argv() == ["opencode", "run"]

    def test_model_appended_when_no_placeholder(self, monkeypatch):
        clear_env(monkeypatch)
        assert llm.build_argv("sonnet") == ["opencode", "run", "--model", "sonnet"]

    def test_no_model_no_flag(self, monkeypatch):
        clear_env(monkeypatch)
        assert "--model" not in llm.build_argv()

    def test_env_swaps_whole_command(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "codex exec")
        assert llm.build_argv() == ["codex", "exec"]

    def test_model_placeholder_substituted(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "opencode run {model}")
        assert llm.build_argv("sonnet") == ["opencode", "run", "sonnet"]

    def test_model_placeholder_empty_when_model_is_none(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "opencode run {model}")
        assert llm.build_argv(None) == ["opencode", "run", ""]

    def test_env_extra_args_appended_last(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_EXTRA_ARGS", "--max-turns 1")
        assert llm.build_argv(extra=["--verbose"]) == \
            ["opencode", "run", "--verbose", "--max-turns", "1"]


class TestPreflight:
    def test_missing_cli_raises(self, monkeypatch):
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "definitely-not-a-real-cli-xyz -p")
        with pytest.raises(llm.LLMError, match="not found on PATH"):
            llm.preflight()

    def test_present_cli_passes(self, monkeypatch):
        monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/opencode")
        assert llm.preflight() is None


class TestComplete:
    def stub_run(self, monkeypatch, *, returncode=0, stdout="ok", stderr="",
                 raise_exc=None):
        recorded = {}

        def fake_run(argv, input, capture_output, text, timeout):
            recorded["argv"] = argv
            recorded["input"] = input
            recorded["timeout"] = timeout
            if raise_exc is not None:
                raise raise_exc

            class Proc:
                pass

            Proc.returncode = returncode
            Proc.stdout = stdout
            Proc.stderr = stderr
            return Proc()

        monkeypatch.setattr(llm.subprocess, "run", fake_run)
        return recorded

    def test_returns_stdout_and_forwards_prompt_and_timeout(self, monkeypatch):
        clear_env(monkeypatch)
        rec = self.stub_run(monkeypatch, stdout="reply text")
        assert llm.complete("the prompt", timeout=300) == "reply text"
        assert rec["argv"] == ["opencode", "run"]
        assert rec["input"] == "the prompt"
        assert rec["timeout"] == 300

    def test_model_and_extra_reach_argv(self, monkeypatch):
        clear_env(monkeypatch)
        rec = self.stub_run(monkeypatch)
        llm.complete("p", model="sonnet", extra=["--output-format", "json"])
        assert rec["argv"] == ["opencode", "run", "--model", "sonnet",
                               "--output-format", "json"]

    def test_nonzero_exit_raises_llmerror_with_stderr(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, returncode=1, stderr="auth expired")
        with pytest.raises(llm.LLMError, match=r"opencode exited 1: auth expired"):
            llm.complete("p")

    def test_timeout_raises_llmerror(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch,
                      raise_exc=subprocess.TimeoutExpired(cmd="opencode", timeout=300))
        with pytest.raises(llm.LLMError, match="timed out after 300s"):
            llm.complete("p", timeout=300)

    def test_missing_binary_raises_llmerror(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, raise_exc=FileNotFoundError(2, "No such file"))
        with pytest.raises(llm.LLMError, match="not found on PATH"):
            llm.complete("p")

    def test_stderr_truncated_to_200_chars(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, returncode=1, stderr="x" * 500)
        with pytest.raises(llm.LLMError) as excinfo:
            llm.complete("p")
        assert len(str(excinfo.value)) < 250
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/nehapriya/Desktop/autoquant && uv run pytest tests/test_llm.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Implement `llm.py`**

Create `llm.py`:

```python
"""Single-shot text completion via a configurable CLI.

The three LLM stages (triage.py, extract.py, codegen.py) call complete()
instead of assembling their own LLM CLI invocation. The provider is
swapped through env vars, no code edits:

    AUTOQUANT_LLM_CMD="opencode run"           # default
    AUTOQUANT_LLM_CMD="claude -p"              # any text-in/text-out CLI
    AUTOQUANT_LLM_CMD="crush run -q {model}"   # {model} placeholder optional

Extra arguments reach the CLI two ways: globally via AUTOQUANT_LLM_EXTRA_ARGS
(shlex-split), or per invocation via the `extra` parameter (wired to each
stage's repeatable --llm-arg flag). Without a {model} placeholder, --model
<model> is appended when a model is given.

This module is also the single place that handles a missing CLI: preflight()
checks PATH before a batch starts, and complete() turns FileNotFoundError
into LLMError so a mid-run failure becomes a per-item error, never a crash.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess

DEFAULT_CMD = "opencode run"
CLI_TIMEOUT = 600


class LLMError(Exception):
    """A completion failed: CLI missing, timed out, or nonzero exit."""


def build_argv(model: str | None = None,
               extra: list[str] | None = None) -> list[str]:
    """Assemble the CLI argv from AUTOQUANT_LLM_CMD (default: `opencode run`)."""
    template = os.environ.get("AUTOQUANT_LLM_CMD", DEFAULT_CMD)
    if "{model}" in template:
        argv = shlex.split(template.format(model=model or ""))
    else:
        argv = shlex.split(template)
        if model:
            argv += ["--model", model]
    argv += list(extra or [])
    argv += shlex.split(os.environ.get("AUTOQUANT_LLM_EXTRA_ARGS", ""))
    return argv


def preflight() -> None:
    """Raise LLMError early if the configured CLI is not on PATH."""
    template = os.environ.get("AUTOQUANT_LLM_CMD", DEFAULT_CMD)
    argv = shlex.split(template)
    if shutil.which(argv[0]) is None:
        raise LLMError(f"`{argv[0]}` CLI not found on PATH "
                       f"(AUTOQUANT_LLM_CMD={template!r})")


def complete(prompt: str, model: str | None = None,
             extra: list[str] | None = None,
             timeout: int = CLI_TIMEOUT) -> str:
    """One single-shot CLI completion: prompt in, text out.

    Raises LLMError on a missing binary, a timeout, or a nonzero exit, so
    callers only need one except clause to produce their error artifact.
    """
    argv = build_argv(model, extra)
    try:
        proc = subprocess.run(argv, input=prompt, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise LLMError(f"{argv[0]} CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise LLMError(f"`{argv[0]}` CLI not found on PATH")
    if proc.returncode != 0:
        raise LLMError(f"{argv[0]} exited {proc.returncode}: {proc.stderr[:200]}")
    return proc.stdout
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: all PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "llm: provider-agnostic single-shot CLI adapter"
```

---

### Task 2: Route `triage.py` through `llm.py`

**Files:**
- Modify: `triage.py` (module docstring line 4-6, imports lines 21-29, `score_one` lines 91-102, `main` lines 117-144)
- Test: `tests/test_triage.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_triage.py`, add `import llm` to the imports (after `import triage`), then add three tests inside `TestScoreOne` (at the end of the class):

```python
    def test_llm_error_becomes_error_row(self, tmp_path, monkeypatch):
        row = dict(ROW, page="p.md")
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(triage, "ROOT", tmp_path)

        def fake_complete(prompt, **k):
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(triage.llm, "complete", fake_complete)
        result = triage.score_one(row, "RUBRIC", None)
        assert result == {"url": row["url"],
                          "error": "`claude` CLI not found on PATH"}

    def test_extra_args_forwarded(self, tmp_path, monkeypatch):
        row = dict(ROW, page="p.md")
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(triage, "ROOT", tmp_path)
        seen = {}

        def fake_complete(prompt, *, model=None, extra=None, timeout=300):
            seen["model"] = model
            seen["extra"] = list(extra or [])
            return json.dumps({"score": 4})

        monkeypatch.setattr(triage.llm, "complete", fake_complete)
        triage.score_one(row, "RUBRIC", "sonnet", ["--output-format", "json"])
        assert seen["model"] == "sonnet"
        assert seen["extra"] == ["--output-format", "json"]

    def test_main_exits_when_preflight_fails(self, monkeypatch):
        def boom():
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(triage.llm, "preflight", boom)
        with pytest.raises(SystemExit, match="not found on PATH"):
            triage.main_with([])
```

Note: `test_main_exits_when_preflight_fails` requires `main_with` to exist and preflight to run *before* any file reads — that is part of this task's implementation.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_triage.py -v`
Expected: FAIL — `AttributeError: module 'triage' has no attribute 'llm'` (or `main_with` missing)

- [ ] **Step 3: Refactor `triage.py`**

3a. Update the module docstring (lines 4-6) — replace:

```
Runs triage_prompt.md over every fetched article via the `opencode` CLI
in headless mode (`opencode run`, prompt piped on stdin), so it uses your
existing opencode auth.
```

with:

```
Runs triage_prompt.md over every fetched article via the LLM CLI configured
in llm.py (default: `opencode run`). Swap the provider with AUTOQUANT_LLM_CMD.
```

3b. Imports: delete `import subprocess` (line 27), add `import llm` after `import json` (keep alphabetical: argparse, concurrent.futures, json, llm, re, sys, pathlib).

3c. Replace `score_one` (currently lines 91-102) with:

```python
def score_one(row: dict, rubric: str, model: str | None,
              extra: list[str] | None = None) -> dict:
    page = ROOT / row["page"]
    prompt = build_prompt(rubric, row, strip_body(page))

    try:
        reply = llm.complete(prompt, model=model, extra=extra,
                             timeout=CLI_TIMEOUT)
    except llm.LLMError as exc:
        return {"url": row["url"], "error": str(exc)}

    try:
        result = parse_json(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"url": row["url"], "error": f"unparseable reply: {exc}"[:300]}
```

(The tail — `result["url"] = ...` through `return result` — stays unchanged. The old `cmd = [...]`, `subprocess.run`, `TimeoutExpired`, and `returncode` blocks are deleted.)

3d. `main()` becomes a `main_with(argv)` pair, and gains preflight + `--llm-arg`. Replace the signature and argparse block (currently lines 117-125):

```python
def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent LLM calls")
    ap.add_argument("--model", default=None,
                    help="override the model (default: your session's)")
    ap.add_argument("--llm-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed through to the LLM CLI (repeatable)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score articles already in triage.jsonl")
    args = ap.parse_args(argv)

    try:
        llm.preflight()
    except llm.LLMError as exc:
        sys.exit(f"error: {exc}")
```

3e. Change the pool submit (currently line 144) to pass the extra args:

```python
        futures = {pool.submit(score_one, r, rubric, args.model,
                               args.llm_arg): r for r in todo}
```

3f. Add a `main()` wrapper before `if __name__ == "__main__":` (mirrors extract.py):

```python
def main() -> int:
    return main_with()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_triage.py tests/test_llm.py -v`
Expected: all PASS — including the pre-existing `TestScoreOne` tests (they patch the global `subprocess.run`, which `llm.py` still calls).

- [ ] **Step 5: Smoke the CLI**

Run: `uv run python triage.py --help`
Expected: help text listing `--llm-arg`, no import errors.

- [ ] **Step 6: Commit**

```bash
git add triage.py tests/test_triage.py
git commit -m "triage: route completions through llm.py, add --llm-arg and preflight"
```

---

### Task 3: Route `extract.py` through `llm.py`

**Files:**
- Modify: `extract.py` (module docstring line 4, imports lines 18-25, `extract_one` lines 138-150, `main_with` lines 204-231)
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_extract.py`, add `import llm` to the imports (after `import extract`), then add two tests inside `TestExtractOne` and one to `TestMain`:

Add to `TestExtractOne`:

```python
    def test_llm_error_becomes_error_row(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_complete(prompt, **k):
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(extract.llm, "complete", fake_complete)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert result == {"url": self.ROW["url"],
                          "error": "`claude` CLI not found on PATH"}

    def test_model_and_extra_forwarded(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        seen = {}

        def fake_complete(prompt, *, model=None, extra=None, timeout=600):
            seen["model"] = model
            seen["extra"] = list(extra or [])
            return yaml.safe_dump(VALID_SPEC)

        monkeypatch.setattr(extract.llm, "complete", fake_complete)
        extract.extract_one(self.ROW, "RUBRIC", "sonnet", ["--max-turns", "1"])
        assert seen["model"] == "sonnet"
        assert seen["extra"] == ["--max-turns", "1"]
```

Add to `TestMain` (after the existing test):

```python
    def test_exits_when_preflight_fails(self, monkeypatch):
        def boom():
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(extract.llm, "preflight", boom)
        with pytest.raises(SystemExit, match="not found on PATH"):
            extract.main_with([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py -v`
Expected: FAIL — `AttributeError: module 'extract' has no attribute 'llm'`

- [ ] **Step 3: Refactor `extract.py`**

3a. Module docstring lines 4-5 — replace `Runs extract_prompt.md over every triaged article via the `opencode` CLI
in headless mode (`opencode run`).` with `Runs extract_prompt.md over every triaged article via the LLM CLI configured in llm.py (default: `opencode run`).`

3b. Imports: delete `import subprocess` (line 22), add `import sys` (stdlib block, after `import re`) and `import llm` (after `import json`).

3c. Replace `extract_one` (currently lines 138-160):

```python
def extract_one(row: dict, rubric: str, model: str | None,
                extra: list[str] | None = None) -> dict:
    """One LLM call -> {"url", "spec"} or {"url", "error"}."""
    page = ROOT / row["page"]
    prompt = build_prompt(rubric, row, strip_body(page))

    try:
        reply = llm.complete(prompt, model=model, extra=extra,
                             timeout=CLI_TIMEOUT)
    except llm.LLMError as exc:
        return {"url": row["url"], "error": str(exc)}

    try:
        spec = parse_spec(reply)
    except (ValueError, yaml.YAMLError) as exc:
        return {"url": row["url"], "error": f"unparseable YAML: {exc}"[:300]}

    errors = validate_spec(spec, row["slug"])
    if errors:
        return {"url": row["url"], "error": "invalid spec: " + "; ".join(errors)}
    return {"url": row["url"], "spec": spec}
```

3d. In `main_with`, after `ap.add_argument("--model", ...)` (line 208) add:

```python
    ap.add_argument("--llm-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed through to the LLM CLI (repeatable)")
```

and after `args = ap.parse_args(argv)` (line 211) add:

```python
    try:
        llm.preflight()
    except llm.LLMError as exc:
        sys.exit(f"error: {exc}")
```

3e. Change the pool submit (currently line 227):

```python
        futures = {pool.submit(extract_one, r, rubric, args.model,
                               args.llm_arg): r for r in todo}
```

3f. Update the `--jobs` help text: `"concurrent claude calls"` → `"concurrent LLM calls"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py tests/test_llm.py -v`
Expected: all PASS. The pre-existing `TestExtractOne`/`TestMain` tests pass unchanged (global `subprocess` patch), except `TestMain.test_publishes_without_touching_articles_jsonl` — `main_with` now calls `llm.preflight()`, which does a real `shutil.which("opencode")`. On machines with `opencode` installed this passes; to make it machine-independent, add this line to that test's monkeypatch block (after line 371 `(tmp_path / "prompt.md").write_text("RUBRIC")`):

```python
        monkeypatch.setattr(extract.llm, "preflight", lambda: None)
```

- [ ] **Step 5: Smoke the CLI**

Run: `uv run python extract.py --help`
Expected: help text listing `--llm-arg`, no import errors.

- [ ] **Step 6: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: route completions through llm.py, add --llm-arg and preflight"
```

---

### Task 4: Route `codegen.py` through `llm.py`

**Files:**
- Modify: `codegen.py` (imports lines 18-29, `generate_one` lines 156-178, `main_with` lines 212-232)
- Test: `tests/test_codegen.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_codegen.py`, add to the imports: `import llm` and `import pytest`, then add two tests to `TestGenerateOne` and one to `TestMain`:

Add to `TestGenerateOne`:

```python
    def test_llm_error_becomes_error_result(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)

        def fake_complete(prompt, **k):
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(codegen.llm, "complete", fake_complete)
        result = codegen.generate_one(spec_path, None)
        assert result["slug"] == "test-slug"
        assert "not found on PATH" in result["error"]

    def test_model_and_extra_forwarded(self, tmp_path, monkeypatch):
        spec_path = write_spec(tmp_path)
        seen = {}

        def fake_complete(prompt, *, model=None, extra=None, timeout=600):
            seen["model"] = model
            seen["extra"] = list(extra or [])
            return VALID_MODULE

        monkeypatch.setattr(codegen.llm, "complete", fake_complete)
        codegen.generate_one(spec_path, "sonnet", ["--max-turns", "1"])
        assert seen["model"] == "sonnet"
        assert seen["extra"] == ["--max-turns", "1"]
```

Add to `TestMain`:

```python
    def test_exits_when_preflight_fails(self, monkeypatch):
        def boom():
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(codegen.llm, "preflight", boom)
        with pytest.raises(SystemExit, match="not found on PATH"):
            codegen.main_with([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_codegen.py -v`
Expected: FAIL — `AttributeError: module 'codegen' has no attribute 'llm'` (and `NameError: name 'pytest' is not defined` for the new class)

- [ ] **Step 3: Refactor `codegen.py`**

3a. Imports: delete `import subprocess` (line 24), add `import sys` (stdlib block, after `import re`) and `import llm` (after `import json`).

3b. Replace `generate_one` (currently lines 156-178):

```python
def generate_one(spec_path: Path, model: str | None,
                 extra: list[str] | None = None) -> dict:
    """One LLM call -> {"slug", "source"} or {"slug", "error"}."""
    try:
        spec = yaml.safe_load(spec_path.read_text())
        slug = spec["meta"]["slug"]
    except (yaml.YAMLError, KeyError, AttributeError, TypeError, OSError) as exc:
        return {"slug": spec_path.stem, "error": f"unreadable spec: {exc}"[:300]}
    if slug != spec_path.stem:
        return {"slug": spec_path.stem,
                "error": "meta.slug does not match spec filename"}

    try:
        source = llm.complete(build_prompt(spec), model=model, extra=extra,
                              timeout=CLI_TIMEOUT)
    except llm.LLMError as exc:
        return {"slug": slug, "error": str(exc)}
    source = strip_fence(source)
    if not source:
        return {"slug": slug, "error": "empty reply"}
    return {"slug": slug, "source": source}
```

3c. In `main_with`, after `ap.add_argument("--model", ...)` (line 216) add:

```python
    ap.add_argument("--llm-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed through to the LLM CLI (repeatable)")
```

and after `args = ap.parse_args(argv)` (line 219) add:

```python
    try:
        llm.preflight()
    except llm.LLMError as exc:
        sys.exit(f"error: {exc}")
```

3d. Change the pool submit (currently line 232):

```python
        futures = {pool.submit(generate_one, p, args.model,
                               args.llm_arg): p for p in todo}
```

3e. Update the `--jobs` help text: `"concurrent claude calls"` → `"concurrent LLM calls"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codegen.py tests/test_llm.py -v`
Expected: all PASS. As in Task 3, add `monkeypatch.setattr(codegen.llm, "preflight", lambda: None)` to the two pre-existing `TestMain` tests (`test_runs_and_resumes_without_articles_jsonl` and `test_malformed_spec_fails_gracefully`) so they don't depend on `opencode` being installed.

- [ ] **Step 5: Smoke the CLI**

Run: `uv run python codegen.py --help`
Expected: help text listing `--llm-arg`, no import errors.

- [ ] **Step 6: Commit**

```bash
git add codegen.py tests/test_codegen.py
git commit -m "codegen: route completions through llm.py, add --llm-arg and preflight"
```

---

### Task 5: Docs and full-suite validation

**Files:**
- Modify: `README.md` (insert after the Stages section, around line 34)

- [ ] **Step 1: Document the env vars and flag in README.md**

Insert this subsection after the paragraph ending `full reports stay out of Git.` (line 33-34), before the next top-level section:

```markdown
### Swapping the LLM provider

The three LLM stages go through `llm.py`. The provider command defaults to
`opencode run` and is overridden with env vars -- no code edits:

```bash
AUTOQUANT_LLM_CMD="claude -p" python3 triage.py
AUTOQUANT_LLM_CMD="crush run -q {model}" python3 extract.py --model sonnet
```

- `AUTOQUANT_LLM_CMD` -- shlex-split command template; a `{model}` placeholder
  is optional (without it, `--model <model>` is appended when `--model` is set).
- `AUTOQUANT_LLM_EXTRA_ARGS` -- extra CLI arguments, shlex-split, appended to
  every call.
- `--llm-arg ARG` -- per-invocation extra argument, repeatable, supported by
  `triage.py`, `extract.py`, and `codegen.py`.

Any CLI that reads the prompt on stdin (or as a positional arg) and prints the
completion on stdout works; the stages' fence-stripping parsers tolerate
formatting differences.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS (existing suites for fetch/harvest/backtest/backtest_batch included), zero failures.

- [ ] **Step 3: Verify the provider swap end-to-end without a real provider**

Run:
```bash
AUTOQUANT_LLM_CMD="definitely-not-a-real-cli-xyz -p" uv run python triage.py --limit 1
```
Expected: `error: `definitely-not-a-real-cli-xyz` CLI not found on PATH (AUTOQUANT_LLM_CMD='definitely-not-a-real-cli-xyz -p')`, exit code 1 — no traceback. (This also demonstrates the issue #5 preflight half.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: LLM provider swap via AUTOQUANT_LLM_CMD, extra-args passthrough"
```

---

## Self-review notes

- **Spec coverage:** shared adapter (Task 1), env-var provider swap with `{model}` placeholder (Task 1), `AUTOQUANT_LLM_EXTRA_ARGS` + per-invocation `extra` wired to repeatable `--llm-arg` on all three stages (Tasks 2-4), stage prompt/parsing logic untouched (only call-site plumbing replaced), `shutil.which` preflight + `FileNotFoundError` → `LLMError` in one place (Tasks 1-4, issue #5 half), README docs (Task 5). The "future: non-CLI provider" section of the issue is explicitly future work — not implemented here (YAGNI).
- **Type consistency:** `complete(prompt: str, model: str | None = None, extra: list[str] | None = None, timeout: int = CLI_TIMEOUT) -> str` used identically at all three call sites; stage functions gain `extra: list[str] | None = None` as the last parameter (keyword-safe default keeps existing 3-arg test calls valid).
- **Timeout differences:** triage passes 300, extract/codegen pass 600 via their own `CLI_TIMEOUT` constants — preserved.
