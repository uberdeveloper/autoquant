# Circuit-Breaker for CLI-Level Batch Failures (Issue #5, remaining half) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the first K completions in a triage/extract/codegen batch all fail at the CLI level (missing binary, auth exit, timeout), abort the batch with one clear message instead of publishing N identical error artifacts.

**Architecture:** `llm.py` gains two pure helpers: `is_cli_error(error)` classifies an error string as CLI-level vs stage-level (parse/validation), and `circuit_break(results, k)` returns an abort message once the first K result rows are all CLI-level errors, else None. Each stage's `main_with` loop accumulates completed results, checks the breaker after each completion, and on trip cancels queued futures, saves/publishes only what already completed, prints the abort message, and returns 1. No changes to prompt building, parsing, or per-item error artifacts.

**Tech Stack:** Python 3.11 stdlib, pytest via `uv run pytest`. Repo conventions: TDD, lowercase `area: summary` commits.

**Repo:** `/Users/nehapriya/Desktop/autoquant`, branch `issue5-circuit-breaker` off `main` (94c4edd).

**Key existing facts the implementer must know:**

- PR #7 already landed the preflight + `FileNotFoundError` -> `LLMError` half of issue #5. Only the optional circuit-breaker (item 3) remains.
- Stage-level (parse/validation) error strings that must NOT trip the breaker: `"unparseable reply: ..."`, `"unparseable YAML: ..."`, `"invalid spec: ..."`, `"unreadable spec: ..."`, `"empty reply"`, `"meta.slug does not match spec filename"`.
- CLI-level error strings produced by `llm.complete` (these are the trip signals): `"<cli> exited <n>: <stderr[:200]>"`, `"<cli> CLI timed out after <n>s"`, `` "`<cli>` CLI not found on PATH" ``.
- All three stages loop over `concurrent.futures.as_completed` inside a `with ThreadPoolExecutor(...)` block. `pool.shutdown(wait=False, cancel_futures=True)` inside the loop cancels not-yet-started work (Python 3.9+; repo floor is 3.11). In-flight completions finish but are not processed — artifact-as-state resume picks them up next run.
- Tests pass `--jobs 1` so completion order == todo order, making "first K" deterministic.
- Existing stage main-loop code locations after PR #7: `triage.py` main_with loop (~lines 143-166), `extract.py` (~lines 230-242), `codegen.py` (~lines 236-248). Verify with grep before editing; line numbers shift per commit.

---

### Task 1: `is_cli_error` + `circuit_break` in `llm.py`

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestCircuitBreaker:
    def test_cli_level_errors_classified(self):
        assert llm.is_cli_error("opencode exited 1: auth expired")
        assert llm.is_cli_error("`claude` CLI not found on PATH")
        assert llm.is_cli_error("opencode CLI timed out after 300s")

    def test_stage_level_errors_not_classified(self):
        assert not llm.is_cli_error("unparseable YAML: while scanning")
        assert not llm.is_cli_error("invalid spec: missing costs")
        assert not llm.is_cli_error("unreadable spec: boom")
        assert not llm.is_cli_error("empty reply")
        assert not llm.is_cli_error("meta.slug does not match spec filename")

    def test_breaker_quiet_below_k(self):
        results = [{"error": "opencode exited 1: auth"}]
        assert llm.circuit_break(results) is None

    def test_breaker_trips_on_k_consecutive_cli_errors(self):
        results = [{"error": "opencode exited 1: auth"} for _ in range(3)]
        msg = llm.circuit_break(results)
        assert msg is not None and "aborting" in msg and "3" in msg

    def test_breaker_spared_by_a_stage_level_error(self):
        results = [{"error": "opencode exited 1: auth"},
                   {"error": "unparseable YAML: x"},
                   {"error": "opencode exited 1: auth"}]
        assert llm.circuit_break(results) is None

    def test_breaker_ignores_successes(self):
        results = [{"score": 4}, {"score": 3}, {"error": "opencode exited 1: a"}]
        assert llm.circuit_break(results) is None

    def test_breaker_k_is_configurable(self):
        results = [{"error": "opencode exited 1: a"} for _ in range(5)]
        assert llm.circuit_break(results, k=5) is not None
        assert llm.circuit_break(results[:4], k=5) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -q`
Expected: FAIL — `AttributeError: module 'llm' has no attribute 'is_cli_error'`

- [ ] **Step 3: Implement in `llm.py`**

Add `import re` to the imports, then append at the end of the file:

```python
CIRCUIT_K = 3  # consecutive CLI-level failures before a batch aborts

_STAGE_LEVEL_PREFIXES = ("unparseable", "invalid spec:", "unreadable spec:",
                         "empty reply", "meta.slug does not match")


def is_cli_error(error: str) -> bool:
    """True if an error-row string came from the LLM CLI plumbing
    (missing binary, nonzero exit, timeout) rather than the stage's own
    reply parsing/validation."""
    if error.startswith(_STAGE_LEVEL_PREFIXES):
        return False
    return ("CLI not found on PATH" in error
            or "CLI timed out after" in error
            or bool(re.match(r"^\S+ exited \d+:", error)))


def circuit_break(results: list[dict], k: int = CIRCUIT_K) -> str | None:
    """Abort message if the first k completed results are ALL CLI-level
    failures -- the provider is down or unauthenticated, and continuing
    would only write N identical error artifacts. None means keep going."""
    head = results[:k]
    if len(head) < k:
        return None
    if all("error" in r and is_cli_error(r["error"]) for r in head):
        first = head[0]["error"]
        return (f"aborting: first {k} results all failed at the CLI level "
                f"(first error: {first[:160]}) -- fix the LLM CLI/auth "
                f"and re-run; completed work is kept")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -q`
Expected: all PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "llm: circuit_break helper for consecutive CLI-level failures"
```

---

### Task 2: Wire the breaker into `triage.py`

**Files:**
- Modify: `triage.py` (main_with loop)
- Test: `tests/test_triage.py`

- [ ] **Step 1: Write the failing test**

Add inside `TestScoreOne`'s module (new class after `TestScoreOne` in `tests/test_triage.py`):

```python
class TestMainCircuitBreaker:
    def test_main_aborts_and_saves_partial(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "data").mkdir()
        rows = []
        for n in range(5):
            page = tmp_path / "data" / "pages" / f"p{n}.md"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text("---\n\nbody\n")
            rows.append({"url": f"https://a.com/{n}", "title": f"T{n}",
                         "source": "S", "posted": "2026-08-09",
                         "slug": f"s{n}", "page": f"data/pages/p{n}.md",
                         "stage": "fetched"})
        articles = tmp_path / "data" / "articles.jsonl"
        articles.write_text("".join(json.dumps(r) + "\n" for r in rows))
        triage_path = tmp_path / "data" / "triage.jsonl"

        monkeypatch.setattr(triage, "ROOT", tmp_path)
        monkeypatch.setattr(triage, "ARTICLES", articles)
        monkeypatch.setattr(triage, "TRIAGE", triage_path)
        monkeypatch.setattr(triage, "PROMPT", tmp_path / "prompt.md")
        (tmp_path / "prompt.md").write_text("RUBRIC")

        def fake_complete(prompt, **k):
            raise llm.LLMError("opencode exited 1: auth expired")

        monkeypatch.setattr(triage.llm, "complete", fake_complete)

        assert triage.main_with(["--jobs", "1", "--limit", "5"]) == 1
        out = capsys.readouterr().out
        assert "aborting" in out
        # exactly the first K=3 error rows were saved; the rest never ran
        saved = [json.loads(l) for l in triage_path.read_text().splitlines()]
        assert len(saved) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_triage.py::TestMainCircuitBreaker -q`
Expected: FAIL — `assert 0 == 1` (main returns 0; no abort exists)

- [ ] **Step 3: Wire the breaker into the `main_with` loop**

In `triage.py`, restructure the executor block to:

```python
    abort = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(score_one, r, rubric, args.model,
                               args.llm_arg): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            abort = llm.circuit_break(results)
            if abort:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            row = by_url[res["url"]]

            if "error" in res:
                row["stage"] = "triage_failed"
                row["triage_error"] = res["error"]
                print(f"[{i}/{len(todo)}] ERR  {res['error'][:60]}")
                continue

            score = res.get("score")
            row["stage"] = "untriageable" if score is None else STAGE_FOR.get(score, "rejected")
            row["triage_score"] = score
            row.pop("triage_error", None)
            print(f"[{i}/{len(todo)}] {str(score):>4}  {row['stage']:<12} "
                  f"{res.get('asset_class', '?'):<14} {row['title'][:44]}")

    # Replace prior entries for re-scored urls rather than appending duplicates.
    fresh = {r["url"] for r in results}
    save(TRIAGE, [r for r in scored if r["url"] not in fresh] + results)
    save(ARTICLES, rows)

    if abort:
        print(f"\n{abort}")
        return 1
```

(The `results` accumulation line `results.append(res)` moves BEFORE the per-row processing; the trailing stage-count printing block stays unchanged after the new `if abort:` early return.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_triage.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add triage.py tests/test_triage.py
git commit -m "triage: abort the batch after consecutive CLI-level failures"
```

---

### Task 3: Wire the breaker into `extract.py`

**Files:**
- Modify: `extract.py` (main_with loop)
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

Add a new class at the end of `tests/test_extract.py`:

```python
class TestMainCircuitBreaker:
    def test_main_aborts_without_publishing_the_rest(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "data").mkdir()
        rows = []
        for n in range(5):
            page = tmp_path / "data" / "pages" / f"p{n}.md"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text("---\n\nbody\n")
            rows.append({"url": f"https://a.com/{n}", "title": f"T{n}",
                         "source": "S", "posted": "2026-08-09",
                         "slug": f"s{n}", "page": f"data/pages/p{n}.md",
                         "stage": "triaged"})
        articles = tmp_path / "data" / "articles.jsonl"
        articles.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (tmp_path / "prompt.md").write_text("RUBRIC")

        monkeypatch.setattr(extract, "ROOT", tmp_path)
        monkeypatch.setattr(extract, "ARTICLES", articles)
        monkeypatch.setattr(extract, "SPECS", tmp_path / "specs")
        monkeypatch.setattr(extract, "PROMPT", tmp_path / "prompt.md")

        def fake_complete(prompt, **k):
            raise llm.LLMError("opencode exited 1: auth expired")

        monkeypatch.setattr(extract.llm, "complete", fake_complete)

        assert extract.main_with(["--jobs", "1", "--limit", "5"]) == 1
        assert "aborting" in capsys.readouterr().out
        # only the first K=3 slugs got an .error marker
        markers = sorted((tmp_path / "specs").glob("*.error"))
        assert len(markers) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extract.py::TestMainCircuitBreaker -q`
Expected: FAIL — `assert 0 == 1`

- [ ] **Step 3: Wire the breaker into the `main_with` loop**

In `extract.py`, restructure the executor block to (note the new `done` list):

```python
    print(f"extracting {len(todo)} articles ({args.jobs} at a time)\n")
    by_url = {r["url"]: r for r in todo}
    stages: dict[str, int] = {}
    done: list[dict] = []

    abort = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(extract_one, r, rubric, args.model,
                               args.llm_arg): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            done.append(result)
            abort = llm.circuit_break(done)
            if abort:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            row = by_url[result["url"]]
            stage = publish(row, result, SPECS)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "spec" else "FAIL" if stage == "extract_failed" else "SKIP"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {row['title'][:50]}")

    if abort:
        print(f"\n{abort}")
        return 1
```

(The trailing stage-count printing block stays unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: abort the batch after consecutive CLI-level failures"
```

---

### Task 4: Wire the breaker into `codegen.py`

**Files:**
- Modify: `codegen.py` (main_with loop)
- Test: `tests/test_codegen.py`

- [ ] **Step 1: Write the failing test**

Add a new class at the end of `tests/test_codegen.py`:

```python
class TestMainCircuitBreaker:
    def test_main_aborts_without_coding_the_rest(self, tmp_path, monkeypatch, capsys):
        specs = tmp_path / "specs"
        specs.mkdir()
        for n in range(5):
            (specs / f"s{n}.yaml").write_text(
                "meta:\n"
                f"  slug: s{n}\n"
                "signal:\n"
                "  definition: d\n"
                "  lag_bars: 1\n")
        strategies = tmp_path / "strategies"

        monkeypatch.setattr(codegen, "ROOT", tmp_path)
        monkeypatch.setattr(codegen, "SPECS", specs)
        monkeypatch.setattr(codegen, "STRATEGIES", strategies)

        def fake_complete(prompt, **k):
            raise llm.LLMError("opencode exited 1: auth expired")

        monkeypatch.setattr(codegen.llm, "complete", fake_complete)

        assert codegen.main_with(["--jobs", "1", "--limit", "5"]) == 1
        assert "aborting" in capsys.readouterr().out
        # only the first K=3 slugs got an .error marker
        markers = sorted(strategies.glob("*.error"))
        assert len(markers) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_codegen.py::TestMainCircuitBreaker -q`
Expected: FAIL — `assert 0 == 1`

- [ ] **Step 3: Wire the breaker into the `main_with` loop**

In `codegen.py`, restructure the executor block to (note the new `done` list):

```python
    print(f"codegen {len(todo)} specs ({args.jobs} at a time)\n")
    stages: dict[str, int] = {}
    done: list[dict] = []

    abort = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate_one, p, args.model,
                               args.llm_arg): p for p in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            done.append(result)
            abort = llm.circuit_break(done)
            if abort:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            spec_path = futures[fut]
            try:
                spec = yaml.safe_load(spec_path.read_text())
            except (yaml.YAMLError, OSError):
                spec = None  # error results never reach the spec-using path
            stage = publish(result, spec, STRATEGIES)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "coded" else "FAIL"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {spec_path.stem}")

    if abort:
        print(f"\n{abort}")
        return 1
```

(The trailing stage-count printing block stays unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_codegen.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add codegen.py tests/test_codegen.py
git commit -m "codegen: abort the batch after consecutive CLI-level failures"
```

---

### Task 5: Docs + full validation + PR

**Files:**
- Modify: `README.md` (the "Swapping the LLM provider" section)

- [ ] **Step 1: Document the breaker in README.md**

Append to the end of the "Swapping the LLM provider" section (after the paragraph about preflight):

```markdown
A batch also has a circuit breaker: if the first 3 completions in a run all
fail at the CLI level (binary missing, auth expired, timeout), the stage
aborts with one clear message and keeps the work already completed, instead
of writing an identical error artifact for every remaining item.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: all PASS (was 187 before this plan; +10 new tests)

- [ ] **Step 3: Live demonstration with a real missing CLI**

Run:
```bash
AUTOQUANT_LLM_CMD="definitely-not-a-real-cli-xyz -p" uv run python extract.py --limit 5
```
Expected: the preflight already stops this before the batch (`error: ... not found on PATH`, exit 1). The breaker covers the subtler case where the CLI exists but auth is broken — covered by the new stage tests.

- [ ] **Step 4: Commit docs**

```bash
git add README.md
git commit -m "docs: circuit breaker aborts a batch on consecutive CLI failures"
```

---

## Self-review notes

- **Spec coverage:** Issue #5 table row 2 ("a broken batch yields N identical .error markers -- no circuit-breaker") is now addressed: `circuit_break` trips on the first K=3 all-CLI-level results; stages cancel queued work, keep completed artifacts, print one clear message, exit 1. Rows 1 and 3 were already resolved by PR #7 (preflight, per-item graceful errors). The issue's "auth-style" heuristic is deliberately broadened to any CLI-level failure (missing binary, exit, timeout) — a timeout-storm is the same operational situation, and parse failures correctly do not trip it (covered by `test_breaker_spared_by_a_stage_level_error`).
- **Placeholder scan:** none — every step has full code.
- **Type consistency:** `circuit_break(results: list[dict], k: int = CIRCUIT_K) -> str | None` consumed identically by all three stages; stage result rows are `{"url", "error"}` (triage/extract) and `{"slug", "error"}` (codegen) — both are dicts with an `"error"` key, which is all the breaker inspects.
- **Determinism:** tests use `--jobs 1` so "first K completions" == first K todo items.
