# Plan — Issue #27: execution harness + external leaderboard + promotion (`run_external.py`)

Date: 2026-10-21 · Branch: `issue27-run-external` (off `main` @ `e1f4d50`)
Part of #22 (external strategy-repo lane). Stage: triage_repo [3b #24] → env_repo [3c #25]
→ offline shim [✅ #26] → **run_external [3c #27]** → epic #22 closes last.

## Goal

`run_external.py` reduces a triaged repo (`data/repo_triage.jsonl`) to **leaderboard rows +
saved executed notebooks**, and promotes interesting artifacts into `specs/ext-…​.yaml` so they
enter the existing extract→codegen→backtest lane. Follows the `backtest_batch.py` isolation model:
each artifact runs in its **own subprocess** with a **hard timeout**, and a re-run **skips** any
artifact that already has a **successful (`ok`)** leaderboard row.

## Hard constraints (binding)

- `uv` for everything; **NO new deps** (see "Notebook execution" — `nbclient`/`nbconvert`/`nbformat`
  are NOT in `uv.lock` and must not be added); `uv.lock` byte-identical
  (`sha256 ffdd71985af44fe4468be5182c05a44a72c3d208ee139ea6be780682f9a60e23`).
- No daemons; no executable bits; no `# type: ignore`; argparse (not Typer);
  `main_with(argv: list[str] | None = None, root: Path | None = None) -> int`.
- TDD, red-first. Tests at `tests/test_run_external.py` (mirrors `run_external.py`).
  **No `conftest.py`.** Tests must NOT hit network / real `uv` / real `uv LLM` — monkeypatch the
  subprocess seam (`_launch`/`subprocess`), `shim_env`, and any LLM call.
- One writer per data file: `run_external.py` OWNS `results/external/*` exclusively.
- Never write into `data/repos/` (the harvested checkouts) or the real `data/catalog/`.
- Promoted spec path: `uv run python backtest.py specs/<slug>.yaml` must not crash ("smoke").

## Contracts reused (verified on this box)

- **`backtest_batch.py`** — the runner to clone:
  - `DEFAULT_TIMEOUT = 600` (L29); `_kill_group` SIGKILLs the whole process group (L36-44).
  - `run_one`: `subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, cwd=ROOT, start_new_session=True)`
    then `proc.communicate(timeout=…)`; on `TimeoutExpired` kill group + reap. Fail shapes:
    `{"error": "could not launch…"/"timed out after Ns"/"exit N: …"}`.
  - `already_run(leaderboard) -> set[str]`: parse JSONL tolerantly, collect `row["slug"]` — the
    resume idiom. `record_failure` appends to a `failed.jsonl`.
  - `main_with(argv)`, `ThreadPoolExecutor(max_workers=args.jobs)`.
- **`env_repo.py`** (`data/repo_envs.jsonl`, ABSENT on this box → rebuild via `env_repo.py`):
  - Row keys: `repo, path, mode, deps_source, venv, python, sha, deps_sha256, env_key, …, blocked, status, error`.
  - **ROW CONTRACT**: a `status=="failed"` row has an EMPTY `blocked` — NEVER read `blocked` off a
    failed row. **MUST-HONOUR**: a non-empty `blocked` list ⇒ `#27 MUST downgrade those artifacts to
    SKIP` with the recorded reason.
  - Run cmd assembly (no run helper exists): `mode in ("repo-venv","legacy-fallback")` →
    `<root>/<row["venv"]>/bin/python <script>`; `mode == "main"` / `venv is null` → main project
    python `["uv","run","python",<script>]`.
- **`triage_repo.py`** — `VERDICTS = ("RUN_WITH_OUR_DATA","RUN_BUNDLED_DATA","NEEDS_HEAVY_ENV",
  "SPEC_ARTIFACT","INTERACTIVE_ONLY","SKIP")`. Row keys: `repo, kind, path, meta, verdict, symbols,
  data_loading, data_mode, dep_weight, heavy_deps, runtime_class, confidence, reason, scored_at, model`.
  Kinds: notebook / script / spec. Checkouts `data/repos/<owner>-<name>/` (ABSENT → re-harvest).
  Calibration (40 rows): RUN_WITH_OUR_DATA 16 (nb), RUN_BUNDLED_DATA 7 (nb), NEEDS_HEAVY_ENV 13
  (9 script + 4 nb), SPEC_ARTIFACT 2 (spec), SKIP 2.
- **Leaderboard** — `results/leaderboard.jsonl` (JSONL, append-only); reuse the JSONL + append +
  "row exists ⇒ done" idiom. #27 creates the **sibling** `results/external/leaderboard.jsonl`
  (fresh contract, below).
- **Specs lane** (promotion target) — canonical `_TEMPLATE.yaml`; `extract.validate_spec(doc, slug)`
  with `REQUIRED_SECTIONS = ("meta","data","signal","rules","costs","validation","verdict","ambiguities")`;
  rules: `meta.slug == filename`, `signal.lag_bars >= 1`, `commission+slippage > 0`, `ambiguities`
  non-empty, `validation.oos_start == meta.posted`, universe resolvable via `catalog.resolve`.
  **`backtest.py` loads `strategies/<slug>.py`** (hard-exits "missing implementation" if absent), so
  promotion must emit BOTH the spec and a strategy that passes `codegen.smoke_test`.
- **`shim.py` (#26)** — `shim_env(catalog_root=None, extra_path=None, existing=None) -> dict`
  returning `{"PYTHONPATH": <shim_dir>[:extra]…, "AUTOQUANT_CATALOG_DIR": <root>,
  "AUTOQUANT_SHIM_OFFLINE": "1"}`. Activate **only** for `RUN_WITH_OUR_DATA`; `RUN_BUNDLED_DATA` runs
  with **no** shim. `hard_offline_prefix()` returns `None` on this box (`unshare -n` unavailable) — do
  not depend on it.

## Design decisions (the writer must follow)

### 1. Isolation + run-copy (mirrors `backtest_batch.run_one`)
- Every notebook/script artifact is copied to a **run-copy** dir (stdlib `shutil.copytree` from
  `data/repos/<owner>-<name>/` into a temp dir, e.g. under the system temp or
  `results/external/_runs/<key>/`). **Never write into `data/repos/`.**
- Executed with `subprocess.Popen(..., start_new_session=True, cwd=<run-copy>)`, hard timeout via
  `proc.communicate(timeout=…)` + `_kill_group` (clone the backtest_batch helpers). Default timeout
  `DEFAULT_TIMEOUT = 600`, `--timeout` overrides. `--jobs N` concurrency via `ThreadPoolExecutor`.
- Env selection per artifact verdict:
  - `RUN_WITH_OUR_DATA` → `env = {**os.environ, **shim_env(catalog_root=<data/catalog>)}` (offline,
    serves our cached prices).
  - `RUN_BUNDLED_DATA` → plain `os.environ` (repo ships data; NO shim).
  - `NEEDS_HEAVY_ENV` → run only if a **ready** `env_repo` row exists for the repo (use its
    `<venv>/bin/python`); if the row is missing, `status=="failed"`, or has a non-empty `blocked`
    list → **downgrade to SKIP** with the recorded reason (honour the env_repo MUST-HONOUR contract).
  - `SPEC_ARTIFACT` → **no execution**; routed to promotion.
  - `SKIP` / `INTERACTIVE_ONLY` → stay SKIP (recorded, not run).
- Interpreter per env row `mode` (see Contracts): `repo-venv`/`legacy-fallback` → the row's venv
  python; `main` → `["uv","run","python", …]`.

### 2. Notebook execution (stdlib only — no new deps)
- `nbclient`/`nbconvert`/`nbformat` are NOT available and MUST NOT be added. Implement a
  **stdlib-only stateful executor** `_run_notebook(path, out_path, python, timeout, env)`:
  1. Read the `.ipynb` with stdlib `json`; take cells where `cell_type == "code"`.
  2. Run ONE subprocess: `<python> -c <runner>` (or a tiny temp `.py` runner) that, in a **single
     shared namespace**, `exec(compile(src, f"<cell {i}>", "exec"), ns)` per cell, capturing each
     cell's `stdout`/`stderr` with `contextlib.redirect_stdout/stderr`, then writes the executed
     `.ipynb` (original JSON + per-cell `outputs` = captured stream text) to `out_path`.
  3. The whole notebook is **one isolated subprocess** with the **total** hard timeout (the
     `backtest_batch` per-artifact model). **Documented limitation**: per-cell timeout is relaxed to
     a total-per-artifact timeout (nbclient is unavailable under the no-new-deps constraint). Cells
     that raise stop the run; the partial executed notebook + traceback are still saved as evidence.
- Evidence: executed notebook saved to `results/external/<repo>/<artifact-stem>.ipynb` (browsable),
  plus a `.log` of captured output. `cells_executed` = count of code cells run before completion/stop.
  An empty (0-byte) log is no evidence: it is removed and never listed in the row's `outputs`.

### 3. External leaderboard (fresh contract, JSONL)
`results/external/leaderboard.jsonl`, append-only history, **one logical row per artifact** (the
latest row of each `artifact_key`). Row keys:
```json
{"repo": "owner/name", "kind": "notebook|script|spec", "path": "<artifact path in repo>",
 "artifact_key": "<repo>|<kind>|<path>  (resume key)",
 "verdict": "ok|error|timeout|skip", "cells_executed": <int|null>,
 "runtime_sec": <float>, "env_id": "<env_key or 'main'|'none'>",
 "data_mode": "ours|bundled|none", "outputs": ["<relative produced files>"],
 "reason": "<skip/error reason or ''>", "run_at": "<iso z>"}
```
Resume: `already_run(results/external/leaderboard.jsonl) -> set[artifact_key]`; an artifact whose
latest row is `ok` is skipped on re-run (acceptance). `skip`/`error`/`timeout` rows are **not**
terminal — they are retried on the next run, and the leaderboard keeps their history (the file is
append-only), so the logical view (one row per `artifact_key`) is what callers read. A repeated,
unchanged verdict is not appended again, so a permanently skipped artifact does not grow the file.
Failures also appended to
`results/external/failed.jsonl`. `SPEC_ARTIFACT` rows get `verdict` set (e.g. `"spec"`) and route to
promotion without execution.

### 4. Promotion (`--promote <artifact>`) → specs + strategy
- Manual trigger (batchable later). Given a `SPEC_ARTIFACT` (or a promoted executed artifact), emit
  `specs/ext-<repo-slug>-<name-slug>.yaml` scaffolded from `_TEMPLATE.yaml`:
  `meta.slug == filename`; `data.universe` from triage `symbols` (resolve via `catalog.resolve` for
  `fred:`/`stooq:` ids / bare tickers); `signal.lag_bars >= 1`; `costs` NONZERO
  (`commission_bps + slippage_bps > 0`); `ambiguities` seeded non-empty (the real signal questions a
  human/LLM must answer); `validation.oos_start == meta.posted`; `verdict.status: PENDING`.
  `meta.source` records the origin repo + artifact path (provenance).
- ALSO emit `strategies/<slug>.py` — a placeholder implementation whose `signal()` returns a valid
  `pd.Series` on `codegen.smoke_frame()` so it passes `codegen.smoke_test` (the real strategy port is
  the human/LLM-reviewed follow-up; the scaffold makes `backtest.py specs/<slug>.yaml` run without
  crashing = "passes smoke validation", acceptance item 3).
- `--promote` does NOT itself run backtest (no network/LLM in this stage); it produces the spec +
  strategy that the existing lane consumes. Document that the placeholder signal is a scaffold, not a
  validated strategy (honest disclosure).
- The publication date is never invented: without a date in the triage row the promotion is refused
  — `--posted YYYY-MM-DD` supplies it (`meta.posted`, `validation.oos_start`). The value must be
  exactly an ISO date (`^\d{4}-\d{2}-\d{2}$`): a truncated/padded typo is refused, never truncated.
- Re-promoting an already promoted artifact is refused; `--force` overwrites it. A **failed**
  promotion (the scaffold fails the smoke test) leaves the tree exactly as it was: the new files are
  removed and a previous promotion is restored unchanged.

### 5. CLI
```
python run_external.py                       # run every triaged artifact not yet on the leaderboard
python run_external.py --jobs 4 --timeout 600
python run_external.py --repo owner/name     # restrict to one repo
python run_external.py --promote <path-or-key>   # scaffold spec+strategy for one artifact
python run_external.py --promote <path-or-key> --posted 2024-05-01   # explicit publication date
python run_external.py --promote <path-or-key> --force                # overwrite an existing promotion
```
`main_with(argv, root=None) -> int`; `root` scopes the data/results paths for tests (like
`env_repo`/`triage_repo`). Uses the triage file `data/repo_triage.jsonl` as input.

## Out of scope (documented, not built here)

- Real end-to-end execution needs `data/repos/` (re-harvest via `harvest_repo.py`) and
  `data/repo_envs.jsonl` (rebuild via `env_repo.py`) — both ABSENT on this box; that is an
  operational step, separate from landing the feature. Tests therefore run against synthetic
  artifacts + monkeypatched subprocess.
- Per-cell notebook timeout (nbclient unavailable). LLM/automated strategy port for promotion
  (kept manual-trigger). `unshare -n` hard-offline (unavailable here).

## Acceptance (map to tests)

- [ ] One command reduces a triaged repo to leaderboard rows + saved executed notebooks
      → `test_run_external.py` drives `main_with(["--repo","…"])` over a synthetic triaged repo and
      asserts `results/external/leaderboard.jsonl` rows + `results/external/<repo>/<stem>.ipynb` exist.
- [ ] Re-run skips completed artifacts → second `main_with` does not re-launch (subprocess seam not
      called for artifacts whose latest leaderboard row is `ok`; `skip`/`error`/`timeout` artifacts
      are retried and the leaderboard keeps one logical row per `artifact_key`).
- [ ] `--promote` produces a spec that passes `backtest.py` smoke validation → promoted spec passes
      `extract.validate_spec` and its `strategies/<slug>.py` passes `codegen.smoke_test` (both
      called directly in the test; `backtest.py` itself not invoked to avoid network).
- Extras: env-mode selection (ours/bundled/heavy/skip) honoured incl. the env_repo `blocked`→SKIP
  downgrade; timeout kills the process group; hard timeout; `uv.lock` unchanged.

## Roster (per handoff + this-box substitutions)

Writer `opencode-go/minimax-m3.1-flash-free`; Reviewer1 `opencode-go/deepseek-v4.1-flash`;
Reviewer2 `opencode-go/muse-spark-1.3-contributor`; Committer — `zai/glm-5.3-flash` has **no API key
on this box**, so per the user's rule the committer runs as `opencode-go/deepseek-v4.1-flash`.
Attribution line appended to every stage report / acceptance comment / log entry:
`Attribution: pi/glm-5.3-flash (orch) · opencode/minimax-m3.1 (W) · opencode/deepseek-v4.1 (R+commit) · opencode/muse-spark-1.3 (R)`.
