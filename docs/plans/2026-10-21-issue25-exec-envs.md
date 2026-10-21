# Issue #25: per-repo execution environments (uv venv builder + legacy fallback) — Implementation Plan

Part of #22 (external strategy-repo lane). Stage `[3c]`, sitting after #24's
`triage_repo.py` (which produced `data/repo_triage.jsonl`) and before #26's
offline data shim and #27's `run_external.py`.

The problem this stage solves: third-party repos carry conflicting
dependencies. Auquan is 2018-era pandas, the RL repo pins torch 1.11 + gym
0.24 + numpy 1.23 against our `pandas>=2.0` / `numpy>=1.26`. **Repo
dependencies must never reach the main project env or `uv.lock`.**

## Scope boundary (read this before reviewing)

Issue #25's acceptance says "Auquan notebooks execute under a pinned legacy
env". **Executing notebooks is #27's job** (`run_external.py`, nbclient). This
stage delivers the *environment* and proves it works with an import smoke
(`import pandas, numpy` succeeding at the pinned versions). The notebook
execution itself is verified in #27, and #25's plan is written so that the env
it builds is directly consumable by #27.

## Stage placement and shape

`env_repo.py` at the repo root, cloned structurally from `triage_repo.py` and
`harvest_repo.py` (the other `*_repo.py` lane stages). It imports
`harvest_repo` for `parse_repo_spec` / `repo_dir` — reuse, never re-implement.
It does **not** import `triage_repo`; the stage graph stays linear
(`harvest_repo` ← {`triage_repo`, `env_repo`}).

    python3 env_repo.py                          # every harvested repo
    python3 env_repo.py Auquan/Tutorials         # one repo (repeatable, nargs="*")
    python3 env_repo.py --repos list.txt         # one spec per line
    python3 env_repo.py --rebuild                # force rebuild even if the key matches
    python3 env_repo.py --legacy-python 3.9      # override the fallback interpreter

Output: `data/repo_envs.jsonl`, one row per repo, resumable (a repo whose
recorded env key still matches is reported as reused, not rebuilt).

## Input

`data/repos/<owner>-<name>/manifest.json` (produced by `harvest_repo.py`) plus
the dep files themselves, read from the clone dir. Enumeration rules mirror
`triage_repo.py` exactly:

- With specs: `harvest_repo.parse_repo_spec` (raises `ValueError` on a
  malformed spec) + `harvest_repo.repo_dir`; the manifest must exist, else
  `ValueError("no manifest for owner/name -- run harvest_repo.py first")`.
- Without specs: every child dir of `<root>/data/repos/` holding a
  `manifest.json`, sorted. The `owner`/`name` label comes from the manifest's
  `repo` record, never from the directory name.
- `dict.fromkeys` dedupe; a manifest lacking owner/name is skipped; a corrupt
  manifest raises `json.JSONDecodeError` which `main_with` contains per-spec.

## Dep detection

`deps_source(dest, manifest) -> str` returns the **first** of
`"requirements.txt"`, `"environment.yml"`, `"none"`, decided by the manifest's
`deps` booleans (the manifest is the contract, exactly as in #24's `repo_deps`).
A `setup.py` alone is **not** a pip-installable source: `setup_py: true` with no
`requirements.txt`/`environment.yml` still yields `"none"` and the repo runs in
the main env (recorded in the row so the decision is auditable).

- `read_requirements(path) -> list[str]` — strips comments and blank lines,
  keeps `-r`/`--index-url`/`-e` lines out (they are not installable pins and
  would make `uv pip install -r` fail on an unsupported flag).
- `pip_requirements_from_environment_yml(path) -> list[str]` — conda YAML:
  takes the `dependencies` list, keeps only the `str` entries, and for the
  `pip:` sub-mapping keeps its list entries. Conda-only entries (`numpy=1.23`)
  are dropped, and the row records how many were dropped so the loss is visible.

## Env reuse key

`env_key(sha: str, deps_text: str) -> str` = first 16 hex chars of
`sha256(f"{sha}\n{deps_text}".encode())`. The row stores `sha`, `deps_sha256`
(full hash of the dep file text), `deps_source` and `env_key`.

Rebuild happens when **any** of these differ from the recorded row:
`env_key` mismatch, `deps_source` mismatch, or the venv directory is missing on
disk (a deleted `.venv` self-heals, mirroring #23's manifest rule). `--rebuild`
forces it.

## Venv construction

Two uv invocations, both through a **single** subprocess call site
`run_uv(argv, timeout=UV_TIMEOUT) -> str` (the only `subprocess` call in the
module, mirroring `harvest_repo.run_git`), raising `RuntimeError` with stderr on
failure and on `subprocess.TimeoutExpired`.

    uv venv --no-project --python <py> <dest>/.venv
    uv pip install --python <dest>/.venv/bin/python -r <tmpfile>

- `--no-project` is **mandatory**: without it `uv venv` discovers the project
  and enforces `requires-python = ">=3.11"`, which would block the legacy
  fallback's older interpreter. (Verified by the orchestrator: `uv venv
  --no-project --python 3.9` works on this box.)
- `--python <venv>/bin/python` on the install step is **mandatory**: it is what
  keeps repo dependencies out of the project env. (Verified: installing
  `pandas==1.3.5` + `numpy==1.22.4` into a 3.9 venv left `uv.lock` byte-
  identical.)
- `-r <tmpfile>`: the filtered requirement list is written to a temp file in
  the repo's own dir (so relative paths in `-r`/`-e` lines resolve) and removed
  afterwards; `requirements.txt` is never modified.
- `UV_TIMEOUT = 1800` (30 min) — torch + dm-acme is a large resolve.
- Repos with `deps_source == "none"` get **no venv**; `mode` is `"main"` and the
  row records `venv: null`.

## Legacy fallback

Trigger, all three conditions:
1. the execution failed,
2. the failure is a pandas/numpy API error — `is_legacy_api_error(text) -> bool`
   matches a signature table (`DataFrame' object has no attribute 'append'`,
   `.ix` indexer, `iteritems`, `Panel`, `numpy.bool`/`bool8`, `np.float`,
   `np.int`, `np.object`, `np.str`, `No module named` is **explicitly not** a
   match — that is a missing dep, not an API break),
3. `deps_source == "none"` (the repo declared no pins, so a newer library
   version is the likely culprit).

On trigger: build `<dest>/.venv-legacy` with `--python LEGACY_PYTHON` and
install `LEGACY_PINS`, then return a row with `mode == "legacy-fallback"` and
`fallback_reason`. Exactly **one** retry — the fallback env is never retried
into a third env. Module constants:

    LEGACY_PYTHON = "3.9"
    LEGACY_PINS = ["pandas==1.3.5", "numpy==1.22.4", "scipy==1.7.3",
                   "scikit-learn==1.0.2", "matplotlib==3.5.3",
                   "statsmodels==0.13.2", "python-dateutil==2.9.0.post0",
                   "pytz", "pyyaml", "requests", "beautifulsoup4",
                   "pandas-datareader"]

(Orchestrator-verified ground truth: `pandas==1.3.5` + `numpy==1.22.4` import
cleanly on CPython 3.9.25 downloaded by uv; the remaining pins are the same
set at their last pandas-1.x-compatible versions and must be verified by the
writer with a real build before claiming acceptance.)

`env_repo.py` does not execute anything, so the trigger is a function
(`plan_fallback(row, error_text) -> dict | None`) that `run_external.py` (#27)
calls; #25 unit-tests the predicate and the fallback row shape, and the writer
documents the handoff in the module docstring.

## ta-lib detection

`talib_required(dest, manifest) -> list[dict]` scans the dep-file text **and**
every script/notebook source for `ta-lib`, `talib`, `import talib`,
`from talib`. Matches become `blocked` entries
`{"kind", "path", "reason": "ta-lib needs a C library (TA-Lib) that is not installed; not built"}`.

Decision, recorded so the reviewer does not flag it: **`env_repo.py` records
`blocked` in its own row and does not write to `data/repo_triage.jsonl`** —
`triage_repo.py` owns that file and its verdict vocabulary (single writer per
file). #27 applies the block by downgrading those artifacts to `SKIP` with the
recorded reason. #25's job is detection plus a durable, auditable record.

## Row schema

    {"repo": "Auquan/Tutorials", "path": "data/repos/Auquan-Tutorials",
     "mode": "repo-venv" | "main" | "legacy-fallback",
     "deps_source": "requirements.txt" | "environment.yml" | "none",
     "venv": "data/repos/Auquan-Tutorials/.venv" | null,
     "python": "3.11.2" | "3.9.25" | null,
     "sha": "<manifest sha>", "deps_sha256": "<full hash>", "env_key": "<16 hex>",
     "dropped_conda_deps": 0,
     "blocked": [{"kind","path","reason"}],
     "status": "ready" | "failed",
     "error": null | "<message>",
     "built_at": "<ISO8601 Z>"}

Failure rows keep the same identity fields with `status: "failed"` and `error`.

## Required public surface (tests import these)

`load`, `save`, `deps_source`, `read_requirements`,
`pip_requirements_from_environment_yml`, `env_key`, `talib_required`,
`is_legacy_api_error`, `plan_fallback`, `run_uv`, `build_env`,
`repo_manifests`, `main_with`, `main`, plus module constants `ROOT`, `REPOS`,
`ENVS`, `VENV_DIRNAME`, `LEGACY_VENV_DIRNAME`, `LEGACY_PYTHON`, `LEGACY_PINS`,
`UV_TIMEOUT`, `DEFAULT_TIMEOUT`.

`import harvest_repo` for `parse_repo_spec` / `repo_dir`.

## main_with contract

`main_with(argv: list[str] | None = None, root: Path | None = None) -> int`,
argparse (not Typer), flags `--repos FILE`, `--root`, `--rebuild`,
`--legacy-python`, `--jobs`, `--timeout`. Order of operations: parse →
enumerate → per repo: `build_env` (skipped when the key matches and not
`--rebuild`) → print a one-line progress row → collect → `save` (atomic
tmp + `Path.replace`, replacing prior rows for the same repo) → verdict counts
over `load(ENVS)` → `return 1` if any repo failed else `0`.

Per-repo containment: one repo's `RuntimeError`/`ValueError` becomes a `failed`
row plus a stderr line and does not abort the batch (mirrors #23/#24).
`ThreadPoolExecutor` is **not** used — env builds are heavy subprocess work and
uv's own cache already serialises the interesting part; keep it sequential and
honest.

## Tests — `tests/test_triage_repo.py`-style, in `tests/test_env_repo.py`

No conftest.py exists in this repo; fixtures live inside the test file.
`run_uv` is the only subprocess call site, so tests monkeypatch it (capturing
`argv`), never run real uv, and never touch the network. Redirect `ROOT` /
`REPOS` / `ENVS` to `tmp_path`. Cover at minimum:

1. `deps_source` — manifest flags win over the filesystem; `setup_py` alone →
   `"none"`.
2. `read_requirements` — comments, blanks, `-r`, `-e`, `--index-url` stripped.
3. `pip_requirements_from_environment_yml` — conda pins dropped, `pip:`
   sub-list kept, `dropped_conda_deps` counted.
4. `env_key` — stable for identical input, differs when sha or deps text
   changes, is 16 hex chars.
5. `build_env` — correct `uv venv --no-project --python` and
   `uv pip install --python <venv>/bin/python -r` argv, in that order; no venv
   for `deps_source == "none"`; the temp requirements file is removed.
6. Reuse — same key → no `run_uv` call at all; changed sha or deps → rebuild;
   missing venv dir on disk → rebuild; `--rebuild` forces it.
7. `is_legacy_api_error` — `DataFrame' object has no attribute 'append'` and
   `np.float` match; `No module named pandas` does **not**; a generic
   `ValueError` does not.
8. `plan_fallback` — only fires when the error matches **and**
   `deps_source == "none"`; returns `mode == "legacy-fallback"` with the legacy
   pins and python; returns `None` when the repo declared pins.
9. `talib_required` — a `requirements.txt` pin and an in-notebook
   `import talib` both produce a `blocked` entry with the C-library reason.
10. `main_with` — resume (unchanged key → reused, zero `run_uv` calls),
    `--rebuild`, per-repo containment (one repo's `RuntimeError` becomes a
    failed row and the others still build), exit code 1 on any failure, exit 0
    on all-ready, "nothing to build" exit 0, a malformed spec exits 1.
11. **The uv.lock invariant** — hash `uv.lock` before and after a mocked
    `build_env` run and assert it is unchanged (a regression guard on the
    acceptance criterion, even though the mock cannot violate it).
12. A real, non-mocked smoke the writer runs by hand and reports (not a unit
    test): build the Auquan legacy env and the rotman repo venv, and paste the
    `uv.lock` hash before/after.

## Conventions (AGENTS.md)

Smallest change; TDD with the failing test observed red first; `uv run` for
everything; **no new dependencies** (argparse + json + hashlib + pyyaml +
subprocess are all already available); no daemons; no executable bits; no
`# type: ignore`; atomic writes via tmp + `Path.replace`; `print()` progress,
no logging framework; module docstring with CLI usage; `from __future__ import
annotations`; `ROOT = Path(__file__).resolve().parent`.

## README

One new Stages row, immediately after the `triage_repo.py` line:

    python3 env_repo.py              # [3c] per-repo execution envs -> data/repo_envs.jsonl

## Files the writer may touch

`env_repo.py` (new), `tests/test_env_repo.py` (new), `README.md` (one row).
Nothing else — in particular not `harvest_repo.py`, `triage_repo.py`,
`pyproject.toml`, or `uv.lock`. Scratch work goes in `/tmp` only.
