# Handoff — autoquant issues #25–#22 delivery loop (machine migration, 2026-10-21)

Purpose: let a FRESH agent session on a NEW machine resume this work with nothing but
a `git clone`. Read this file plus the plan docs it points at before doing anything.

## Objective (user-requested, still open)

Complete GitHub issues #25 → #26 → #27 → #22 of this repo in dependency order using the
3-agent delivery loop below. Issues #23 and #24 are DONE. Epic #22 closes last.

Epic #22 = a second ingestion lane for third-party GitHub strategy repos, run against our
own data ("accept the repository content as a matter of fact" — score RUNNABILITY, do not
judge the repo's internal logic):

```
harvest_repo.py [3a, #23] → triage_repo.py [3b, #24] → env_repo.py [3c, #25]
  → #26 offline data shim → #27 run_external.py + external leaderboard + promotion into specs/
```

Same properties as the article lane (README Stages): every stage independent and
resumable, working artifacts stay out of Git, verdicts land in a leaderboard.

## Current state (as of this commit)

- main = `3da9773` (post-#24 merge; suite 365 passed).
- Branch `issue25-exec-envs`: plan doc `7ad00cb`, then a **wip commit** containing
  `env_repo.py`, `tests/test_env_repo.py` (102 tests), README Stages row. Full suite
  **467 passed, 1 warning** (the warning is the pre-existing Pandas4Warning at
  `backtest.py:70`).
- The wip commit is NOT the final PR state: **fix round 2 is pending** (list below).
- This handoff is committed on the same branch so one `git checkout` gets everything.

### Resume point — issue #25 fix round 2 (both round-2 reviews are in)

Two read-only reviewers reviewed the fix round; both said NEEDS FIXES with small,
precise lists. Consolidated work, in order:

**Product fixes (env_repo.py):**
1. N1 — `environment_python` (around env_repo.py:253-273) returns a conda matchspec
   verbatim, so `dependencies: [python=3.9.*]` reaches `uv venv --python '3.9.*'` and
   real uv 0.9.26 hard-fails (`No interpreter found for executable name '3.9.*'`) —
   the repo gets no env at all. Normalise: strip trailing `.*`, accept only
   `^\d+(\.\d+)*$`, else return `None` (fall back to `DEFAULT_PYTHON`).
2. N2 — `_environment_yml_requirements` (around env_repo.py:229) calls `yaml.safe_load`
   with no guard, unlike its sibling `environment_python`. `yaml.YAMLError` matches
   none of the per-repo containment types, so ONE malformed `environment.yml` aborts
   the whole batch with a traceback and NO output file (proven live). Guard it the
   same way and return `([], 0)`.
3. N3 — a declared-but-missing dep file raises before `state` is built, so the failed
   row records `deps_source: "none"` — the opposite of the truth for #27. Pass the
   declared source through to `_failed_row`.

**Doc fix:**
4. I4 — module docstring: rows built with explicit `requirements=` key `env_key`/
   `deps_sha256` over the joined pins (not file text), are REPLACED by later
   `main_with` runs (ephemeral); `mode` discriminates `python: null`; never read
   `blocked` off a `status == "failed"` row.

**Tests (tests/test_env_repo.py), TDD red-first:**
5. Wildcard conda pins: `3.9`, `3.9.*`, `3.*`, `3`, `3.9.7=h1234` (→ plain version or None).
6. Corrupt `environment.yml` → one `failed` row, sibling repo still `ready`, batch survives.
7. The #27 handoff shape: `build_env(mode="legacy-fallback", python="3.9",
   requirements=list(LEGACY_PINS))` asserting both argv calls, `.venv-legacy`, and the
   row's mode/venv/python/deps_sha256.
8. Selective run: two repos built, then `main_with(["Good/One"])` keeps `Bad/Two`'s row.
9. Precedence: an `environment.yml` with `python=3.9` + explicit `python="3.11"`
   builds with `3.9` (pins the documented order: legacy arg > conda pin > explicit > default).
10. Optional: replace the tautological `test_project_requirements_are_not_rewritten`
    with "source `requirements.txt` is byte-identical after the build".

Then: re-run both suites, hand to the committer (6-point checklist), PR (`Closes #25`),
merge with a merge commit (repo convention), close, post the acceptance summary.

Known acceptance limitation to record in the PR body: item (g) "RL repo env installs
torch+gym" is **unachievable on aarch64 boxes** — `dm-launchpad==0.5.0` ships only
`manylinux2010_x86_64` wheels (`dm-acme` depends on it) and `torch==1.11.0` has no
cp311 wheels (it resolves on 3.10; `gym==0.24.1` installs fine). Proven by direct
wheel checks; not a code defect.

## The delivery loop (per issue)

1. Plan doc `docs/plans/<date>-issueNN-<slug>.md` committed; branch `issueNN-<slug>` off main.
2. Writer implements TDD — failing test observed and reported honestly before each change.
3. Reviewers (parallel, read-only) map acceptance checkboxes to file:line, mutation-test
   in `/tmp` scratch copies, live-probe with monkeypatched seams. Reports go to the
   delivery archive (see below), never into this repo.
4. Max 2 fix rounds; the orchestrator independently re-verifies (suites, `git status`,
   `uv.lock` hash, live probe).
5. Committer runs the 6-point pre-commit checklist (git status exact, diff vs main,
   full suite, data-plane ignore check, `--help` + offline smoke, no scratch files),
   commits exact paths (NO `git add -A`), message `feat: <stage> — <what> (#NN)` with
   body `Closes #NN`, opens the PR.
6. Orchestrator merges (merge commit), closes the issue, posts the acceptance summary.

## Roster (user-set; each model smoke-tested before use)

| Role | Model |
|---|---|
| Writer | `opencode-go/minimax-m3.1-flash-free` (fallback `opencode-go/mimo-v2.6-flash`, proven on #23/#24) |
| Reviewer 1 | `opencode-go/deepseek-v4.1-flash` |
| Reviewer 2 | `opencode-go/muse-spark-1.3-contributor` |
| Committer | `zai/glm-5.3-flash` (user: "use zai/glm flash for committers stick to others as is") |

- Substitution rule (user): "if glm 5.3 fails use deepseek flash 4.1".
- NEVER use: `opencode-go/beam-free` (corrupted files with its own commentary),
  `opencode-go/longcat-2.5-preview-free` (silent no-output timeouts).
- Smoke before first real use: a trivial delegate task that writes `smoke ok\n` (exactly
  9 bytes) to `/tmp` and runs the focused suite — then `od -c` the file.

## Hard constraints

- `uv` for everything; NO new dependencies; no daemons; no executable bits; no
  `# type: ignore`; TDD; tests mirror `tests/<module>.py`; no conftest.py (fixtures
  live inside each test file); argparse not Typer; `main_with(argv, root=None) -> int`.
- Working artifacts NEVER committed: `data/`, `results/`, `specs/`, logs stay gitignored.
  Delivery logs/reviews live OUTSIDE the repo.
- `uv.lock` must stay byte-identical through env work: sha256
  `ffdd71985af44fe4468be5182c05a44a72c3d208ee139ea6be780682f9a60e23`.
  Every `uv venv --no-project` / `uv pip install --python <venv>` build must not move it.
- Network only for real clones / real env builds / LLM calls; tests never hit the
  network, real `uv`, or a real LLM (monkeypatch the single call sites:
  `run_git` / `run_uv` / `llm.complete`).
- This box is aarch64, no Docker, `unshare -n` blocked → #26 ships shim-only isolation
  with a documented limitation (decided).
- Never commit secrets. Provider keys live in `~/.pi/agent/auth.json`, never in files.

## Remaining issues after #25

- **#26 offline data shim** — stub yfinance/pandas-datareader so execution is forced
  offline; document the `unshare -n` limitation.
- **#27 run_external.py** — executes triaged artifacts in their envs (`env_repo.build_env`
  handoff shape is in the `env_repo.py` module docstring), applies ta-lib `blocked` →
  SKIP downgrades (MUST-HONOUR, `env_repo.py:37-39`), calls `plan_fallback` for the
  legacy retry exactly once, writes an external leaderboard row (status/runtime/env/
  data-mode/outputs; primary verdict "ran and produced output"), promotes winners into
  `specs/`.
- **#22** closes last; its acceptance includes the pilot run.

## Open items

1. #24 acceptance has ONE deliberately open checkbox: live pilot scoring of the
   remaining ~583 LastAncientOne artifacts (real LLM spend; 40-artifact calibration
   already done: NEEDS_HEAVY_ENV=13, RUN_WITH_OUR_DATA=16, RUN_BUNDLED_DATA=7, SKIP=2,
   SPEC_ARTIFACT=2 — distribution matched expectations). Decide with the user.
2. User instruction pending clarification: "we already have a installation with uv in
   HOME directory .venv/bin … use it for testing the repo (in case the repo dependencies
   are less than 3.11)". Finding: `~/.venv` is Python **3.13.5** (pandas 2.3.3), i.e.
   NOT <3.11, so it cannot serve as the legacy interpreter for pandas-1.x pins.
   Clarify intended use before implementing.
3. Scale decision from #22 (locked): curated top-N after triage, `--top N`, default 5.

## Delivery archive (outside git — rsync it to the new machine)

`~/autoquant-delivery/` (976K, 25 files at time of writing):
- `log.md` — chronological agent-attributed error/discovery/finding/decision entries.
- `issue-25/` — all four reviewer reports (round 1 + round 2, both reviewers).
- `probe25.py` — 27-check live-verification harness for `env_repo` (monkeypatches
  `run_uv`; re-run on the new box to re-verify).
- `delegates/` — raw delegate outputs + activity logs (provenance; `.session.jsonl`
  transcripts deliberately excluded).
- `HANDOFF.md` — the earlier free-form handoff (superseded by this file).

NOT carried: scratch venvs (`/tmp/i25-*`, `/tmp/r2*`), smoke byte-files, session
transcripts — all machine-local or re-creatable.

## New-machine setup checklist

1. Tools: pi 1.3.0, `uv` (0.9.26), `gh`, `git`, and the `opencode` CLI (v2.1.0+;
   `llm.py` shells out to `opencode run` — required for triage/pilot scoring).
2. Secrets: `~/.pi/agent/auth.json` (zai / opencode-go / openai-codex keys),
   `~/.pi/agent/{models-store.json,settings.json}`; `gh auth login` as `uberdeveloper`;
   `opencode` login. Copy files, never paste values into chat or repo.
3. `git clone https://github.com/uberdeveloper/autoquant.git && cd autoquant
   && git checkout issue25-exec-envs && uv sync`.
4. `uv run python -m pytest -q` → **467 passed, 1 warning**; `sha256sum uv.lock` →
   `ffdd71985af44fe4468be5182c05a44a72c3d208ee139ea6be780682f9a60e23`.
5. `rsync` the delivery archive (above). Optionally copy `data/repo_triage.jsonl`
   (40 scored artifacts = real LLM spend) or re-run `harvest_repo.py` + triage later.
6. Smoke-test each roster model once before real work.
