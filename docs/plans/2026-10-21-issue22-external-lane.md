# Issue #22 closeout -- the external strategy-repo lane

Date: 2026-10-21
Branch: `issue22-external-lane`
Status: sub-issues #23-#27 closed and merged; this document records the
operational closeout of the epic (#22) -- the acceptance evidence that the
three pilot repos reduce to leaderboard rows and that one promoted strategy
enters the existing backtest lane.

## The lane

| stage | script | artifact |
| --- | --- | --- |
| 1 harvest | `uv run python harvest_repo.py --repos <file>` | `data/repos/<owner>-<name>/manifest.json` |
| 2 triage | `uv run python triage_repo.py` | `data/repo_triage.jsonl` |
| 3 envs | `uv run python env_repo.py` | `data/repo_envs.jsonl` |
| 4 run | `uv run python run_external.py --jobs 4 --timeout 300` | `results/external/leaderboard.jsonl` |
| 5 promote | `uv run python run_external.py --promote <key> --posted YYYY-MM-DD` | `specs/ext-*.yaml` + `strategies/ext-*.py` |

`data/`, `results/`, `specs/` and `strategies/` are gitignored working
artifacts: the lane is reproducible, its outputs are not committed.

## Acceptance A -- the three pilot repos become leaderboard rows

Pilot repos: `LastAncientOne/Stock_Analysis_For_Quant`, `Auquan/Tutorials`,
`rotmanfinhub/gamma-vega-rl-hedging` (cloned from github.com without
credentials).

`results/external/leaderboard.jsonl` holds 66 raw lines -> 40 logical rows
(one per triaged artifact) via `latest_by_key`: the file is append-only
history, so superseded rows (the 26 pre-fix `exit 1: ` rows) stay on disk
while readers take the latest row per key.  Each logical row has the
`verdict`, `runtime_sec`, `env_id`, `data_mode`
and `outputs` keys (plus `repo`, `kind`, `path`, `artifact_key`,
`cells_executed`, `reason`, `run_at`).  An error row's `reason` carries the
traceback tail the artifact died on (with a `...(truncated) ...` marker when
the tail exceeds its ~300-char budget), so each row is diagnosable on its
own without opening the output files:

| repo | rows | verdicts |
| --- | --- | --- |
| Auquan/Tutorials | 23 | 20 error, 3 skip (2 triage SKIP, 1 SPEC_ARTIFACT) |
| LastAncientOne/Stock_Analysis_For_Quant | 6 | 6 skip (env row is ready but carries 93 blocked entries -- the env_repo MUST-HONOUR contract) |
| rotmanfinhub/gamma-vega-rl-hedging | 11 | 11 skip (10 env build failed, 1 SPEC_ARTIFACT) |

Totals: 0 ok, 20 error, 20 skip.  The 2 SPEC_ARTIFACT rows (the Auquan
README and the rotman README) are skips by design: they are routed to
promotion, not executed.

## Acceptance B -- a promoted strategy enters the existing lane

`uv run python run_external.py --promote "Auquan/Tutorials|spec|README.md"
--posted 2020-08-01` produced `specs/ext-Auquan-Tutorials-readme.yaml` and
`strategies/ext-Auquan-Tutorials-readme.py`; `validate_spec` returned no
error, `smoke_test` returned OK, and

```
uv run python backtest.py specs/ext-Auquan-Tutorials-readme.yaml
ext-Auquan-Tutorials-readme: Sharpe nan net (B&H 0.651), 0 trades
  FAIL: only 0 trades -- declining to conclude
```

The FAIL verdict is the expected outcome of the placeholder (flat) signal:
the acceptance is that the promoted spec passes through the lane, not that
the scaffold wins.

### Provenance of the publication date

The triage rows carry no publication date, so `--posted` was supplied from
the harvested clone's own git history (the clones are depth 1, so the
history holds a single commit -- the date is that commit's author date, and
the file was last touched in it):

```
git -C data/repos/Auquan-Tutorials log -1 --format='%H %as %s' -- README.md
b0bca8e42a1b8082c7897faa59077451231d9a2c 2020-08-01 Merge pull request #39 from yankovai-google/master
```

## Defects found and fixed (TDD, red first)

1. **The notebook runner died on Jupyter-only syntax.** `%matplotlib inline`,
   `!pip install ...` and `%%capture` are valid in a notebook but are not
   Python, so every notebook that used them failed with a `SyntaxError`
   instead of its own outcome. The runner now skips those lines.
   Red: `tests/test_run_external.py::TestNotebook::test_ipython_magics_and_shell_lines_are_skipped_not_fatal`.
2. **An env build failure buried its own cause.** The 60 kB uv log was copied
   into every leaderboard row of the failed repo. The reason now quotes the
   tail of the log, where the cause (`error: [Errno 2] No such file or
   directory: 'cmake'`) lives.
   Red: `tests/test_run_external.py::TestSelectEnv::test_heavy_env_failed_row_keeps_the_root_cause_of_a_long_log`.
3. **The unconditional magic strip broke valid Python.** A line that merely
   looks like a magic (`  % mean_squared_error(...))`, a modulo continuation
   inside parentheses) was deleted, turning 8 valid cells in 2 Auquan
   notebooks into `SyntaxError`; magic-looking lines inside triple-quoted
   strings were silently deleted (the wrong source ran), and `%%bash` /
   `%%writefile` bodies executed as Python. The runner now compiles the
   cell FIRST and runs it byte-for-byte unchanged if it compiles; only a
   failing cell has its magic/shell lines stripped and retried, and a cell
   magic whose body is not Python (`%%bash`, `%%writefile`, `%%HTML`) skips
   the whole cell (python-body magics -- `%%time`, `%%timeit`, `%%capture`,
   `%%prun` -- run their body).
   Red: `tests/test_run_external.py::TestNotebook::test_valid_python_with_magic_looking_lines_runs_unmodified`,
   `...::test_a_magic_looking_line_inside_a_string_literal_is_preserved`,
   `...::test_a_non_python_cell_magic_skips_the_whole_cell`.
4. **An error row's reason was empty.** The notebook runner wrote the
   traceback only to the log and the executed notebook, never to stderr, so
   every error row said `exit 1: ` with nothing after it and the cause lived
   only in the output files. The runner now writes the traceback to stderr
   (so the launcher's tail carries it into the row's `reason`), and the
   launcher's 300-char cut got the same `...(truncated) ...` marker rule
   `_error_tail` already applied to env-build logs.
   Red: `tests/test_run_external.py::TestMain::test_a_failing_notebook_row_carries_the_traceback_tail`,
   `tests/test_run_external.py::TestLaunch::test_a_long_stderr_tail_is_truncated_with_a_marker`.
5. **A ready env row's `blocked` list was ignored on the RUN_* verdicts.**
   The env_repo ROW CONTRACT makes a non-empty `blocked` list on a READY
   row the SKIP signal on every verdict, but `run_external.py` consulted it
   only on `NEEDS_HEAVY_ENV`; `RUN_WITH_OUR_DATA` / `RUN_BUNDLED_DATA`
   executed anyway. Those verdicts now downgrade to skip with a reason
   naming the blocked entries (LastAncientOne's ready row carries 93
   ta-lib blocks, so its 6 rows are skips, not errors).
   Red: `tests/test_run_external.py::TestSelectEnv::test_run_verdict_ready_row_with_blocked_downgrades_to_skip`,
   `tests/test_run_external.py::TestMain::test_a_blocked_env_row_downgrades_a_run_verdict_without_launching`.

## Epic open questions -- decisions

1. **Isolation: per-repo uv venv (`env_repo.py`), not Docker.**  The pilot
   artifacts are data-local (they read `data/repos/` and our catalog), and
   Docker is not available on this box; a per-repo venv gives dependency
   isolation where the artifacts actually run.  A stronger `unshare -n`
   hard-offline prefix exists as `hard_offline_prefix()` (`shim.py`): the
   `unshare` binary IS present on this box, but `unshare -n` fails with
   EPERM (unprivileged user namespaces are not permitted here), and
   `hard_offline_prefix()` is exposed API (`shim.py.__all__`) that
   `run_external.py` never calls -- unwired, not merely feature-flagged --
   so the shipped guarantee is the shim, not network-namespace isolation.
2. **Network policy: offline-by-default deterministic shim, not
   allow-fetch.**  `external_shim/` intercepts artifact fetches with a
   deterministic local shim (`AUTOQUANT_SHIM_OFFLINE` defaults to 1); a run
   that phones home is not reproducible, and reproducibility outweighs the
   convenience of live downloads.  The lane applies the shim
   (`shim.shim_env`) ONLY on `RUN_WITH_OUR_DATA` (`run_external.py:342`);
   `RUN_BUNDLED_DATA` and `NEEDS_HEAVY_ENV` run with plain `os.environ`.
   Of the 20 executed rows, 10 ran with the shim (`data_mode="ours"`) and
   10 ran without it (7 `bundled` + 3 `none`).
3. **Scale: curated breadth-first top-N, not run-all.**  Triage scored 40
   artifacts across the 3 pilot repos (breadth-first per repo: one repo
   holds 575 notebooks -- `LastAncientOne/Stock_Analysis_For_Quant`,
   verified with `find … -name '*.ipynb' | wc -l` -- but
   only a representative set is triaged and run).  Running everything would
   multiply LLM triage spend and harness wall-time without adding evidence
   the epic needs; a curated representative set answers "does the lane
   work" first.
4. **Success metric: "ran and produced output with a recorded verdict" is
   the epic's acceptance bar.**  Under that bar 0 ok + 20 error + 20 skip
   satisfies acceptance (A): every artifact produced a leaderboard row with
   a diagnosable verdict.  A "comparable backtest number" is the NEXT
   iteration's bar, once per-repo dependencies resolve (of the current 20
   errors, 19 are genuine missing-dependency rows -- `ModuleNotFoundError`
   -- and 1 is a `SyntaxError` in `Random Variables.ipynb`, a compile-
   context artifact rather than a missing dependency; none is silent).

## Deferred

* The pilot repos declare no requirements (`Auquan/Tutorials`,
  `LastAncientOne/...`), so they run in the project environment, which has no
  `matplotlib` / `scipy` / `statsmodels` and no `backtester` /
  `auquanToolbox`; their artifacts therefore end as honest `error` rows.
  Building a per-repo environment from scanned imports would fix this, but it
  is a new feature, not a defect fix.
* `rotmanfinhub/gamma-vega-rl-hedging` cannot be built at all here:
  `dm-acme[envs]==0.4.0` pulls `atari-py`, which needs `cmake`.
* The promoted spec is a scaffold: a human/LLM still has to transcribe the
  source strategy into `signal.definition` and the strategy body.
