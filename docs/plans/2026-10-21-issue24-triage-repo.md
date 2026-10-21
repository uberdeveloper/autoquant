# Issue #24: artifact triage — LLM runnability scoring for repo artifacts (notebooks/scripts/specs) — Implementation Plan

Part of #22 (external strategy-repo lane). Stage [3b], sitting between #23's
`harvest_repo.py` (which produced `data/repos/<owner>-<name>/manifest.json`) and
#25/#26/#27 (per-repo envs, offline data shim, run + promotion).

The lane's contract is **accept the repo as fact**: this stage scores
*runnability* — what is worth executing and how — not whether the strategy is
sensible, profitable, or correct.

## Stage placement and shape

`triage_repo.py` at the repo root, cloned structurally from `triage.py` (stage
[3]) because the issue requires reusing `llm.py` and the `triage_prompt.md`
pattern. A new prompt file `triage_repo_prompt.md` lives next to
`triage_prompt.md` and follows its format (role → input → evidence search
procedure → verdict table → required JSON output → field rules →
non-negotiable rules).

    python3 triage_repo.py                      # every harvested repo
    python3 triage_repo.py Auquan/Tutorials     # one repo (repeatable, nargs="*")
    python3 triage_repo.py --repos list.txt     # one spec per line
    python3 triage_repo.py --limit 20 --jobs 8  # calibrate first
    python3 triage_repo.py --rescore            # rubric changed: re-score all

## Input: manifest.json (produced by harvest_repo.py, merged in 61acfe0)

    {"repo": {"owner","name","url","sha","cloned_at"},
     "notebooks": [{"path","kernel","cells","has_outputs"}],
     "scripts":   [{"path","has_main"}],
     "specs":     [{"path"}],
     "data":      [{"path","bytes"}],
     "deps": {"requirements_txt","environment_yml","setup_py"},   # booleans
     "counts": {"notebooks","scripts","specs","data"}}

Manifest rows carry **metadata only, no content**, so `triage_repo.py` reads
artifact content itself from the clone dir. `.ipynb` files are JSON, so cell
source extraction is this stage's job.

### What gets scored (classification)

Exactly the three artifact kinds the issue names:

| manifest key | `kind` | row `meta` |
| --- | --- | --- |
| `notebooks` | `notebook` | `{"kernel","cells","has_outputs"}` |
| `scripts` | `script` | `{"has_main"}` |
| `specs` | `spec` | `{}` |

`data` rows are **not** scored (they are not executable artifacts; they exist in
the manifest for #26/#27's data shim). `deps` is not scored either — its text is
attached to the prompt as evidence.

### Enumeration rules

- With specs: each spec is parsed with `harvest_repo.parse_repo_spec` (raises
  `ValueError` on a malformed spec), located with `harvest_repo.repo_dir`, and its
  `manifest.json` must exist — otherwise `ValueError("no manifest for
  owner/name -- run harvest_repo.py first")`.
- Without specs: every child dir of `<root>/data/repos/` that holds a
  `manifest.json`, sorted. The `owner`/`name` label comes from the manifest's
  `repo` record (never from the directory name, which is ambiguous for colliding
  specs such as `a/b-c` vs `a-b/c` — the known #23 limitation).
- Specs are deduped with `dict.fromkeys`.
- A repo whose manifest record lacks owner/name is skipped silently by
  enumeration; a corrupt manifest raises `json.JSONDecodeError` (a `ValueError`)
  which `main_with` contains as a per-spec error.

## Verdict vocabulary and row schema

Appended to `data/repo_triage.jsonl`, one row per artifact, resumable.

    {"repo": "owner/name", "kind": "notebook", "path": "Python_Stock/Annual_Returns.ipynb",
     "meta": {"kernel": "python3", "cells": 29, "has_outputs": true},
     "verdict": "RUN_WITH_OUR_DATA",
     "symbols": ["SPY"],
     "data_loading": ["yf.download"],
     "data_mode": "network",            # network | bundled | both | none
     "dep_weight": "ordinary",          # ordinary | heavy
     "heavy_deps": [],
     "runtime_class": "seconds",        # seconds | minutes | hours
     "confidence": 0.9,
     "reason": "one line",
     "scored_at": "2026-10-21T...Z",
     "model": "the model id, or 'default'"}

Failure rows carry `{"repo","kind","path","meta","error"}` instead of the verdict
fields — same shape as `triage.py`'s error rows, and they still count as
"already scored" for resume.

Verdict definitions (the prompt file must state them exactly):

- `RUN_WITH_OUR_DATA` — runs against data this pipeline already serves: bare
  Yahoo tickers or `yahoo:`/`fred:`/`stooq:` catalog ids.
- `RUN_BUNDLED_DATA` — the artifact loads its own bundled files
  (`pd.read_csv` on a repo path, `pandas_datareader`, a pickle/h5/parquet in the
  repo). Nothing in the pipeline can feed those today; that is #26/#27.
- `NEEDS_HEAVY_ENV` — needs heavy deps (torch, tensorflow, gym, ta-lib) or pins
  that conflict with `pyproject.toml` (numpy < 1.26, pyyaml < 6). `ta-lib` needs
  a C library, which no per-repo venv fixes without a system package.
- `SPEC_ARTIFACT` — text/yaml describing a strategy with no executable code.
  Routes to promotion (#27), not execution.
- `INTERACTIVE_ONLY` — needs a human/UI/broker/notebook session (Jupyter
  interact(), broker API, plotting-only exploration).
- `SKIP` — nothing runnable (pure EDA, scraped data cleaning, docs, unrelated).

## Required public surface (tests import these)

    load(path: Path) -> list[dict]                      # jsonl rows, [] if absent
    save(path: Path, rows: list[dict]) -> None          # tmp + Path.replace, atomic
    artifact_rows(manifest: dict, repo: str) -> list[dict]
    notebook_source(path: Path) -> str                  # code + markdown cells joined
    artifact_text(dest: Path, row: dict) -> str         # dispatch on row["kind"]
    repo_deps(dest: Path, manifest: dict) -> dict       # {"requirements_txt","environment_yml"} text
    parse_json(text: str) -> dict                       # fence/prose tolerant (copy triage.py's)
    build_prompt(rubric: str, row: dict, text: str, deps: dict) -> str
    validate(result: dict) -> dict                      # normalized fields, raises ValueError
    score_one(dest: Path, row: dict, rubric: str, model: str | None,
              extra: list[str] | None = None) -> dict
    repo_manifests(root: Path, specs: list[str]) -> list[tuple[Path, dict, str]]
    main_with(argv: list[str] | None = None, root: Path | None = None) -> int

Module constants (tests redirect them, exactly like `tests/test_triage.py`):

    ROOT = Path(__file__).resolve().parent
    REPOS = ROOT / "data" / "repos"
    TRIAGE = ROOT / "data" / "repo_triage.jsonl"
    PROMPT = ROOT / "triage_repo_prompt.md"
    MAX_ARTIFACT_CHARS = 40_000     # a verdict needs the code, not every cell
    MAX_DEPS_CHARS = 4_000          # dep evidence is a pin list, not a novel
    CLI_TIMEOUT = 300

Imports: `import harvest_repo` for `parse_repo_spec` / `repo_dir` (reuse, do not
re-implement spec parsing), plus `llm`, `argparse`, `concurrent.futures`, `json`,
`re`, `sys`, `datetime`/`timezone`, `pathlib`. No new dependencies.

## Behavior contract

1. **Content extraction.** `notebook_source` reads the `.ipynb` as JSON and joins
   the `source` of every `code` and `markdown` cell with `"\n\n"`. `source` is
   either a string or a list of lines (both are legal nbformat) — handle both.
   `raw`/`output` cells are dropped. `artifact_text` dispatches on `kind`:
   notebook → `notebook_source`, script/spec → `read_text(errors="replace")`.
   An unreadable artifact (OSError, JSONDecodeError) becomes an error row
   prefixed `unreadable artifact:` and is visible in the summary.
2. **Prompt.** `build_prompt` is `triage.py`'s shape: rubric, `---`, the
   "return ONLY the JSON object" instruction, then the JSON payload
   `{"repo","kind","path","meta","repo_deps","text"}`. Text longer than
   `MAX_ARTIFACT_CHARS` is cut and suffixed `"\n\n[...truncated for length...]"`.
   `repo_deps` attaches the (truncated) text of `requirements.txt` /
   `environment.yml` when the manifest says they exist — dep-weight evidence for
   `NEEDS_HEAVY_ENV` must come from the repo, not from the model's memory.
3. **Validation.** `validate` normalizes the model's reply and raises
   `ValueError` (message starting `invalid `) on: a verdict outside the six,
   a `data_mode` / `dep_weight` / `runtime_class` outside its set, a non-numeric
   or out-of-range `confidence` (a `bool` is **not** a number here),
   `symbols` / `data_loading` / `heavy_deps` that are not lists of strings
   (`None` → `[]`, blank strings dropped), or an empty `reason`. `score_one`
   turns that into an error row. Error prefixes are deliberately chosen so
   `llm.is_cli_error` returns `False` for them — no `llm.py` change is needed and
   a bad reply can never trip the circuit breaker.
4. **LLM plumbing.** `llm.preflight()` first (`sys.exit` on `LLMError`);
   `ThreadPoolExecutor(max_workers=args.jobs)`; `llm.circuit_break(results)`
   checked after each future, `pool.shutdown(wait=False, cancel_futures=True)`
   then `return 1` with the abort message printed. Timeout `CLI_TIMEOUT`.
5. **Resume.** Key is `(repo, kind, path)`. `seen` is built from existing rows
   unless `--rescore`. Save replaces prior entries for re-scored keys rather than
   duplicating them (`triage.py:175-176`). Exit codes: `0` success or nothing to
   do, `1` on abort, bad spec, missing manifest, or an unreadable prompt file.
6. **Progress + summary.** `[i/N] VERDICT kind repo:path` per artifact, then a
   verdict-distribution line counted over **all** rows in the file (this is the
   issue's "verdict distribution sanity" acceptance item) and
   `scores -> data/repo_triage.jsonl`.

## README

Add one row to the Stages block (`README.md:18-31`), after the `triage.py` row:

    python3 triage_repo.py            # [3b] repo artifact runnability verdicts -> data/repo_triage.jsonl

## Tests: `tests/test_triage_repo.py` (no network, no LLM CLI)

Mirror `tests/test_triage.py`: no `conftest.py` exists in this repo, fixtures live
inside the test file; a `make_repo(tmp_path)` helper builds
`<root>/data/repos/<owner>-<name>/` with a manifest plus real artifact files;
`llm.complete` / `llm.preflight` / `llm.circuit_break` are monkeypatched; the
module constants `ROOT`, `TRIAGE`, `PROMPT` are redirected to `tmp_path` and
`main_with([...], root=tmp_path)` is used.

Cover at least:

1. `notebook_source` joins code+markdown cell sources; `source` as list and as
   string both work; `raw`/`output` cells excluded.
2. `artifact_rows` maps notebooks/scripts/specs with their `meta`, drops `data`
   rows, and labels the repo from the manifest's `repo` record.
3. `parse_json` handles bare, fenced, and prose-wrapped objects (and raises on
   garbage).
4. `build_prompt` starts with the rubric, contains the payload fields, appends
   the truncation marker past `MAX_ARTIFACT_CHARS`, and includes `repo_deps`.
5. `validate` accepts a full reply and rejects each of: unknown verdict, bad
   `data_mode`, `confidence` as bool/string/2.0, `symbols` as a string, empty
   `reason`.
6. `score_one` returns a normalized row carrying `repo`/`kind`/`path`/`model`;
   an unparseable reply yields `error` starting `unparseable reply:`; an
   `LLMError` yields the CLI message verbatim; an unreadable artifact yields
   `unreadable artifact:`.
7. `main_with`: resume skips scored artifacts (`nothing to triage`, exit 0);
   `--rescore` replaces the row (no duplicate keys in the file); `--limit` caps
   the batch; the circuit breaker aborts with exit 1, prints `aborting`, and
   keeps exactly the completed rows; a bad spec and a missing manifest each exit
   1 with a message on stderr; `preflight` failure exits with the LLM error.
8. The verdict-distribution summary line is printed for a mixed batch.

## Acceptance mapping (#24)

- [ ] Pilot manifests fully scored with no manual edits — `triage_repo.py` with no
      args over `data/repos/` (LastAncientOne 575 notebooks, Auquan 22, rotman 9
      scripts + 1 notebook). Run in batches with `--limit`; resume finishes the
      rest.
- [ ] Verdict distribution sanity — LastAncientOne yields a large
      `SKIP`/`INTERACTIVE_ONLY` tail, Auquan mostly `RUN_WITH_OUR_DATA` (the
      python2-kernel / old-pandas notebooks still get the flag), the hedging repo
      is `NEEDS_HEAVY_ENV`.
- [ ] `triage_repo_prompt.md` lives next to `triage_prompt.md` and follows its
      format.
