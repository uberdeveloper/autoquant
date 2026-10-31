#!/usr/bin/env python3
"""External execution harness + external leaderboard + promotion (#27).

Stage [3c] of the external strategy-repo lane (#22).  A repo scored by
``triage_repo.py`` (``data/repo_triage.jsonl``) is turned here into

* **executed artifacts** -- every notebook/script runs in its own subprocess,
  in a *copy* of the harvested checkout, with a hard timeout, and
* **leaderboard rows** -- ``results/external/leaderboard.jsonl``, an
  append-only history whose **logical** view is one row per artifact (the
  latest row of each ``artifact_key``; a repeated, unchanged verdict is not
  appended again), and
* **promoted specs** -- ``--promote`` scaffolds ``specs/ext-*.yaml`` plus a
  placeholder ``strategies/<slug>.py`` so the existing extract->codegen->
  backtest lane can pick the idea up.

The isolation model is cloned from ``backtest_batch.py``: one process per
artifact, ``start_new_session=True`` so the timeout kill reaches the whole
process group, and a JSONL leaderboard whose existing rows make a re-run skip
the artifact (resume).  Only a ``verdict: "ok"`` row is terminal: a
``skip``/``error``/``timeout`` row is evidence, and that artifact is retried
on the next run.  The leaderboard file stays append-only; ``latest_by_key``
turns its history into one verdict per artifact.

Hard rules this module keeps
-----------------------------
* ``data/repos/`` is read-only here: every artifact runs in a **run-copy**
  (``results/external/_runs/<key>/``), never in the original checkout.  The
  run copy holds no symlinks, so no write can reach the checkout through it.
* ``results/external/*`` belongs to this module alone.
* A promoted spec never carries an invented publication date: without a
  date in the triage row the promotion is refused (``--posted YYYY-MM-DD``).
* Notebook execution is **stdlib only** (``json`` + ``exec``): ``nbclient`` /
  ``nbconvert`` / ``nbformat`` are not available and must not be added.  The
  documented limitation is that the *per-cell* timeout of a real Jupyter
  kernel is relaxed to a **total per-artifact** timeout -- one hung cell eats
  the whole budget.  Jupyter-only syntax (a cell magic like
  ``%matplotlib inline``, a shell escape like ``!pip install ...``) is not
  Python: those lines are **skipped** by the runner, so such a notebook is
  executed instead of dying with a SyntaxError.

    python3 run_external.py                          # every triaged artifact
    python3 run_external.py --jobs 4 --timeout 600
    python3 run_external.py --repo owner/name
    python3 run_external.py --promote <path-or-key> --posted 2024-05-17
    python3 run_external.py --promote <path-or-key> --posted 2024-05-17 --force
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

import catalog
import codegen
import extract
import shim

ROOT = Path(__file__).resolve().parent
TRIAGE = ROOT / "data" / "repo_triage.jsonl"
ENV_ROWS = ROOT / "data" / "repo_envs.jsonl"
RESULTS = ROOT / "results" / "external"
LEADERBOARD = RESULTS / "leaderboard.jsonl"
FAILED = RESULTS / "failed.jsonl"
SPECS = ROOT / "specs"
STRATEGIES = ROOT / "strategies"
TEMPLATE = ROOT / "_TEMPLATE.yaml"
DEFAULT_TIMEOUT = 600

VERDICTS = ("RUN_WITH_OUR_DATA", "RUN_BUNDLED_DATA", "NEEDS_HEAVY_ENV",
            "SPEC_ARTIFACT", "INTERACTIVE_ONLY", "SKIP")


class Layout:
    """Every path this module touches, scoped by ``root``.

    ``root`` is the project root in production and a tmp dir in the tests:
    ``root/data`` holds the triage/envar rows and the harvested checkouts,
    ``root/results/external`` holds the leaderboard and the evidence, and
    ``root/specs`` / ``root/strategies`` hold the promotion output.
    """

    def __init__(self, root: Path | str = ROOT) -> None:
        self.root = Path(root)
        self.triage = self.root / "data" / "repo_triage.jsonl"
        self.env_rows = self.root / "data" / "repo_envs.jsonl"
        self.repos = self.root / "data" / "repos"
        self.catalog = self.root / "data" / "catalog"
        self.results = self.root / "results" / "external"
        self.leaderboard = self.results / "leaderboard.jsonl"
        self.failed = self.results / "failed.jsonl"
        self.runs = self.results / "_runs"
        self.specs = self.root / "specs"
        self.strategies = self.root / "strategies"

    def repo_dir(self, repo: str) -> Path:
        """The checkout dir of ``owner/name`` (``owner-name``)."""
        return self.repos / repo_slug(repo)


# ------------------------------------------------------------------ jsonl

def load_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader: blank/torn lines are skipped, never fatal."""
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def latest_by_key(rows: list[dict]) -> dict[str, dict]:
    """The **logical** leaderboard: one row per ``artifact_key``.

    ``leaderboard.jsonl`` is append-only history, so an artifact that was
    retried has several rows.  The last row per key is its current verdict;
    readers that want "one verdict per artifact" use this view.
    """
    out: dict[str, dict] = {}
    for row in rows:
        key = row.get("artifact_key")
        if key:
            out[key] = row
    return out


def already_run(leaderboard: Path) -> set[str]:
    """Artifact keys that already **completed** (``verdict == "ok"``).

    The resume idiom of ``backtest_batch.already_run``, keyed by
    ``artifact_key`` instead of ``slug``: a crash mid-write must not make
    every future resume raise.

    Only a successful row is terminal.  A ``skip`` row (e.g. a
    NEEDS_HEAVY_ENV artifact with no ``env_repo`` row yet), an ``error`` and
    a ``timeout`` are *evidence*, not completion -- they are retried on the
    next run, so an artifact that waited for an environment can finally run
    once ``data/repo_envs.jsonl`` grows a ready row.

    The verdict of an artifact is its **latest** row (see
    ``latest_by_key``): an artifact is done only when its current verdict is
    ``ok``.
    """
    return {key for key, row in latest_by_key(load_jsonl(leaderboard)).items()
            if row.get("verdict") == "ok"}


def record_failure(row: dict, path: Path) -> None:
    """Append one failure row to the failed log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def append_row(row: dict, path: Path) -> None:
    """Append one leaderboard row.

    The file stays **append-only** (other readers may rely on the history),
    but the logical leaderboard is *one row per artifact* (see
    ``latest_by_key``).  A repeated, unchanged verdict -- the same skip
    reason on every run of a permanently skipped artifact -- is therefore
    NOT appended again, so the leaderboard does not grow without bound
    (three runs of a skipped artifact used to leave three rows).  A changed
    verdict (skip -> error -> ok) IS appended: the history then shows how
    the artifact behaved.
    """
    key = row.get("artifact_key")
    if key:
        previous = latest_by_key(load_jsonl(path)).get(key)
        if (previous is not None
                and previous.get("verdict") == row.get("verdict")
                and previous.get("reason") == row.get("reason")):
            return                      # the logical verdict is unchanged
    record_failure(row, path)


# ------------------------------------------------------------------- keys

def artifact_key(row: dict) -> str:
    """The resume key of an artifact: ``<repo>|<kind>|<path>``."""
    return f"{row.get('repo', '')}|{row.get('kind', '')}|{row.get('path', '')}"


def repo_slug(repo: str) -> str:
    """``owner/name`` -> ``owner-name`` (a safe directory name)."""
    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in repo)
    return slug.strip("-") or "repo"


def slugify(text: str) -> str:
    """A lowercase, dash-separated slug of any text."""
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "-".join("".join(out).split("-"))
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def now_iso() -> str:
    """UTC timestamp, ISO-8601 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------- env choice

def interpreter(env_row: dict, layout: Layout) -> list[str]:
    """The command prefix that runs an artifact under an env row.

    ``repo-venv`` / ``legacy-fallback`` rows carry a real interpreter
    (``<root>/<venv>/bin/python``); ``main`` rows and rows without a venv run
    in the main project env (``uv run python``).
    """
    mode = env_row.get("mode")
    venv = env_row.get("venv")
    if mode in ("repo-venv", "legacy-fallback") and venv:
        return [str(layout.root / venv / "bin" / "python")]
    return ["uv", "run", "python"]


def env_rows_by_repo(path: Path) -> dict[str, dict]:
    """Index the env_repo rows by repo label (a ready row wins)."""
    out: dict[str, dict] = {}
    for row in load_jsonl(path):
        repo = row.get("repo")
        if not repo:
            continue
        prev = out.get(repo)
        if prev is None or (prev.get("status") != "ready"
                            and row.get("status") == "ready"):
            out[repo] = row
    return out


def _error_tail(error: str | None, limit: int = 600) -> str:
    """The end of a long build log -- that is where the cause is.

    A failed env build keeps the whole uv log (tens of kB) in
    ``data/repo_envs.jsonl``; copying it into every leaderboard row buries
    the actual cause under the download noise, so only the tail is quoted.
    """
    text = (error or "").strip()
    if not text:
        return "unknown error"
    if len(text) <= limit:
        return text
    return "...(truncated) " + text[-limit:]


def _blocked_reason(repo: str, blocked: list) -> str:
    """A bounded skip reason that still names the blocked entries.

    A ready row's ``blocked`` list can hold an entire repo's block dump
    (93 dict reprs in production); joining it unbounded copied 16 kB into
    every skip row of that repo -- 71% of the leaderboard for none of the
    skipped artifacts.  Only the tail of the list is quoted, with the same
    ``_error_tail`` marker rule as env-build logs.
    """
    entries = "; ".join(str(b) for b in blocked)
    return "env for %s is blocked: %s" % (repo, _error_tail(entries,
                                                            limit=300))


def select_env(row: dict, env_rows: dict[str, dict],
               layout: Layout) -> dict:
    """Pick the interpreter + environment for one triaged artifact.

    Returns ``{"run": bool, "reason": str, "env": dict, "env_id": str,
    "data_mode": str, "python": list[str]}``.

    * ``RUN_WITH_OUR_DATA`` -> the offline shim over our own price cache.
    * ``RUN_BUNDLED_DATA``  -> plain environment, the repo ships its data.
    * ``NEEDS_HEAVY_ENV``   -> only with a **ready** env_repo row; a missing
      row, a ``failed`` row, or a row with a non-empty ``blocked`` list
      downgrades the artifact to SKIP with the recorded reason.  A ``failed``
      row has an EMPTY ``blocked`` list -- ``blocked`` is never read off it.
    * ``SPEC_ARTIFACT``     -> not executed here; it is routed to promotion.
    * ``SKIP`` / ``INTERACTIVE_ONLY`` -> stay skipped.
    """
    verdict = row.get("verdict", "")
    repo = row.get("repo", "")
    env_row = env_rows.get(repo)
    base = dict(os.environ)

    def skip(reason: str, data_mode: str = "none") -> dict:
        return {"run": False, "reason": reason, "env": None, "env_id": "none",
                "data_mode": data_mode, "python": []}

    if verdict in ("SKIP", "INTERACTIVE_ONLY"):
        return skip(f"triage verdict {verdict}")
    if verdict == "SPEC_ARTIFACT":
        return skip("SPEC_ARTIFACT -- routed to promotion, not executed")
    if verdict not in ("RUN_WITH_OUR_DATA", "RUN_BUNDLED_DATA",
                       "NEEDS_HEAVY_ENV"):
        return skip(f"unknown triage verdict {verdict!r}")

    if verdict == "NEEDS_HEAVY_ENV":
        if env_row is None:
            return skip(f"no env_repo row for {repo} -- heavy deps need a "
                        "built environment")
        status = env_row.get("status")
        if status == "failed":
            return skip(f"env build failed for {repo}: "
                        f"{_error_tail(env_row.get('error'))}")
        if status != "ready":
            return skip(f"env row for {repo} is not ready (status={status!r})")
        blocked = env_row.get("blocked") or []
        if blocked:
            return skip(_blocked_reason(repo, blocked))
        return {"run": True, "reason": "", "env": base,
                "env_id": env_row.get("env_key") or "main",
                "data_mode": "none", "python": interpreter(env_row, layout)}

    if verdict == "RUN_WITH_OUR_DATA":
        env = {**base, **shim.shim_env(catalog_root=layout.catalog)}
        data_mode = "ours"
    else:
        env = base
        data_mode = "bundled"

    if (env_row is not None and env_row.get("status") == "ready"
            and (env_row.get("blocked") or [])):
        # C3 / env_repo ROW CONTRACT: a non-empty `blocked` list on a READY
        # row is the SKIP signal on every verdict, RUN_* included.
        return skip(_blocked_reason(repo, env_row["blocked"]),
                    data_mode=data_mode)

    if (env_row is not None and env_row.get("status") == "ready"
            and not (env_row.get("blocked") or [])):
        return {"run": True, "reason": "", "env": env,
                "env_id": env_row.get("env_key") or "main",
                "data_mode": data_mode, "python": interpreter(env_row, layout)}
    return {"run": True, "reason": "", "env": env, "env_id": "main",
            "data_mode": data_mode, "python": ["uv", "run", "python"]}


# ------------------------------------------------------------- isolation

def _kill_group(pid: int) -> None:
    """SIGKILL the whole process group.

    The child runs in its own session (``start_new_session=True``), so the
    timeout kill has to reach the whole group: killing only the launcher
    would leave the real work running, free to write output the run has
    already recorded as failed.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass  # already gone, or re-parented before we got to it


def _as_text(value: object) -> str:
    """Whatever ``Popen`` handed back as text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _launch(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
            timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run one command in an isolated process with a hard timeout.

    Returns ``{"ok", "error", "timeout", "stdout", "stderr"}``.  A timeout
    kills the whole process group and reaps it; it is never raised.
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                cwd=str(cwd) if cwd is not None else None,
                                env=env, start_new_session=True)
    except OSError as exc:
        return {"ok": False, "error": f"could not launch {cmd[0]}: {exc}",
                "timeout": False, "stdout": "", "stderr": ""}
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Keep whatever the child managed to print before it hung: a
        # timeout with no output is a timeout with no diagnostic.
        out = _as_text(exc.output)
        err = _as_text(exc.stderr)
        _kill_group(proc.pid)
        try:
            more_out, more_err = proc.communicate(timeout=10)  # reap the group
            out += _as_text(more_out)
            err += _as_text(more_err)
        except subprocess.TimeoutExpired as exc2:
            out += _as_text(exc2.output)
            err += _as_text(exc2.stderr)
            # A straggler survived the group kill: kill the launcher itself
            # and wait a bounded time, then abandon it -- an unbounded wait
            # would hang the whole batch.
            try:
                proc.kill()
            except (OSError, AttributeError):
                pass
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, AttributeError):
                pass
        return {"ok": False, "error": f"timed out after {timeout}s",
                "timeout": True, "stdout": out, "stderr": err}
    if proc.returncode != 0:
        return {"ok": False,
                "error": f"exit {proc.returncode}: "
                         f"{_error_tail(stderr, limit=300)}",
                "timeout": False, "stdout": stdout, "stderr": stderr}
    return {"ok": True, "error": "", "timeout": False, "stdout": stdout,
            "stderr": stderr}


def _make_run_copy(src: Path, dest: Path) -> Path:
    """Copy a checkout into a scratch dir -- the artifact never runs in place.

    ``data/repos/`` is read-only for this module: external code writes
    outputs, caches and stray files wherever it likes, and those must not
    land in the harvested checkout.

    Symlinks are **not** copied as symlinks.  A preserved symlink would let
    the artifact write *through* the copy back into the harvested checkout
    (the exact thing this function exists to prevent), and a dereferenced
    symlink could point anywhere on the disk.  A symlink whose target is a
    regular file INSIDE the checkout is copied as a plain file (its content
    is snapshotted); every other symlink is dropped, so the run copy is
    self-contained and no write can reach ``data/repos/`` through it.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    root = src.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel = Path(dirpath).relative_to(root)
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for name in list(dirnames):
            if (Path(dirpath) / name).is_symlink():
                dirnames.remove(name)          # dropped: never followed
        for name in filenames:
            source = Path(dirpath) / name
            target = dest / rel / name
            if source.is_symlink():
                resolved = source.resolve()
                if resolved.is_file() and resolved.is_relative_to(root):
                    shutil.copy2(resolved, target)
                continue                      # dropped: a symlink out of the tree
            shutil.copy2(source, target)
    return dest


def _file_set(directory: Path) -> set[str]:
    """The relative paths of every file under ``directory``."""
    out: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_file():
            out.add(str(path.relative_to(directory)))
    return out


# ------------------------------------------------------- notebook execution

# The stdlib-only notebook runner.  It is handed to the interpreter as
# ``python -c <this>``; the arguments are ``<notebook> <out.ipynb> <out.log>``.
#
# LIMITATION (documented, deliberate): nbclient/nbconvert/nbformat are NOT
# available and must not be added, so there is no real Jupyter kernel.  The
# cells are executed in ONE subprocess, in ONE shared namespace, and the
# timeout is a **total per-artifact** budget -- a per-cell timeout (as
# nbclient offers) is not possible here.  A cell that raises stops the run;
# the partial executed notebook and the traceback are still written as
# evidence.
#
# Jupyter-only lines (a cell magic "%matplotlib inline", a shell escape
# "!pip install ...") are SKIPPED by the runner: they are valid notebook
# syntax but not Python, and dying on them would hide the artifact's real
# outcome behind a SyntaxError.
_NOTEBOOK_RUNNER = r'''
import contextlib
import io
import json
import re
import sys
import traceback

nb_path, out_path, log_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(nb_path, "r", errors="replace") as fh:
    nb = json.load(fh)
if not isinstance(nb, dict):
    nb = {}
nb.setdefault("metadata", {})
cells_note = None
if not isinstance(nb.get("cells"), list):
    # ``cells`` missing or null (a malformed notebook must not crash the
    # runner): record why and run nothing.
    cells_note = "the notebook has no 'cells' list -- nothing to execute"
    nb["cells"] = []

ns = {"__name__": "__main__"}
executed = 0
failure = None

# A Jupyter-only line: a line magic ("%matplotlib inline") or a shell
# escape ("!pip install ...").  They are valid notebook syntax but not
# Python, so they are dropped from a cell that fails to compile.
_MAGIC_RE = re.compile(r"^[ \t]*[%!]")

# Cell magics whose BODY is Python: the magic line is dropped and the body
# runs.  Any other cell magic (%%bash, %%writefile, %%HTML, ...) has a
# non-Python body and its cell is skipped whole -- executing a shell script
# or HTML as Python would be wrong.
_PYTHON_CELL_MAGICS = ("time", "timeit", "capture", "prun")


def _prep(src):
    """Python source for one cell body, or None if it cannot be made Python.

    compile-first: a cell whose source is valid Python runs byte-for-byte
    unchanged -- a line that merely LOOKS like a magic (a `%` modulo
    continuation, magic-looking text inside a triple-quoted string) is
    preserved.  Only a cell that fails to compile has its magic/shell lines
    stripped, and it is retried as-is after that.
    """
    try:
        compile(src, "<cell>", "exec")
        return src
    except SyntaxError:
        pass
    return "\n".join(ln for ln in src.splitlines()
                     if not _MAGIC_RE.match(ln))


def _src(cell):
    """The Python to run for one cell, or None if the cell is skipped."""
    src = cell.get("source")
    if isinstance(src, list):
        src = "".join(src)
    if not isinstance(src, str):
        return ""
    first = src.lstrip().split("\n", 1)[0]
    if first.startswith("%%"):
        name = first[2:].split()[0] if first[2:].split() else ""
        if name not in _PYTHON_CELL_MAGICS:
            return None               # non-Python body: skip the whole cell
        body = src.split("\n", 1)[1] if "\n" in src else ""
        prepared = _prep(body)
        return "" if prepared is None else prepared
    prepared = _prep(src)
    return "" if prepared is None else prepared


def _flush():
    """Persist the executed notebook (partial evidence on a crash)."""
    nb["metadata"]["autoquant"] = {"cells_executed": executed,
                                   "error": failure}
    if cells_note:
        nb["metadata"]["autoquant"]["note"] = cells_note
    with open(out_path, "w") as fh:
        json.dump(nb, fh, indent=1)


log = open(log_path, "w")
if cells_note:
    log.write(cells_note + "\n")
try:
    for i, cell in enumerate(nb["cells"]):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        out, err = io.StringIO(), io.StringIO()
        src = _src(cell)
        if src is None:
            continue          # a non-Python cell magic: skipped, not executed
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(src, "<cell %d>" % i, "exec"), ns)
        except BaseException:
            failure = traceback.format_exc()
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": out.getvalue()},
                {"output_type": "stream", "name": "stderr", "text": err.getvalue()},
                {"output_type": "stream", "name": "traceback", "text": failure},
            ]
            executed += 1
            log.write(out.getvalue())
            log.write(err.getvalue())
            log.write(failure)
            _flush()
            # C1: the leaderboard row's reason is built from the launcher's
            # stderr tail, so the traceback MUST reach stderr -- writing it
            # only to the log left every error row an empty "exit 1: ".
            sys.stderr.write(failure)
            sys.exit(1)
        executed += 1
        cell["outputs"] = [
            {"output_type": "stream", "name": "stdout", "text": out.getvalue()},
            {"output_type": "stream", "name": "stderr", "text": err.getvalue()},
        ]
        log.write(out.getvalue())
        log.write(err.getvalue())
        _flush()
    _flush()
finally:
    log.close()
'''


def _run_notebook(nb_rel: str, out_ipynb: Path, out_log: Path,
                  python_cmd: list[str], timeout: int, env: dict,
                  cwd: Path) -> dict:
    """Execute one notebook in a single isolated subprocess.

    ``nb_rel`` is relative to ``cwd`` (the run copy); the outputs are written
    to ``out_ipynb`` (the executed notebook) and ``out_log`` (the captured
    output).  Returns ``{"ok", "error", "timeout", "cells_executed"}``.
    """
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    cmd = list(python_cmd) + ["-c", _NOTEBOOK_RUNNER, nb_rel,
                              str(out_ipynb), str(out_log)]
    res = _launch(cmd, cwd=cwd, env=env, timeout=timeout)
    if res.get("stderr"):
        with out_log.open("a") as fh:
            fh.write(res["stderr"])
    cells = _executed_cells(out_ipynb)
    return {"ok": res["ok"], "error": res["error"], "timeout": res["timeout"],
            "cells_executed": cells}


def _executed_cells(out_ipynb: Path) -> int | None:
    """The number of code cells the runner executed, from its own bookkeeping."""
    if not out_ipynb.exists():
        return None
    try:
        doc = json.loads(out_ipynb.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None
    meta = doc.get("metadata")
    if isinstance(meta, dict):
        auto = meta.get("autoquant")
        if isinstance(auto, dict) and isinstance(auto.get("cells_executed"), int):
            return auto["cells_executed"]
    return None


def _run_script(script_rel: str, python_cmd: list[str], timeout: int, env: dict,
                cwd: Path) -> dict:
    """Execute one script in a single isolated subprocess."""
    cmd = list(python_cmd) + [script_rel]
    res = _launch(cmd, cwd=cwd, env=env, timeout=timeout)
    return {"ok": res["ok"], "error": res["error"], "timeout": res["timeout"],
            "cells_executed": None, "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", "")}


# ------------------------------------------------------------------ promotion

STRATEGY_SCAFFOLD = '''"""SCAFFOLD for {slug} -- NOT a port of the source strategy.

Generated by ``run_external.py --promote``.  ``signal`` returns a flat
(all-zero) position on every bar: it exists so the promoted spec can be
loaded and smoke-tested by the existing lane, NOT because the source idea has
been translated.  A human (or an LLM) must replace this body -- and the
spec's ``signal.definition`` -- with the real rule before any result is
believed.

Provenance: {repo} / {path}
"""
from __future__ import annotations

import pandas as pd


def signal(df, **params):
    """Placeholder signal: flat exposure on every bar.

    ``df``     -- an OHLCV frame (single ticker) or a ``{{ticker: frame}}``
                  mapping (multi-ticker universe).
    ``params`` -- the spec's signal parameters; unused by the placeholder.
    """
    if isinstance(df, dict):
        frames = {{k: v for k, v in df.items() if isinstance(v, pd.DataFrame)}}
        if not frames:
            return pd.DataFrame()
        index = next(iter(frames.values())).index
        return pd.DataFrame(0.0, index=index, columns=list(frames))
    if isinstance(df, pd.DataFrame):
        return pd.Series(0.0, index=df.index, dtype=float)
    raise TypeError("signal() expects a DataFrame or a mapping of DataFrames")
'''


def _posted_from_meta(meta: dict) -> str | None:
    """The raw publication date carried by the triage row, if any."""
    for key in ("posted", "date", "published", "published_at"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iso_date(raw: str) -> str | None:
    """Normalize a date-ish string to ISO ``YYYY-MM-DD``; None when unusable.

    The value must be *exactly* an ISO date: the regex rejects a truncated
    or padded value (``"2024-01-029"``, ``"2024-01-02junk"``, ``"20240102"``)
    before ``date.fromisoformat`` ever sees it, and ``fromisoformat`` then
    rejects the impossible ones (``"2024-02-30"``).  A typo'd date is
    refused, never silently truncated into a different date.
    """
    text = str(raw).strip()
    if not text or not _ISO_DATE.fullmatch(text):
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _resolve_posted(meta: dict, override: str | None) -> tuple[str | None, str | None]:
    """The artifact's publication date, or ``(None, error message)``.

    An explicit ``--posted`` wins; otherwise the triage row's date is used.
    There is **no** fallback to today: a fabricated publication date makes
    the promoted spec claim an out-of-sample window it does not have.
    """
    if override is not None:
        parsed = _iso_date(override)
        if parsed is None:
            return None, (f"invalid --posted value {override!r}: expected an "
                          "ISO date (YYYY-MM-DD)")
        return _not_in_the_future(parsed)
    raw = _posted_from_meta(meta)
    if raw is None:
        return None, ("no publication date for this artifact -- pass "
                      "--posted YYYY-MM-DD (the triage row carries no date, "
                      "and today is never used as a stand-in)")
    parsed = _iso_date(raw)
    if parsed is None:
        return None, (f"invalid publication date {raw!r} for this artifact -- "
                      "pass --posted YYYY-MM-DD")
    return _not_in_the_future(parsed)


def _not_in_the_future(parsed: str) -> tuple[str | None, str | None]:
    """Refuse a publication date that has not happened yet.

    An out-of-sample window that starts in the future contains no data: the
    backtest would report an empty result and call it out-of-sample.
    """
    if parsed > date.today().isoformat():
        return None, (f"publication date {parsed} is in the future -- an "
                      "out-of-sample window cannot start before the artifact "
                      "was published")
    return parsed, None


def _universe(symbols: object, notes: list[str]) -> list[str]:
    """The spec's ``data.universe``, from the triage row's symbols.

    ``fred:``/``stooq:`` ids must resolve in the catalog (the spec validator
    requires it); bare tickers are kept even when the catalog does not know
    them yet, and every drop/soft spot is recorded in ``notes``.
    """
    picked: list[str] = []
    for symbol in symbols if isinstance(symbols, list) else []:
        if not isinstance(symbol, str):
            continue
        text = symbol.strip()
        if not text:
            continue
        if ":" in text:
            source = text.split(":", 1)[0].lower()
            if source not in ("yahoo", "fred", "stooq"):
                notes.append(f"dropped symbol {text!r}: unknown data source")
                continue
            if source != "yahoo":
                try:
                    catalog.resolve(text)
                except ValueError:
                    notes.append(f"dropped symbol {text!r}: not in the catalog -- "
                                 "seed it (python3 catalog.py add) or name a bare "
                                 "ticker instead")
                    continue
            picked.append(text)
            continue
        ticker = text.upper()
        if not re.fullmatch(r"[A-Z0-9\-\.\^=]{1,12}", ticker):
            notes.append(f"dropped symbol {text!r}: not a valid ticker")
            continue
        try:
            catalog.resolve(ticker)
        except ValueError:
            notes.append(f"symbol {ticker!r} is not in the catalog yet -- fetch it "
                         "before the backtest")
        picked.append(ticker)
    universe: list[str] = []
    for item in picked:
        if item not in universe:
            universe.append(item)
    if not universe:
        notes.append("the artifact named no resolvable ticker -- SPY is used as "
                     "a placeholder universe; replace it with the real one")
        universe = ["SPY"]
    return universe


def _spec_scaffold(row: dict, slug: str, posted: str) -> dict:
    """A spec scaffold for one triaged artifact, from ``_TEMPLATE.yaml``.

    ``posted`` is the artifact's publication date (ISO ``YYYY-MM-DD``); it is
    never defaulted to today -- a caller without a date must not promote.
    """
    template = yaml.safe_load(TEMPLATE.read_text())
    doc = copy.deepcopy(template)
    repo = row.get("repo", "")
    path = row.get("path", "")
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    notes: list[str] = []
    universe = _universe(row.get("symbols"), notes)
    close_only = any(catalog.is_close_only(u) for u in universe)

    doc["meta"] = {
        "slug": slug,
        "title": meta.get("title") or f"{repo}:{path}",
        "url": meta.get("url") or "unknown",
        "source": f"external:{repo}#{path}",
        "posted": posted,
        "claim": meta.get("claim") or (
            "PLACEHOLDER -- the claim of the source artifact has not been "
            "transcribed yet. A human/LLM must read the artifact and rewrite "
            "this line (and signal.definition) before the spec is backtested."),
        "author_evidence": {
            "sample": meta.get("sample") or "unknown",
            "headline_metrics": meta.get("headline_metrics") or {},
            "costs_included": bool(meta.get("costs_included", False)),
        },
        "origin": {
            "repo": repo,
            "artifact": artifact_key(row),
            "path": path,
            "kind": row.get("kind"),
            "triage_verdict": row.get("verdict"),
        },
    }
    doc["data"] = {
        "universe": universe,
        "bar": "1d",
        "fields": ["open", "high", "low", "close", "volume"],
        "start": None,
        "end": None,
        "source": f"external:{repo}",
        "caveats": [
            "The universe comes from the triage model's symbol list, not from a "
            "human read of the artifact -- verify it.",
            "No point-in-time universe (survivorship bias).",
        ],
    }
    doc["signal"] = {
        "definition": ("PLACEHOLDER -- the source artifact's rule has not been "
                       "ported. Replace this text and strategies/"
                       f"{slug}.py together."),
        "lookback_bars": 200,
        "lag_bars": 1,
    }
    doc["rules"] = dict(doc["rules"])
    doc["rules"]["execution_price"] = "close" if close_only else "next_open"
    doc["costs"] = dict(doc["costs"])
    doc["ambiguities"] = list(doc["ambiguities"]) + [
        {"id": f"promotion_{i + 1}", "question": note, "default": "TBD",
         "alternatives": [], "material": True}
        for i, note in enumerate(notes)
    ] or list(doc["ambiguities"])
    doc["validation"] = dict(doc["validation"])
    doc["validation"]["oos_start"] = posted
    doc["verdict"] = {"status": "PENDING", "reason": None,
                      "report": f"reports/{slug}.md"}
    return doc


def _find_artifact(rows: list[dict], target: str) -> dict | list[dict] | None:
    """The triaged artifact named by ``target`` (a key or a path).

    Returns the row, ``None`` when nothing matches, or the list of matches
    when the target is ambiguous.
    """
    exact = [r for r in rows if artifact_key(r) == target]
    if exact:
        return exact[0] if len(exact) == 1 else exact
    base = target.rsplit("|", 1)[-1]
    matches = [r for r in rows
               if r.get("path") == target
               or Path(str(r.get("path", ""))).name == base
               or str(r.get("path", "")).endswith("/" + base)]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else matches


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _restore(blob: bytes | None, path: Path) -> None:
    """Put a file back exactly as it was (or remove it when it did not exist).

    A failed promotion must leave the tree as it was: a ``--force``
    promotion that fails the smoke test restores the previous, good
    spec/strategy instead of deleting them.
    """
    if blob is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(blob)


def promote_main(target: str, repo_filter: str | None, layout: Layout,
                 posted: str | None = None, force: bool = False) -> int:
    """Scaffold ``specs/ext-*.yaml`` + ``strategies/<slug>.py`` for one artifact.

    No backtest is run here: the promoted spec and its placeholder strategy
    are what the existing lane consumes.  The placeholder signal is a
    scaffold, not a port -- the real rule is a human/LLM review away.

    ``posted`` is the artifact's publication date; without a date in the
    triage row (and without ``--posted``) the promotion is REFUSED: a
    fabricated date would claim an out-of-sample window the spec does not
    have.  ``force`` overwrites an existing promotion.
    """
    rows = load_jsonl(layout.triage)
    if repo_filter:
        rows = [r for r in rows if r.get("repo") == repo_filter]
    found = _find_artifact(rows, target)
    if found is None:
        print(f"error: no triaged artifact matches {target!r}")
        return 1
    if isinstance(found, list):
        print(f"error: {target!r} matches several artifacts:")
        for row in found:
            print(f"  - {artifact_key(row)}")
        return 1
    row = found
    repo = row.get("repo", "")
    path = row.get("path", "")
    slug = f"ext-{repo_slug(repo)}-{slugify(Path(path).stem)}"
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    posted_date, date_error = _resolve_posted(meta, posted)
    if date_error:
        print(f"error: {date_error}")
        return 1
    doc = _spec_scaffold(row, slug, posted_date)
    errors = extract.validate_spec(doc, slug)
    if errors:
        print(f"error: the scaffolded spec is invalid for {slug}:")
        for err in errors:
            print(f"  - {err}")
        return 1

    header = ("# SCAFFOLD -- generated by run_external.py --promote.\n"
              "# Placeholder spec: the universe, the signal and the claim still\n"
              "# need a human/LLM review. The verdict stays PENDING until then.\n")
    spec_path = layout.specs / f"{slug}.yaml"
    strat_path = layout.strategies / f"{slug}.py"
    if not force and (spec_path.exists() or strat_path.exists()):
        print(f"error: {slug} is already promoted "
              f"({spec_path} / {strat_path}) -- pass --force to overwrite it")
        return 1
    # Snapshot whatever is there: a failed promotion must restore it.
    old_spec = spec_path.read_bytes() if spec_path.exists() else None
    old_strat = strat_path.read_bytes() if strat_path.exists() else None
    _write_atomic(spec_path, header
                  + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    _write_atomic(strat_path, STRATEGY_SCAFFOLD.format(slug=slug, repo=repo,
                                                       path=path))

    universe = doc["data"]["universe"]
    params = {k: v for k, v in doc["signal"].items()
              if k not in ("definition", "lag_bars")}
    multi = len(universe) > 1
    if multi:
        smoke_data = {t: codegen.smoke_data(t) for t in universe}
        err = codegen.smoke_test(strat_path, params, multi=True, data=smoke_data)
    else:
        err = codegen.smoke_test(strat_path, params, multi=False)
    if err:
        # No half-promotion: a spec without a working strategy is an orphan.
        # A previous promotion (the --force case) is restored unchanged.
        _restore(old_strat, strat_path)
        _restore(old_spec, spec_path)
        print(f"error: the scaffolded strategy failed the smoke test: {err}")
        return 1

    print(f"promoted {repo}:{path}")
    print(f"  spec     {spec_path}")
    print(f"  strategy {strat_path}")
    print(f"  posted   {posted_date} (the out-of-sample window starts here)")
    print("  NOTE: the strategy is a PLACEHOLDER scaffold (flat signal) -- the "
          "real port is a human/LLM review, and no backtest was run.")
    return 0


# ---------------------------------------------------------------- the batch run

def _copy_produced(run_dir: Path, before: set[str], out_dir: Path) -> list[str]:
    """Copy the files the artifact produced into ``out_dir``; return their
    relative paths (relative to the results root)."""
    produced: list[str] = []
    for rel in sorted(_file_set(run_dir) - before):
        src = run_dir / rel
        dest = out_dir / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        except OSError:
            continue
        produced.append(rel)
    return produced


def _evidence_file(path: Path) -> bool:
    """True when a file is evidence worth listing as an output.

    An **empty** log (a timeout before the child printed anything) carries
    no diagnostic: it is removed and not listed, so the ``outputs`` of a
    row never points at a 0-byte file.
    """
    try:
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            path.unlink()
            return False
    except OSError:
        return False
    return True


def _safe_artifact_path(path: str) -> PurePosixPath | None:
    """The artifact path as a safe relative path, or None when it is unsafe.

    The triage file is editable by hand and by an LLM: an absolute path or
    a ``..`` component would let an artifact escape its run copy (and read
    or write outside the isolated tree), so such a path is refused.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    rel = PurePosixPath(path.strip())
    if rel.is_absolute() or ".." in rel.parts:
        return None
    return rel


def _output_stem(path: str) -> str:
    """A file-name stem for the artifact's evidence, unique inside a repo.

    The bare stem collides when two artifacts of one repo share a file name
    ("a/nb.ipynb" and "b/nb.ipynb"), so the parent directories are part of
    the name.
    """
    rel = PurePosixPath(path)
    raw = str(rel.with_suffix("")).replace("/", "-")
    return slugify(raw) or "artifact"


def _unique_stems(rows: list[dict]) -> dict[str, str]:
    """Map every artifact key to a stem that is unique inside its repo.

    ``slugify`` maps "a b.py" and "a-b.py" to the same stem, so two
    artifacts of one repo would overwrite each other's evidence.  Only the
    colliding ones get a short hash of their artifact key appended.
    """
    base = {artifact_key(r): _output_stem(r.get("path", "")) for r in rows}
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = artifact_key(row)
        groups.setdefault((row.get("repo", ""), base[key]), []).append(key)
    stems: dict[str, str] = {}
    for row in rows:
        key = artifact_key(row)
        stem = base[key]
        if len(groups[(row.get("repo", ""), stem)]) > 1:
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
            stem = f"{stem}-{digest}"
        stems[key] = stem
    return stems


def _run_dir_name(key: str) -> str:
    """A collision-free run-directory name for an artifact key.

    ``slugify`` maps "a b.ipynb" and "a-b.ipynb" to the same string, so a
    short hash of the key is appended: two artifacts of one repo must never
    share a run directory.
    """
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(key)}-{digest}"


def run_artifact(row: dict, layout: Layout, env_rows: dict[str, dict],
                 timeout: int = DEFAULT_TIMEOUT, stem: str | None = None) -> dict:
    """Run one triaged artifact and return its leaderboard row.

    The artifact runs in a **copy** of the checkout (``results/external/
    _runs/<key>/``), in its own process, with a hard timeout.  The copy is
    removed afterwards; the evidence (executed notebook, log, produced
    files) is kept under ``results/external/<repo>/``.
    """
    started = time.monotonic()
    repo = row.get("repo", "")
    kind = row.get("kind", "")
    path = row.get("path", "")
    key = artifact_key(row)
    slug = repo_slug(repo)
    choice = select_env(row, env_rows, layout)
    out_row: dict = {"repo": repo, "kind": kind, "path": path,
                     "artifact_key": key, "cells_executed": None,
                     "runtime_sec": 0.0, "env_id": choice["env_id"],
                     "data_mode": choice["data_mode"], "outputs": [],
                     "reason": "", "run_at": now_iso()}

    def finish(verdict: str, reason: str = "") -> dict:
        out_row["verdict"] = verdict
        out_row["reason"] = reason
        out_row["runtime_sec"] = round(time.monotonic() - started, 3)
        return out_row

    if not choice["run"]:
        return finish("skip", choice["reason"])
    if kind not in ("notebook", "script"):
        return finish("skip", f"{kind} artifacts are not executed")

    rel = _safe_artifact_path(path)
    if rel is None:
        return finish("error", f"unsafe artifact path {path!r}: it must be a "
                               "relative path without '..'")

    src = layout.repo_dir(repo)
    if not src.is_dir():
        return finish("error", f"missing checkout: {src}")

    stem = stem or _output_stem(path)
    run_dir = layout.runs / _run_dir_name(key)
    before = _file_set(src)
    outputs: list[str] = []
    try:
        _make_run_copy(src, run_dir)
        env_dir = layout.results / slug
        env_dir.mkdir(parents=True, exist_ok=True)
        if kind == "notebook":
            out_ipynb = env_dir / f"{stem}.ipynb"
            out_log = env_dir / f"{stem}.log"
            res = _run_notebook(rel.as_posix(), out_ipynb, out_log,
                                choice["python"], timeout, choice["env"], run_dir)
            out_row["cells_executed"] = res["cells_executed"]
            candidates = [out_ipynb, out_log]
        else:
            out_log = env_dir / f"{stem}.log"
            res = _run_script(rel.as_posix(), choice["python"], timeout,
                              choice["env"], run_dir)
            # Scripts get the same evidence as notebooks: their captured
            # output is persisted next to the produced files.
            out_log.write_text((res.get("stdout") or "") + (res.get("stderr") or ""))
            candidates = [out_log]
        for candidate in candidates:
            if _evidence_file(candidate):
                outputs.append(f"{slug}/{candidate.name}")
        for rel_produced in _copy_produced(run_dir, before, env_dir / stem):
            outputs.append(f"{slug}/{stem}/{rel_produced}")
        out_row["outputs"] = outputs
    except Exception as exc:              # one bad artifact must not stop the batch
        return finish("error", f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    if res.get("timeout"):
        return finish("timeout", res.get("error") or "timed out")
    if not res.get("ok"):
        return finish("error", res.get("error") or "failed")
    return finish("ok")


def main_with(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Run every triaged artifact that has no leaderboard row yet.

    ``root`` scopes the data/results paths (the tests pass a tmp dir).
    """
    ap = argparse.ArgumentParser(
        description="run triaged external artifacts and record the external "
                    "leaderboard (stage [3c])")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent artifacts")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="per-artifact subprocess cap in seconds")
    ap.add_argument("--repo", default=None, help="restrict to one owner/name")
    ap.add_argument("--promote", default=None, metavar="ARTIFACT",
                    help="scaffold spec+strategy for one artifact (path or key)")
    ap.add_argument("--posted", default=None, metavar="YYYY-MM-DD",
                    help="publication date of the promoted artifact (required "
                         "when the triage row carries no date)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an already promoted spec/strategy")
    args = ap.parse_args(argv)

    layout = Layout(root if root is not None else ROOT)
    if args.promote:
        return promote_main(args.promote, args.repo, layout, posted=args.posted,
                            force=args.force)
    if args.posted or args.force:
        print("error: --posted/--force are only meaningful together with "
              "--promote")
        return 1

    rows = load_jsonl(layout.triage)
    if args.repo:
        rows = [r for r in rows if r.get("repo") == args.repo]
    done = already_run(layout.leaderboard)
    todo = [r for r in rows if artifact_key(r) not in done]
    if not todo:
        print("nothing to run -- every triaged artifact has a completed "
              "(ok) leaderboard row")
        return 0

    env_rows = env_rows_by_repo(layout.env_rows)
    stems = _unique_stems(rows)
    print(f"running {len(todo)} artifacts "
          f"({args.jobs} at a time, {args.timeout}s cap each)\n")
    counts = {"ok": 0, "error": 0, "timeout": 0, "skip": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_artifact, r, layout, env_rows, args.timeout,
                                stems.get(artifact_key(r))): r
                   for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row = fut.result()
            append_row(row, layout.leaderboard)
            verdict = row.get("verdict", "error")
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict in ("error", "timeout"):
                record_failure(row, layout.failed)
            print(f"[{i}/{len(todo)}] {verdict.upper():<8} "
                  f"{row['kind']} {row['repo']}:{row['path']} "
                  f"{row['reason'][:60]}")

    print(f"\n{counts.get('ok', 0)} ok, {counts.get('error', 0)} error, "
          f"{counts.get('timeout', 0)} timeout, {counts.get('skip', 0)} skip "
          f"(see {layout.leaderboard})")
    return 0


if __name__ == "__main__":
    sys.exit(main_with())
