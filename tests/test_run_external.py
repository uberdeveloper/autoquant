# tests/test_run_external.py
"""Tests for run_external.py — external execution harness, leaderboard, promotion.

The subprocess seam (`_launch`) and the environment seam (`shim.shim_env`) are
monkeypatched, and the notebook runner is executed in-process against
synthetic notebooks — except in the isolation tests, where a real interpreter
is used to prove that an artifact runs in the run copy and cannot write back
into the harvested checkout.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import codegen
import extract
import run_external


# ---------------------------------------------------------------- fixtures

def notebook(cells: list[str]) -> dict:
    """A minimal nbformat-4 notebook with the given code cells."""
    return {
        "cells": [{"cell_type": "code", "source": src, "outputs": [],
                   "metadata": {}, "execution_count": None} for src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_repo(root: Path, repo: str = "owner/name", files: dict | None = None) -> Path:
    """Create a synthetic checkout under <root>/data/repos/<owner>-<name>/."""
    dest = root / "data" / "repos" / repo.replace("/", "-")
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):
            path.write_text(json.dumps(content))
        else:
            path.write_text(content)
    return dest


def triage_row(repo: str = "owner/name", kind: str = "notebook",
               path: str = "nb.ipynb", verdict: str = "RUN_WITH_OUR_DATA",
               symbols: list[str] | None = None,
               meta: dict | None = None) -> dict:
    row = {"repo": repo, "kind": kind, "path": path,
           "verdict": verdict, "symbols": symbols if symbols is not None else ["SPY"],
           "reason": "triage reason", "scored_at": "2026-10-21T13:21:23Z"}
    if meta is not None:
        row["meta"] = meta
    return row


def write_triage(root: Path, rows: list[dict]) -> Path:
    path = root / "data" / "repo_triage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def write_envs(root: Path, rows: list[dict]) -> Path:
    path = root / "data" / "repo_envs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ---------------------------------------------------------- subprocess seam

def exec_runner(cmd: list[str], cwd=None) -> int:
    """Run the notebook runner in-process -- no real subprocess.

    No process-global ``os.chdir``: the batch runs its artifacts in a thread
    pool and a global chdir would move the ground under the other threads.
    The notebook path is made absolute instead, which is all the runner
    needs (its outputs are absolute already).
    """
    argv = sys.argv
    code = 0
    try:
        args = list(cmd[cmd.index("-c") + 2:])
        if cwd is not None and args and not os.path.isabs(args[0]):
            args[0] = str(Path(cwd) / args[0])
        sys.argv = ["-c"] + args
        exec(compile(cmd[cmd.index("-c") + 1], "<runner>", "exec"),
             {"__name__": "__runner__"})
    except SystemExit as exc:
        code = exc.code or 0
    finally:
        sys.argv = argv
    return code


def make_launcher(calls: list):
    """A fake `_launch` that runs the notebook runner in-process."""
    def _launch(cmd, cwd=None, env=None, timeout=None):
        calls.append({"cmd": list(cmd), "cwd": cwd, "env": dict(env or {}),
                      "timeout": timeout})
        if "-c" in cmd and cmd[cmd.index("-c") + 1] == run_external._NOTEBOOK_RUNNER:
            code = exec_runner(cmd, cwd=cwd)
            if code:
                return {"ok": False, "error": f"exit {code}: cell failed",
                        "timeout": False, "stdout": "", "stderr": ""}
        return {"ok": True, "error": "", "timeout": False, "stdout": "ok\n",
                "stderr": ""}
    return _launch


class TestLaunch:
    def test_success_returns_stdout(self, monkeypatch):
        class FakePopen:
            pid = 1
            returncode = 0

            def __init__(self, *a, **k):
                pass

            def communicate(self, timeout=None):
                return "all good\n", ""

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        res = run_external._launch(["python", "-c", "x"], timeout=10)
        assert res["ok"] is True
        assert res["stdout"] == "all good\n"

    def test_timeout_kills_the_whole_process_group(self, monkeypatch):
        killed = []

        class FakePopen:
            pid = 4242
            returncode = None

            def __init__(self, *a, **k):
                pass

            def communicate(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return "", ""

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        monkeypatch.setattr(run_external.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(run_external.os, "killpg",
                            lambda pgid, sig: killed.append((pgid, sig)))
        res = run_external._launch(["python", "-c", "x"], timeout=5)
        assert res["ok"] is False
        assert res["timeout"] is True
        assert "timed out" in res["error"]
        assert killed == [(4242, run_external.signal.SIGKILL)]

    def test_nonzero_exit_is_an_error(self, monkeypatch):
        class FakePopen:
            pid = 1
            returncode = 3

            def __init__(self, *a, **k):
                pass

            def communicate(self, timeout=None):
                return "", "boom"

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        res = run_external._launch(["python", "-c", "x"], timeout=10)
        assert res["ok"] is False
        assert "exit 3" in res["error"]

    def test_a_long_stderr_tail_is_truncated_with_a_marker(self, monkeypatch):
        # C1: a silent 300-char cut hides where the tail came from; the
        # launcher must use the same truncation-marker rule as _error_tail.
        class FakePopen:
            pid = 1
            returncode = 1

            def __init__(self, *a, **k):
                pass

            def communicate(self, timeout=None):
                return "", "noise\n" * 500 + "ModuleNotFoundError: nope"

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        res = run_external._launch(["python", "-c", "x"], timeout=10)
        assert res["ok"] is False
        assert res["error"].startswith("exit 1: ...(truncated)")
        assert "ModuleNotFoundError: nope" in res["error"]
        assert len(res["error"]) < 400

    def test_launch_failure_is_reported_not_raised(self, monkeypatch):
        def fake_popen(*a, **k):
            raise FileNotFoundError(2, "No such file or directory", "uv")

        monkeypatch.setattr(run_external.subprocess, "Popen", fake_popen)
        res = run_external._launch(["uv", "run", "python"], timeout=10)
        assert res["ok"] is False
        assert "could not launch" in res["error"]

    def test_a_straggler_child_is_killed_after_a_bounded_second_wait(self,
                                                                    monkeypatch):
        """C3/M19: the reap after the group kill must be bounded.

        The child ignores the group kill: every ``communicate`` times out.
        The launcher must then kill the child and wait a *bounded* time --
        an unbounded wait would hang the whole batch.
        """
        calls: list = []

        class FakePopen:
            pid = 4242
            returncode = None

            def __init__(self, *a, **k):
                pass

            def communicate(self, timeout=None):
                calls.append(("communicate", timeout))
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

            def kill(self):
                calls.append(("kill", None))

            def wait(self, timeout=None):
                calls.append(("wait", timeout))
                return -9

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        monkeypatch.setattr(run_external.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(run_external.os, "killpg", lambda pgid, sig: None)

        res = run_external._launch(["python", "-c", "x"], timeout=5)

        assert res["ok"] is False
        assert res["timeout"] is True
        # every wait is bounded -- no unbounded wait anywhere
        waits = [t for name, t in calls if name == "wait"]
        assert waits, calls
        assert all(t is not None for t in waits), calls
        assert max(waits) <= 5, calls
        # the straggler is killed, and the kill happens before the wait
        assert ("kill", None) in calls, calls
        assert calls.index(("kill", None)) < calls.index(("wait", waits[0]))
        # the second communicate is bounded too
        timeouts = [t for name, t in calls if name == "communicate"]
        assert timeouts == [5, 10], calls


class TestNotebook:
    def _work(self, tmp_path, cells):
        work = tmp_path / "work"
        work.mkdir()
        (work / "nb.ipynb").write_text(json.dumps(notebook(cells)))
        return work

    def test_cells_run_in_one_shared_namespace(self, tmp_path, monkeypatch):
        work = self._work(tmp_path, ["x = 41\n", "print(x + 1)\n"])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is True
        assert res["cells_executed"] == 2
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for o in doc["cells"][1]["outputs"]]
        assert any("42" in t for t in texts)
        assert "42" in out_log.read_text()

    def test_failing_cell_keeps_the_partial_notebook_as_evidence(self, tmp_path,
                                                                 monkeypatch):
        work = self._work(tmp_path, ["x = 1\n", "raise ValueError('nope')\n",
                                     "print('never')\n"])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is False
        assert res["error"]
        assert res["cells_executed"] == 2      # the failing cell counts
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for o in doc["cells"][1]["outputs"]]
        assert any("ValueError" in t for t in texts)
        assert "nope" in out_log.read_text()

    def test_markdown_cells_are_not_counted(self, tmp_path, monkeypatch):
        nb = notebook(["print('a')\n"])
        nb["cells"].insert(0, {"cell_type": "markdown", "source": "# title"})
        work = tmp_path / "work"
        work.mkdir()
        (work / "nb.ipynb").write_text(json.dumps(nb))
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["cells_executed"] == 1

    def test_a_cell_source_given_as_a_list_of_lines_runs(self, tmp_path,
                                                         monkeypatch):
        nb = notebook(["print('x')\n"])
        nb["cells"][0]["source"] = ["print('x')\n"]     # nbformat allows a list
        repo = tmp_path / "data" / "repos" / "owner-name"
        repo.mkdir(parents=True)
        (repo / "nb.ipynb").write_text(json.dumps(nb))
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        out = tmp_path / "out.ipynb"
        res = run_external._run_notebook("nb.ipynb", out, tmp_path / "out.log",
                                         ["python"], 10, {}, repo)
        assert res["ok"] is True
        assert res["cells_executed"] == 1

    def test_ipython_magics_and_shell_lines_are_skipped_not_fatal(
            self, tmp_path, monkeypatch):
        # Jupyter-only syntax (`%magic`, `!shell`) is valid in a notebook but
        # is not Python: the runner must skip those lines, not die with a
        # SyntaxError that hides the artifact's real failure.
        work = self._work(tmp_path, [
            "%matplotlib inline\nprint('after magic')\n",
            "!pip install pandas==0.24.1 --user\nprint('after shell')\n",
            "%%capture\nprint('after cell magic')\n",
        ])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is True
        assert res["cells_executed"] == 3
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for c in doc["cells"] for o in c["outputs"]]
        assert any("after magic" in t for t in texts)
        assert any("after shell" in t for t in texts)
        assert any("after cell magic" in t for t in texts)

    def test_valid_python_with_magic_looking_lines_runs_unmodified(
            self, tmp_path, monkeypatch):
        # C2: a line that merely *looks* like a magic (starts with %) but is
        # valid Python (a modulo continuation inside parentheses) must run
        # unmodified -- the old unconditional strip broke it into a
        # SyntaxError.  compile-first: strip only if the source fails to
        # compile.
        work = self._work(tmp_path, ["x = (100\n  % 7)\nprint(x)\n"])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is True, out_log.read_text()
        assert res["cells_executed"] == 1
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for c in doc["cells"] for o in c["outputs"]]
        assert any("2" in t for t in texts)      # 100 % 7 == 2

    def test_a_magic_looking_line_inside_a_string_literal_is_preserved(
            self, tmp_path, monkeypatch):
        # C2: stripping must never delete text inside a triple-quoted string
        # (the old strip silently rewrote the source that ran).  A cell whose
        # source compiles is executed byte-for-byte unchanged.
        cell = ('note = """\n%pip install pandas\n!rm -rf /\n"""\n'
                "print('kept' if '!rm -rf' in note else 'lost')\n")
        work = self._work(tmp_path, [cell])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is True, out_log.read_text()
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for c in doc["cells"] for o in c["outputs"]]
        assert any("kept" in t for t in texts)

    def test_a_non_python_cell_magic_skips_the_whole_cell(
            self, tmp_path, monkeypatch):
        # C2: %%bash's body is a shell script, not Python; executing it as
        # Python is wrong (it used to fail with a SyntaxError that hid the
        # notebook's real outcome).  The whole cell is skipped and the run
        # continues with the next cell.
        work = self._work(tmp_path, ["%%bash\necho hi > made.txt\n",
                                     "print('b')\n"])
        out_ipynb, out_log = tmp_path / "out.ipynb", tmp_path / "out.log"
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        res = run_external._run_notebook("nb.ipynb", out_ipynb, out_log,
                                         ["python"], 30, {}, work)
        assert res["ok"] is True, out_log.read_text()
        assert res["cells_executed"] == 1       # %%bash cell skipped, not run
        doc = json.loads(out_ipynb.read_text())
        texts = [o.get("text", "") for c in doc["cells"] for o in c["outputs"]]
        assert any("b" in t for t in texts)
        assert not (work / "made.txt").exists()

    def test_one_subprocess_per_notebook(self, tmp_path, monkeypatch):
        work = self._work(tmp_path, ["print('a')\n", "print('b')\n"])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external._run_notebook("nb.ipynb", tmp_path / "o.ipynb",
                                   tmp_path / "o.log", ["python"], 30, {}, work)
        assert len(calls) == 1
        assert calls[0]["cmd"][1] == "-c"


class TestRunCopy:
    def test_run_copy_never_writes_into_the_source_checkout(self, tmp_path):
        src = make_repo(tmp_path, files={"nb.ipynb": json.dumps(notebook([])),
                                         "data.txt": "original"})
        dest = tmp_path / "run"
        run_external._make_run_copy(src, dest)
        (dest / "data.txt").write_text("changed")
        assert (src / "data.txt").read_text() == "original"
        assert (dest / "nb.ipynb").exists()

    def test_file_set_is_relative(self, tmp_path):
        src = make_repo(tmp_path, files={"a.txt": "x", "sub/b.txt": "y"})
        assert run_external._file_set(src) == {"a.txt", "sub/b.txt"}


# --------------------------------------------------------- stage 3: the batch

class TestMain:
    def _repo(self, tmp_path, cells=("print('hi')\n", "print('there')\n"),
              repo="owner/name", name="nb.ipynb"):
        make_repo(tmp_path, repo=repo, files={name: json.dumps(notebook(list(cells)))})
        write_triage(tmp_path, [triage_row(repo=repo, path=name)])
        return tmp_path

    def test_run_writes_a_leaderboard_row_and_the_executed_notebook(self, tmp_path,
                                                                    monkeypatch):
        self._repo(tmp_path)
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        assert run_external.main_with(["--repo", "owner/name"], root=tmp_path) == 0
        rows = [json.loads(line) for line in
                (tmp_path / "results" / "external" / "leaderboard.jsonl")
                .read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["verdict"] == "ok"
        assert row["artifact_key"] == "owner/name|notebook|nb.ipynb"
        assert row["kind"] == "notebook"
        assert row["repo"] == "owner/name"
        assert row["cells_executed"] == 2
        assert row["data_mode"] == "ours"
        assert row["env_id"] == "main"
        assert isinstance(row["runtime_sec"], float)
        assert row["run_at"].endswith("Z")
        assert set(row) == {"repo", "kind", "path", "artifact_key", "verdict",
                            "cells_executed", "runtime_sec", "env_id",
                            "data_mode", "outputs", "reason", "run_at"}
        ipynb = tmp_path / "results" / "external" / "owner-name" / "nb.ipynb"
        assert ipynb.exists()
        assert "owner-name/nb.ipynb" in row["outputs"]

    def test_rerun_skips_artifacts_already_on_the_leaderboard(self, tmp_path,
                                                               monkeypatch):
        self._repo(tmp_path)
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert len(calls) == 1
        calls.clear()
        run_external.main_with([], root=tmp_path)
        assert calls == []          # the second run re-executed nothing
        rows = (tmp_path / "results" / "external" / "leaderboard.jsonl").read_text()
        assert len(rows.splitlines()) == 1

    def test_the_source_checkout_is_never_written(self, tmp_path, monkeypatch):
        self._repo(tmp_path)
        src = tmp_path / "data" / "repos" / "owner-name"
        before = run_external._file_set(src)
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)
        assert run_external._file_set(src) == before

    def test_a_blocked_env_row_downgrades_a_run_verdict_without_launching(
            self, tmp_path, monkeypatch):
        # C3: the env_repo ROW CONTRACT -- a ready row with a non-empty
        # ``blocked`` list is the SKIP signal on EVERY verdict, including
        # RUN_WITH_OUR_DATA.  The artifact must be skipped with a reason
        # naming the blocked entries and zero launcher calls.
        self._repo(tmp_path)
        write_envs(tmp_path, [{"repo": "owner/name", "status": "ready",
                               "mode": "main", "venv": None,
                               "env_key": "abc123", "deps_sha256": "",
                               "error": "", "blocked": [
                                   {"kind": "notebook", "path": "nb.ipynb",
                                    "reason": "ta-lib is required"}]}])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        assert run_external.main_with([], root=tmp_path) == 0
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "skip"
        assert "blocked" in row["reason"]
        assert "ta-lib is required" in row["reason"]
        assert calls == []                       # never launched

    def test_spec_artifact_is_recorded_but_not_executed(self, tmp_path, monkeypatch):
        make_repo(tmp_path, files={"strategy.yaml": "meta: {}\n"})
        write_triage(tmp_path, [triage_row(kind="spec", path="strategy.yaml",
                                           verdict="SPEC_ARTIFACT")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "skip"
        assert "promot" in row["reason"].lower()

    def test_heavy_env_without_a_row_is_skipped(self, tmp_path, monkeypatch):
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="NEEDS_HEAVY_ENV")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "skip"
        assert row["env_id"] == "none"
        assert row["data_mode"] == "none"

    def test_failing_artifact_is_recorded_in_failed_jsonl(self, tmp_path,
                                                          monkeypatch):
        self._repo(tmp_path, cells=("raise ValueError('boom')\n",))
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "error"
        assert row["reason"]
        failed = (tmp_path / "results" / "external" / "failed.jsonl").read_text()
        assert "boom" in failed or "exit 1" in failed

    def test_a_failing_notebook_row_carries_the_traceback_tail(
            self, tmp_path, monkeypatch):
        # C1: a real subprocess (the runner writes the traceback to stderr
        # on failure) must surface that traceback in the leaderboard row's
        # reason -- "exit 1: " with an empty tail is not diagnosable.
        self._repo(tmp_path, cells=("raise ValueError('boom')\n",))

        real_select = run_external.select_env

        def select(row, env_rows, layout):
            choice = real_select(row, env_rows, layout)
            choice["python"] = [sys.executable]      # a real interpreter
            return choice

        monkeypatch.setattr(run_external, "select_env", select)
        assert run_external.main_with([], root=tmp_path) == 0
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "error"
        assert "ValueError" in row["reason"]
        assert "boom" in row["reason"]

    def test_timeout_is_recorded_as_timeout(self, tmp_path, monkeypatch):
        self._repo(tmp_path)

        def timeout_launch(cmd, cwd=None, env=None, timeout=None):
            return {"ok": False, "error": "timed out after 600s", "timeout": True,
                    "stdout": "", "stderr": ""}

        monkeypatch.setattr(run_external, "_launch", timeout_launch)
        run_external.main_with([], root=tmp_path)
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "timeout"
        assert "timed out" in row["reason"]

    def test_repo_filter_limits_the_run(self, tmp_path, monkeypatch):
        make_repo(tmp_path, repo="owner/one", files={"a.py": "print(1)\n"})
        make_repo(tmp_path, repo="owner/two", files={"b.py": "print(2)\n"})
        write_triage(tmp_path, [triage_row(repo="owner/one", kind="script",
                                           path="a.py"),
                                triage_row(repo="owner/two", kind="script",
                                           path="b.py")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with(["--repo", "owner/one"], root=tmp_path)
        assert len(calls) == 1
        assert calls[0]["cmd"][-1].endswith("a.py")

    def test_missing_checkout_is_an_error_row_not_a_crash(self, tmp_path,
                                                          monkeypatch):
        write_triage(tmp_path, [triage_row()])       # no checkout on disk
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        assert run_external.main_with([], root=tmp_path) == 0
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "error"
        assert "checkout" in row["reason"]

    def test_a_crashing_artifact_does_not_kill_the_batch(self, tmp_path,
                                                         monkeypatch):
        make_repo(tmp_path, files={"a.py": "print(1)\n", "b.py": "print(2)\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="a.py"),
                                triage_row(kind="script", path="b.py")])

        def boom(cmd, cwd=None, env=None, timeout=None):
            raise RuntimeError("launcher exploded")

        monkeypatch.setattr(run_external, "_launch", boom)
        assert run_external.main_with([], root=tmp_path) == 0
        rows = [json.loads(line) for line in
                (tmp_path / "results" / "external" / "leaderboard.jsonl")
                .read_text().splitlines()]
        assert len(rows) == 2
        assert {row["verdict"] for row in rows} == {"error"}
        assert all("launcher exploded" in row["reason"] for row in rows)

    def test_nothing_to_run_returns_zero(self, tmp_path, monkeypatch, capsys):
        write_triage(tmp_path, [])
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        assert run_external.main_with([], root=tmp_path) == 0
        assert "nothing to run" in capsys.readouterr().out

    def test_bundled_data_runs_without_the_shim(self, tmp_path, monkeypatch):
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="RUN_BUNDLED_DATA")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert "AUTOQUANT_CATALOG_DIR" not in calls[0]["env"]


# --------------------------------------------------------- stage 4: promotion

class TestPromote:
    SLUG = "ext-owner-name-strategy"
    POSTED = "2024-01-02"

    def _row(self, tmp_path, symbols=None, path="strategy.md", meta=None):
        make_repo(tmp_path, files={path: "meta: {}\n"})
        write_triage(tmp_path, [triage_row(kind="spec", path=path,
                                           verdict="SPEC_ARTIFACT",
                                           symbols=symbols, meta=meta)])
        return f"owner/name|spec|{path}"

    def _promote(self, tmp_path, key, *extra):
        return run_external.main_with(["--promote", key, *extra], root=tmp_path)

    def _spec(self, tmp_path):
        return yaml.safe_load(
            (tmp_path / "specs" / f"{self.SLUG}.yaml").read_text())

    def test_promote_writes_a_spec_that_passes_validate_spec(self, tmp_path):
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 0
        spec_path = tmp_path / "specs" / f"{self.SLUG}.yaml"
        assert spec_path.exists()
        doc = self._spec(tmp_path)
        assert doc["meta"]["slug"] == self.SLUG
        assert extract.validate_spec(doc, self.SLUG) == []
        assert doc["data"]["universe"] == ["SPY"]
        assert doc["signal"]["lag_bars"] >= 1
        assert doc["costs"]["commission_bps"] + doc["costs"]["slippage_bps"] > 0
        assert doc["ambiguities"]
        assert doc["validation"]["oos_start"] == doc["meta"]["posted"]
        assert doc["verdict"]["status"] == "PENDING"
        assert "owner/name" in doc["meta"]["source"]
        assert "strategy.md" in doc["meta"]["source"]

    def test_promote_writes_a_strategy_that_passes_the_smoke_test(self, tmp_path):
        key = self._row(tmp_path)
        self._promote(tmp_path, key, "--posted", self.POSTED)
        strat = tmp_path / "strategies" / f"{self.SLUG}.py"
        assert strat.exists()
        doc = self._spec(tmp_path)
        params = {k: v for k, v in doc["signal"].items()
                  if k not in ("definition", "lag_bars")}
        assert codegen.smoke_test(strat, params) is None
        spec_ = importlib.util.spec_from_file_location(f"strategies.{self.SLUG}",
                                                       strat)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        out = mod.signal(codegen.smoke_frame())
        assert isinstance(out, pd.Series)
        assert out.index.equals(codegen.smoke_frame().index)

    def test_a_multi_ticker_universe_passes_the_multi_smoke_test(self, tmp_path):
        key = self._row(tmp_path, symbols=["SPY", "QQQ"])
        self._promote(tmp_path, key, "--posted", self.POSTED)
        strat = tmp_path / "strategies" / f"{self.SLUG}.py"
        params = {t: codegen.smoke_data(t) for t in ("SPY", "QQQ")}
        assert codegen.smoke_test(strat, params, multi=True) is None

    def test_promote_accepts_a_bare_path(self, tmp_path):
        self._row(tmp_path)
        assert self._promote(tmp_path, "strategy.md", "--posted", self.POSTED) == 0
        assert (tmp_path / "specs" / f"{self.SLUG}.yaml").exists()

    def test_promote_of_an_unknown_artifact_fails_without_writing(self, tmp_path):
        write_triage(tmp_path, [])
        assert self._promote(tmp_path, "nope", "--posted", self.POSTED) == 1
        assert not (tmp_path / "specs").exists()

    def test_promote_does_not_run_a_backtest(self, tmp_path):
        key = self._row(tmp_path)
        self._promote(tmp_path, key, "--posted", self.POSTED)
        assert not (tmp_path / "results").exists()
        assert not (tmp_path / "reports").exists()

    def test_an_unresolvable_fred_symbol_is_dropped_and_noted(self, tmp_path,
                                                              monkeypatch):
        key = self._row(tmp_path, symbols=["fred:NOPE", "SPY"])

        def fake_resolve(series_id):
            if str(series_id).startswith("fred:"):
                raise ValueError(f"{series_id!r} not in catalog")
            return {"id": series_id}

        monkeypatch.setattr(run_external.catalog, "resolve", fake_resolve)
        self._promote(tmp_path, key, "--posted", self.POSTED)
        doc = self._spec(tmp_path)
        assert doc["data"]["universe"] == ["SPY"]
        assert any("fred:NOPE" in json.dumps(a) for a in doc["ambiguities"])

    # ------------------------------------------------- the publication date

    def test_promotion_without_a_publication_date_is_refused(self, tmp_path,
                                                             capsys):
        """F1: no date in the triage row -> no promotion, no fabricated date."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key) == 1
        out = capsys.readouterr().out
        assert "publication date" in out
        assert "--posted" in out
        assert not (tmp_path / "specs").exists()
        assert not (tmp_path / "strategies").exists()

    def test_promotion_uses_the_explicit_posted_date(self, tmp_path):
        """F1: --posted is the publication date, and the OOS window starts there."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "2024-01-02") == 0
        doc = self._spec(tmp_path)
        assert doc["meta"]["posted"] == "2024-01-02"
        assert doc["validation"]["oos_start"] == "2024-01-02"

    def test_promotion_never_defaults_to_today(self, tmp_path):
        """F1: the promotion date is never 'today'."""
        key = self._row(tmp_path)
        self._promote(tmp_path, key, "--posted", "2024-01-02")
        doc = self._spec(tmp_path)
        today = run_external.date.today().isoformat()
        assert doc["meta"]["posted"] != today

    def test_a_date_in_the_triage_row_is_used(self, tmp_path):
        key = self._row(tmp_path, meta={"posted": "2023-05-06"})
        assert self._promote(tmp_path, key) == 0
        doc = self._spec(tmp_path)
        assert doc["meta"]["posted"] == "2023-05-06"
        assert doc["validation"]["oos_start"] == "2023-05-06"

    def test_a_symbol_less_artifact_falls_back_to_spy_and_records_it(self,
                                                                     tmp_path):
        """N3: the SPY fallback is documented -- and recorded as an ambiguity."""
        key = self._row(tmp_path, symbols=[])
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 0
        doc = self._spec(tmp_path)
        assert doc["data"]["universe"] == ["SPY"]
        assert any("SPY" in json.dumps(a) for a in doc["ambiguities"])

    def test_the_promoted_date_is_parseable_by_the_backtest_lane(self, tmp_path):
        """F4: backtest.py does `pd.Timestamp(meta.posted)` -- it must parse."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "2024-01-02") == 0
        doc = self._spec(tmp_path)
        stamp = pd.Timestamp(doc["meta"]["posted"])
        assert stamp is not None
        assert stamp == pd.Timestamp(doc["validation"]["oos_start"])

    def test_a_future_publication_date_is_refused(self, tmp_path, capsys):
        """A future oos_start would be an empty (vacuous) out-of-sample window."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "2099-01-01") == 1
        assert "future" in capsys.readouterr().out
        assert not (tmp_path / "specs").exists()

    def test_a_non_date_publication_date_is_refused(self, tmp_path, capsys):
        """F4: 'unknown' would crash backtest.py -- the promotion refuses it."""
        key = self._row(tmp_path, meta={"posted": "unknown"})
        assert self._promote(tmp_path, key) == 1
        out = capsys.readouterr().out
        assert "publication date" in out
        assert not (tmp_path / "specs").exists()

    def test_an_invalid_explicit_posted_value_is_refused(self, tmp_path, capsys):
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "not-a-date") == 1
        out = capsys.readouterr().out
        assert "--posted" in out
        assert not (tmp_path / "specs").exists()

    def test_a_truncated_date_is_refused(self, tmp_path):
        key = self._row(tmp_path, meta={"posted": "2024-05"})
        assert self._promote(tmp_path, key) == 1
        assert not (tmp_path / "specs").exists()

    def test_a_truncated_explicit_posted_value_is_refused(self, tmp_path,
                                                          capsys):
        """D4: '2024-01-029' must not be truncated into '2024-01-02'."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "2024-01-029") == 1
        out = capsys.readouterr().out
        assert "--posted" in out
        assert not (tmp_path / "specs").exists()
        assert not (tmp_path / "strategies").exists()

    def test_a_padded_explicit_posted_value_is_refused(self, tmp_path):
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", "2024-01-02junk") == 1
        assert not (tmp_path / "specs").exists()

    # ------------------------------------------------------ re-promotion

    def test_repromoting_refuses_to_overwrite(self, tmp_path, capsys):
        """N6: an existing promotion is not silently clobbered."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 0
        spec = tmp_path / "specs" / f"{self.SLUG}.yaml"
        spec.write_text("# hand edited\n")
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 1
        assert "--force" in capsys.readouterr().out
        assert spec.read_text() == "# hand edited\n"

    def test_force_overwrites_an_existing_promotion(self, tmp_path):
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 0
        spec = tmp_path / "specs" / f"{self.SLUG}.yaml"
        spec.write_text("# hand edited\n")
        assert self._promote(tmp_path, key, "--posted", self.POSTED, "--force") == 0
        assert "# hand edited" not in spec.read_text()

    def test_a_failed_smoke_test_leaves_no_orphan_spec(self, tmp_path,
                                                       monkeypatch):
        key = self._row(tmp_path)
        monkeypatch.setattr(run_external.codegen, "smoke_test",
                            lambda *a, **k: "boom")
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 1
        assert not (tmp_path / "specs" / f"{self.SLUG}.yaml").exists()
        assert not (tmp_path / "strategies" / f"{self.SLUG}.py").exists()

    def test_a_failed_forced_promotion_restores_the_previous_one(self, tmp_path,
                                                                  monkeypatch):
        """D2: --force + a failing smoke test must not destroy a good promotion."""
        key = self._row(tmp_path)
        assert self._promote(tmp_path, key, "--posted", self.POSTED) == 0
        spec = tmp_path / "specs" / f"{self.SLUG}.yaml"
        strat = tmp_path / "strategies" / f"{self.SLUG}.py"
        spec.write_text("# hand edited spec\n")
        strat.write_text("# hand edited strategy\n")

        monkeypatch.setattr(run_external.codegen, "smoke_test",
                            lambda *a, **k: "boom")
        assert self._promote(tmp_path, key, "--posted", self.POSTED,
                             "--force") == 1
        assert spec.read_text() == "# hand edited spec\n"
        assert strat.read_text() == "# hand edited strategy\n"

    def test_posted_without_promote_is_an_error(self, tmp_path, capsys):
        assert run_external.main_with(["--posted", "2024-01-02"],
                                       root=tmp_path) == 1
        assert "--promote" in capsys.readouterr().out


# --------------------------------------------------------- stage 1: helpers

class TestJsonlHelpers:
    def test_load_jsonl_tolerates_torn_lines(self, tmp_path):
        path = tmp_path / "x.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n{"torn\n')
        rows = run_external.load_jsonl(path)
        assert rows == [{"a": 1}, {"b": 2}]

    def test_load_jsonl_missing_file_is_empty(self, tmp_path):
        assert run_external.load_jsonl(tmp_path / "nope.jsonl") == []

    def test_already_run_collects_artifact_keys(self, tmp_path):
        lb = tmp_path / "leaderboard.jsonl"
        lb.write_text('{"artifact_key": "a|b|c", "verdict": "ok"}\n\n'
                      '{"artifact_key": "d|e|f", "verdict": "ok"}\n')
        assert run_external.already_run(lb) == {"a|b|c", "d|e|f"}

    def test_already_run_only_counts_completed_rows(self, tmp_path):
        """F2: skip/error/timeout rows are evidence, not finished work."""
        lb = tmp_path / "leaderboard.jsonl"
        lb.write_text('{"artifact_key": "a|b|c", "verdict": "skip"}\n'
                      '{"artifact_key": "d|e|f", "verdict": "error"}\n'
                      '{"artifact_key": "g|h|i", "verdict": "timeout"}\n')
        assert run_external.already_run(lb) == set()

    def test_already_run_missing_file_is_empty(self, tmp_path):
        assert run_external.already_run(tmp_path / "nope.jsonl") == set()

    def test_record_failure_appends_jsonl(self, tmp_path):
        path = tmp_path / "failed.jsonl"
        run_external.record_failure({"artifact_key": "a|b|c", "error": "boom"}, path)
        run_external.record_failure({"artifact_key": "d|e|f", "error": "bang"}, path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["artifact_key"] for r in rows] == ["a|b|c", "d|e|f"]

    def test_latest_by_key_keeps_the_last_row_per_key(self):
        rows = [{"artifact_key": "a", "verdict": "skip"},
                {"artifact_key": "a", "verdict": "ok"},
                {"artifact_key": "b", "verdict": "skip"}]
        logical = run_external.latest_by_key(rows)
        assert set(logical) == {"a", "b"}
        assert logical["a"]["verdict"] == "ok"
        assert logical["b"]["verdict"] == "skip"

    def test_iso_date_refuses_a_truncated_or_padded_value(self):
        """D4: a typo'd date is refused, never silently truncated."""
        assert run_external._iso_date("2024-01-02") == "2024-01-02"
        assert run_external._iso_date("2024-01-029") is None
        assert run_external._iso_date("2024-01-02junk") is None
        assert run_external._iso_date("20240102") is None
        assert run_external._iso_date("2024-02-30") is None
        assert run_external._iso_date("2024-05") is None
        assert run_external._iso_date("unknown") is None

    def test_artifact_key_is_repo_kind_path(self):
        row = triage_row(repo="owner/name", kind="notebook", path="a b.ipynb")
        assert run_external.artifact_key(row) == "owner/name|notebook|a b.ipynb"

    def test_repo_slug_is_a_safe_directory_name(self):
        assert run_external.repo_slug("owner/name") == "owner-name"


class TestSelectEnv:
    def _layout(self, tmp_path):
        return run_external.Layout(tmp_path)

    def test_our_data_gets_the_shim_env(self, tmp_path):
        layout = self._layout(tmp_path)
        res = run_external.select_env(triage_row(verdict="RUN_WITH_OUR_DATA"),
                                      {}, layout)
        assert res["run"] is True
        assert res["data_mode"] == "ours"
        assert res["env"]["AUTOQUANT_CATALOG_DIR"] == str(layout.catalog)
        assert "AUTOQUANT_SHIM_OFFLINE" in res["env"]

    def test_bundled_data_runs_without_the_shim(self, tmp_path):
        layout = self._layout(tmp_path)
        res = run_external.select_env(triage_row(verdict="RUN_BUNDLED_DATA"),
                                      {}, layout)
        assert res["run"] is True
        assert res["data_mode"] == "bundled"
        assert "AUTOQUANT_CATALOG_DIR" not in res["env"]

    def test_heavy_env_without_a_row_downgrades_to_skip(self, tmp_path):
        layout = self._layout(tmp_path)
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {}, layout)
        assert res["run"] is False
        assert res["env_id"] == "none"
        assert "env" in res["reason"].lower()

    def test_heavy_env_failed_row_is_skipped_and_never_reads_blocked(self, tmp_path):
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "failed", "error": "ta-lib missing",
               "blocked": [], "mode": "repo-venv", "venv": None, "env_key": ""}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is False
        assert "ta-lib missing" in res["reason"]

    def test_run_verdict_ready_row_with_blocked_downgrades_to_skip(self, tmp_path):
        # C3: env_repo ROW CONTRACT -- non-empty blocked on a READY row is
        # the SKIP signal on the RUN_* paths too, not only NEEDS_HEAVY_ENV.
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "ready",
               "blocked": [{"kind": "notebook", "path": "nb.ipynb",
                            "reason": "ta-lib is required"}],
               "mode": "main", "venv": None, "env_key": "abc123"}
        res = run_external.select_env(triage_row(verdict="RUN_WITH_OUR_DATA"),
                                      {"owner/name": row}, layout)
        assert res["run"] is False
        assert "blocked" in res["reason"]
        assert "ta-lib is required" in res["reason"]

    def test_a_long_blocked_list_is_bounded_with_a_truncation_marker(
            self, tmp_path):
        # N1: a ready row's ``blocked`` list can hold an entire repo's
        # block dump (93 dict reprs in production); joining it unbounded
        # copied 16 kB into ONE skip row -- 71% of the leaderboard. The
        # reason must stay small while still naming the blocked entries,
        # on both skip paths (RUN_* and NEEDS_HEAVY_ENV).
        layout = self._layout(tmp_path)
        blocked = [{"kind": "notebook", "path": f"nb_{i:03d}.ipynb",
                    "reason": "ta-lib needs a C library (TA-Lib) that is "
                              "not installed"} for i in range(93)]
        row = {"repo": "owner/name", "status": "ready", "blocked": blocked,
               "mode": "main", "venv": None, "env_key": "abc123"}
        for verdict in ("RUN_WITH_OUR_DATA", "NEEDS_HEAVY_ENV"):
            res = run_external.select_env(triage_row(verdict=verdict),
                                          {"owner/name": row}, layout)
            assert res["run"] is False
            assert "env for owner/name is blocked:" in res["reason"]
            assert "nb_092.ipynb" in res["reason"]   # still names entries
            assert "...(truncated)" in res["reason"]
            assert len(res["reason"]) < 400

    def test_a_short_blocked_list_is_not_truncated(self, tmp_path):
        # The bound must only bite when the list is long: a short blocked
        # list is quoted in full (no marker, nothing dropped).
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "ready",
               "blocked": [{"kind": "notebook", "path": "nb.ipynb",
                            "reason": "ta-lib is required"}],
               "mode": "main", "venv": None, "env_key": "abc123"}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is False
        assert "...(truncated)" not in res["reason"]
        assert "nb.ipynb" in res["reason"]
        assert "ta-lib is required" in res["reason"]

    def test_heavy_env_failed_row_keeps_the_root_cause_of_a_long_log(
            self, tmp_path):
        # A uv build failure is thousands of lines of download noise with the
        # actual cause at the very end: the reason must keep that cause and
        # must not copy the whole log into every leaderboard row.
        layout = self._layout(tmp_path)
        error = "Downloading scipy (40.2MiB)\n" * 200 + (
            "error: [Errno 2] No such file or directory: 'cmake'\n")
        row = {"repo": "owner/name", "status": "failed", "error": error,
               "blocked": [], "mode": "repo-venv", "venv": None, "env_key": ""}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is False
        assert "cmake" in res["reason"]
        assert len(res["reason"]) < 1000

    def test_heavy_env_blocked_row_is_skipped_with_the_blocked_reason(self, tmp_path):
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "ready",
               "blocked": ["ta-lib is required"], "mode": "repo-venv",
               "venv": ".venv", "env_key": "abc123"}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is False
        assert "ta-lib is required" in res["reason"]

    def test_heavy_env_ready_row_uses_the_repo_venv_python(self, tmp_path):
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "ready", "blocked": [],
               "mode": "repo-venv", "venv": "data/repos/owner-name/.venv",
               "env_key": "abc123"}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is True
        assert res["env_id"] == "abc123"
        assert res["python"] == [str(tmp_path / "data/repos/owner-name/.venv"
                                      / "bin" / "python")]

    def test_heavy_env_main_mode_row_uses_uv_run_python(self, tmp_path):
        layout = self._layout(tmp_path)
        row = {"repo": "owner/name", "status": "ready", "blocked": [],
               "mode": "main", "venv": None, "env_key": "abc123"}
        res = run_external.select_env(triage_row(verdict="NEEDS_HEAVY_ENV"),
                                      {"owner/name": row}, layout)
        assert res["run"] is True
        assert res["python"] == ["uv", "run", "python"]

    def test_spec_artifact_is_not_executed(self, tmp_path):
        layout = self._layout(tmp_path)
        res = run_external.select_env(triage_row(kind="spec", verdict="SPEC_ARTIFACT"),
                                      {}, layout)
        assert res["run"] is False
        assert "promot" in res["reason"].lower()

    def test_skip_verdicts_stay_skipped(self, tmp_path):
        layout = self._layout(tmp_path)
        for verdict in ("SKIP", "INTERACTIVE_ONLY"):
            res = run_external.select_env(triage_row(verdict=verdict), {}, layout)
            assert res["run"] is False
            assert verdict in res["reason"]


# ------------------------------------------- stage 3b: resume + isolation

class TestResume:
    """F2: only a completed ("ok") leaderboard row stops an artifact."""

    def _repo(self, tmp_path):
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py")])
        return tmp_path

    def _seed(self, tmp_path, verdict):
        lb = tmp_path / "results" / "external" / "leaderboard.jsonl"
        lb.parent.mkdir(parents=True, exist_ok=True)
        lb.write_text(json.dumps({"artifact_key": "owner/name|script|s.py",
                                  "verdict": verdict}) + "\n")

    def test_a_skipped_artifact_is_retried_on_the_next_run(self, tmp_path,
                                                           monkeypatch):
        self._repo(tmp_path)
        self._seed(tmp_path, "skip")
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert len(calls) == 1

    def test_a_timed_out_artifact_is_retried_on_the_next_run(self, tmp_path,
                                                             monkeypatch):
        self._repo(tmp_path)
        self._seed(tmp_path, "timeout")
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert len(calls) == 1

    def test_an_errored_artifact_is_retried_on_the_next_run(self, tmp_path,
                                                            monkeypatch):
        self._repo(tmp_path)
        self._seed(tmp_path, "error")
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert len(calls) == 1

    def test_a_completed_artifact_is_not_run_again(self, tmp_path, monkeypatch):
        self._repo(tmp_path)
        self._seed(tmp_path, "ok")
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []

    def test_a_heavy_env_artifact_runs_once_its_env_row_is_ready(self,
                                                                 tmp_path,
                                                                 monkeypatch):
        """The workflow this box actually needs: the env row arrives later."""
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="NEEDS_HEAVY_ENV")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []                      # no env row -> skipped
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "skip"

        write_envs(tmp_path, [{"repo": "owner/name", "status": "ready",
                               "blocked": [], "mode": "main", "venv": None,
                               "env_key": "abc123"}])
        calls.clear()
        run_external.main_with([], root=tmp_path)
        assert len(calls) == 1                  # the skip row did not block it


class TestLeaderboardRows:
    """D1: the leaderboard is one *logical* row per artifact."""

    def _skip_repo(self, tmp_path):
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="NEEDS_HEAVY_ENV")])

    def _rows(self, tmp_path):
        path = tmp_path / "results" / "external" / "leaderboard.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_a_repeated_skip_does_not_grow_the_leaderboard(self, tmp_path,
                                                           monkeypatch):
        """D1: the same skip on every run must not append a row every time."""
        self._skip_repo(tmp_path)
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        for _ in range(3):
            run_external.main_with([], root=tmp_path)
        rows = self._rows(tmp_path)
        assert len(rows) == 1, rows
        assert rows[0]["verdict"] == "skip"

    def test_a_failing_then_succeeding_artifact_keeps_one_verdict(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """D1: error -> ok leaves ONE logical verdict per artifact key."""
        make_repo(tmp_path, files={"s.py": "print('x')\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py")])
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)          # -> error
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)          # -> ok
        rows = self._rows(tmp_path)
        logical = run_external.latest_by_key(rows)
        assert list(logical) == ["owner/name|script|s.py"], rows
        assert logical["owner/name|script|s.py"]["verdict"] == "ok"
        lb = tmp_path / "results" / "external" / "leaderboard.jsonl"
        assert run_external.already_run(lb) == {"owner/name|script|s.py"}


class TestIsolation:
    """M7/M8/F3: the artifact runs in its own session, in a run copy."""

    def test_the_child_is_spawned_in_its_own_process_group(self, monkeypatch):
        """M7: the child must be a process-group leader of its own."""
        seen: dict = {}

        class FakePopen:
            pid = 1
            returncode = 0

            def __init__(self, *a, **k):
                seen.update(k)

            def communicate(self, timeout=None):
                return "", ""

        monkeypatch.setattr(run_external.subprocess, "Popen",
                            lambda *a, **k: FakePopen(*a, **k))
        run_external._launch(["python", "-c", "x"], timeout=10)
        assert seen.get("start_new_session") is True

    def test_a_real_script_runs_in_the_run_copy_not_in_the_checkout(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """M8: a real subprocess, a real write, a real relative read."""
        script = ("import pathlib\n"
                  "pathlib.Path('out.txt').write_text('written\\n')\n"
                  "print('ADJACENT=' + pathlib.Path('adjacent.txt').read_text())\n")
        make_repo(tmp_path, files={"s.py": script, "adjacent.txt": "sidecar\n"})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="RUN_BUNDLED_DATA")])

        real_select = run_external.select_env

        def select(row, env_rows, layout):
            choice = real_select(row, env_rows, layout)
            choice["python"] = [sys.executable]      # a real interpreter
            return choice

        monkeypatch.setattr(run_external, "select_env", select)
        assert run_external.main_with([], root=tmp_path) == 0

        src = tmp_path / "data" / "repos" / "owner-name"
        assert not (src / "out.txt").exists()        # nothing landed in the checkout
        assert (src / "adjacent.txt").read_text() == "sidecar\n"

        rows = [json.loads(line) for line in
                (tmp_path / "results" / "external" / "leaderboard.jsonl")
                .read_text().splitlines()]
        assert rows[0]["verdict"] == "ok"
        produced = [o for o in rows[0]["outputs"] if o.endswith("/out.txt")]
        assert produced, rows[0]["outputs"]
        out_file = tmp_path / "results" / "external" / produced[0]
        assert out_file.read_text() == "written\n"
        log = [o for o in rows[0]["outputs"] if o.endswith(".log")]
        assert log, rows[0]["outputs"]
        assert "ADJACENT=sidecar" in (tmp_path / "results" / "external"
                                      / log[0]).read_text()

    def test_a_real_notebook_runs_in_the_run_copy_not_in_the_checkout(self,
                                                                     tmp_path,
                                                                     monkeypatch):
        """M8b: the *notebook* cwd must be the run copy, not the checkout.

        The executed cell writes a file into its working directory and
        records that directory: the file has to land in the run copy (and
        from there in the evidence dir), never in ``data/repos/``.
        """
        cell = ("import os, pathlib\n"
                "pathlib.Path('nb_out.txt').write_text(os.getcwd())\n")
        make_repo(tmp_path, files={"nb.ipynb": json.dumps(notebook([cell]))})
        write_triage(tmp_path, [triage_row(path="nb.ipynb",
                                           verdict="RUN_BUNDLED_DATA")])

        real_select = run_external.select_env

        def select(row, env_rows, layout):
            choice = real_select(row, env_rows, layout)
            choice["python"] = [sys.executable]      # a real interpreter
            return choice

        monkeypatch.setattr(run_external, "select_env", select)
        assert run_external.main_with([], root=tmp_path) == 0

        src = tmp_path / "data" / "repos" / "owner-name"
        assert not (src / "nb_out.txt").exists()   # nothing landed in the checkout

        rows = [json.loads(line) for line in
                (tmp_path / "results" / "external" / "leaderboard.jsonl")
                .read_text().splitlines()]
        assert rows[0]["verdict"] == "ok", rows[0]
        produced = [o for o in rows[0]["outputs"] if o.endswith("/nb_out.txt")]
        assert produced, rows[0]["outputs"]
        cwd = (tmp_path / "results" / "external" / produced[0]).read_text()
        assert "repos" not in cwd, cwd
        assert "_runs" in cwd, cwd

    def test_a_symlink_cannot_write_back_into_the_checkout(self, tmp_path,
                                                           monkeypatch):
        """F3: a preserved symlink would let the artifact write home."""
        src = make_repo(tmp_path, files={"target.txt": "original\n"})
        (src / "link.txt").symlink_to(src / "target.txt")
        script = ("import pathlib\n"
                  "pathlib.Path('link.txt').open('a').write('mutated\\n')\n")
        make_repo(tmp_path, files={"s.py": script})
        write_triage(tmp_path, [triage_row(kind="script", path="s.py",
                                           verdict="RUN_BUNDLED_DATA")])

        real_select = run_external.select_env

        def select(row, env_rows, layout):
            choice = real_select(row, env_rows, layout)
            choice["python"] = [sys.executable]
            return choice

        monkeypatch.setattr(run_external, "select_env", select)
        run_external.main_with([], root=tmp_path)
        assert (src / "target.txt").read_text() == "original\n"

    def test_the_run_copy_contains_no_symlinks(self, tmp_path):
        src = make_repo(tmp_path, files={"target.txt": "original\n"})
        (src / "link.txt").symlink_to(src / "target.txt")
        (src / "dirlink").symlink_to(src)
        dest = tmp_path / "run"
        run_external._make_run_copy(src, dest)
        for path in dest.rglob("*"):
            assert not path.is_symlink(), path
        assert (dest / "link.txt").is_file()


class TestEvidence:
    """N1/N2 + the LOW output/collision items."""

    def _script_repo(self, tmp_path, path="s.py", body="print('hello')\n"):
        make_repo(tmp_path, files={path: body})
        write_triage(tmp_path, [triage_row(kind="script", path=path,
                                           verdict="RUN_BUNDLED_DATA")])

    def test_a_script_run_persists_its_output_as_a_log(self, tmp_path,
                                                       monkeypatch):
        self._script_repo(tmp_path)
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        logs = [o for o in row["outputs"] if o.endswith(".log")]
        assert logs, row["outputs"]
        assert (tmp_path / "results" / "external" / logs[0]).exists()

    def test_a_timeout_row_lists_no_output_that_was_never_produced(self,
                                                                    tmp_path,
                                                                    monkeypatch):
        """C2: the timeout happens before the runner wrote anything.

        A notebook killed before its first flush has no ``.log`` and no
        executed notebook on disk: the row must not list them.
        """
        make_repo(tmp_path, files={"nb.ipynb": json.dumps(notebook(["print('x')"]))})
        write_triage(tmp_path, [triage_row(path="nb.ipynb")])

        def timeout_launch(cmd, cwd=None, env=None, timeout=None):
            return {"ok": False, "error": "timed out after 600s", "timeout": True,
                    "stdout": "", "stderr": ""}

        monkeypatch.setattr(run_external, "_launch", timeout_launch)
        run_external.main_with([], root=tmp_path)
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "timeout"
        assert row["outputs"] == [], row["outputs"]
        env_dir = tmp_path / "results" / "external" / "owner-name"
        assert not (env_dir / "nb.log").exists()
        assert not (env_dir / "nb.ipynb").exists()
        for out in row["outputs"]:
            assert (tmp_path / "results" / "external" / out).exists(), out

    def test_a_timeout_leaves_no_empty_log_behind(self, tmp_path, monkeypatch):
        """D3: a 0-byte log is no evidence -- it must not be listed."""
        self._script_repo(tmp_path)

        def timeout_launch(cmd, cwd=None, env=None, timeout=None):
            return {"ok": False, "error": "timed out after 600s", "timeout": True,
                    "stdout": "", "stderr": ""}

        monkeypatch.setattr(run_external, "_launch", timeout_launch)
        run_external.main_with([], root=tmp_path)
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "timeout"
        for out in row["outputs"]:
            path = tmp_path / "results" / "external" / out
            assert path.exists(), out
            assert path.stat().st_size > 0, out
        env_dir = tmp_path / "results" / "external" / "owner-name"
        if env_dir.exists():
            for path in env_dir.rglob("*"):
                if path.is_file():
                    assert path.stat().st_size > 0, path

    def test_two_artifacts_with_the_same_stem_get_distinct_evidence_files(
            self, tmp_path, monkeypatch):
        """LOW: "a b.py" and "a-b.py" slug to the same stem."""
        self._script_repo(tmp_path, path="a b.py")
        write_triage(tmp_path, [triage_row(kind="script", path="a b.py"),
                                triage_row(kind="script", path="a-b.py")])
        make_repo(tmp_path, files={"a-b.py": "print('x')\n"})
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        run_external.main_with([], root=tmp_path)
        rows = [json.loads(line) for line in
                (tmp_path / "results" / "external" / "leaderboard.jsonl")
                .read_text().splitlines()]
        outs = [o for row in rows for o in row["outputs"]]
        assert len(outs) == len(set(outs)), outs
        assert len(outs) == 2, outs

    def test_an_artifact_path_that_escapes_the_run_copy_is_refused(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """LOW: "../../evil.py" must never be executed."""
        write_triage(tmp_path, [triage_row(kind="script", path="../../evil.py")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "error"
        assert "unsafe artifact path" in row["reason"]

    def test_an_absolute_artifact_path_is_refused(self, tmp_path, monkeypatch):
        write_triage(tmp_path, [triage_row(kind="script",
                                           path="/tmp/evil.py")])
        calls: list = []
        monkeypatch.setattr(run_external, "_launch", make_launcher(calls))
        run_external.main_with([], root=tmp_path)
        assert calls == []
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] == "error"

    def test_a_notebook_without_cells_is_recorded_not_crashed(self, tmp_path,
                                                               monkeypatch):
        """LOW: `cells: null` must not kill the runner."""
        nb = {"cells": None, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        make_repo(tmp_path, files={"nb.ipynb": json.dumps(nb)})
        write_triage(tmp_path, [triage_row(path="nb.ipynb")])
        monkeypatch.setattr(run_external, "_launch", make_launcher([]))
        assert run_external.main_with([], root=tmp_path) == 0
        row = json.loads((tmp_path / "results" / "external" / "leaderboard.jsonl")
                          .read_text().splitlines()[0])
        assert row["verdict"] in ("ok", "error")
        logs = [o for o in row["outputs"] if o.endswith(".log")]
        assert logs, row["outputs"]
        log = (tmp_path / "results" / "external" / logs[0]).read_text()
        assert "cells" in log
