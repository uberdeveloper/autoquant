# tests/test_backtest_batch.py
"""Tests for backtest_batch.py — isolated, timeout-capped, resumable batch runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

import backtest_batch


class TestRunOne:
    def test_success_returns_output(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        class FakePopen:
            pid = 1
            returncode = 0

            def communicate(self, timeout=None):
                return "s: Sharpe 0.4 net\n", ""

        monkeypatch.setattr(backtest_batch.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        result = backtest_batch.run_one(spec, 10)
        assert "output" in result
        assert result["spec"].endswith("s.yaml")

    def test_timeout_is_recorded_not_raised(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")
        killed = []

        class FakePopen:
            pid = 987654321  # not a real pid; kill must be stubbed too
            returncode = None

            def communicate(self, timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return "", ""

        monkeypatch.setattr(backtest_batch.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        monkeypatch.setattr(backtest_batch, "_kill_group",
                            lambda pid: killed.append(pid))
        result = backtest_batch.run_one(spec, 10)
        assert "timed out" in result["error"]
        assert killed == [987654321]  # the whole group got the kill

    def test_nonzero_exit_records_stderr_tail(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        class FakePopen:
            pid = 1
            returncode = 1

            def communicate(self, timeout=None):
                return "", "no price data for SPY"

        monkeypatch.setattr(backtest_batch.subprocess, "Popen",
                            lambda *a, **k: FakePopen())
        result = backtest_batch.run_one(spec, 10)
        assert "exit 1" in result["error"]
        assert "no price data" in result["error"]

    def test_missing_interpreter_is_an_error_not_a_crash(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_popen(*a, **k):
            raise FileNotFoundError(2, "No such file or directory", "uv")

        monkeypatch.setattr(backtest_batch.subprocess, "Popen", fake_popen)
        result = backtest_batch.run_one(spec, 10)
        assert "could not launch" in result["error"]

    def test_timeout_kills_whole_process_group(self, tmp_path, monkeypatch):
        """The python grandchild under `uv` must die with the wrapper --
        otherwise an orphan finishes later and silently 'completes' a spec
        the batch already recorded as failed."""
        pidfile = tmp_path / "grandchild.pid"
        grandchild = tmp_path / "grandchild.py"
        grandchild.write_text(
            "import os, time\n"
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        # wrapper plays the role of `uv`: spawns the python grandchild, stays alive
        wrapper = [
            sys.executable, "-c",
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(grandchild)!r}])\n"
            "time.sleep(30)\n"
        ]
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")
        monkeypatch.setattr(backtest_batch, "_backtest_cmd", lambda s: wrapper)

        result = backtest_batch.run_one(spec, 2)
        assert "timed out" in result["error"]

        deadline = time.time() + 5
        while not pidfile.exists() and time.time() < deadline:
            time.sleep(0.05)
        pid = int(pidfile.read_text())
        for _ in range(100):  # ~5s for the reaper to collect it
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("grandchild survived the timeout (orphaned)")


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

    def test_malformed_spec_does_not_abort_batch(self, tmp_path, monkeypatch):
        ran = []
        self._patch(tmp_path, monkeypatch, ran)
        (tmp_path / "specs" / "broken.yaml").write_text("meta:\n  slug: [unclosed\n")
        (tmp_path / "specs" / "nokey.yaml").write_text("meta:\n  other: x\n")
        assert backtest_batch.main_with(["--jobs", "2"]) == 0
        failed = (tmp_path / "results" / "failed.jsonl").read_text()
        assert "broken.yaml" in failed
        assert "nokey.yaml" in failed
        assert ran == ["a.yaml", "b.yaml"]  # valid specs still ran

    def test_no_specs_returns_zero_with_message(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(backtest_batch, "ROOT", tmp_path)
        monkeypatch.setattr(backtest_batch, "SPECS", tmp_path / "empty")
        monkeypatch.setattr(backtest_batch, "LEADERBOARD", tmp_path / "results" / "leaderboard.jsonl")
        monkeypatch.setattr(backtest_batch, "FAILED", tmp_path / "results" / "failed.jsonl")
        assert backtest_batch.main_with([]) == 0
        assert "nothing to backtest" in capsys.readouterr().out
