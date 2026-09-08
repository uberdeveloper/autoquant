# tests/test_backtest_batch.py
"""Tests for backtest_batch.py — isolated, timeout-capped, resumable batch runs."""
from __future__ import annotations

import json

import pytest

import backtest_batch


class TestRunOne:
    def test_success_returns_output(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = "s: Sharpe 0.4 net\n"
                stderr = ""
            return Proc()

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "output" in result
        assert result["spec"].endswith("s.yaml")

    def test_timeout_is_recorded_not_raised(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            raise backtest_batch.subprocess.TimeoutExpired(cmd="x", timeout=10)

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "timed out" in result["error"]

    def test_nonzero_exit_records_stderr_tail(self, tmp_path, monkeypatch):
        spec = tmp_path / "s.yaml"
        spec.write_text("meta:\n  slug: s\n")

        def fake_run(*a, **k):
            class Proc:
                returncode = 1
                stdout = ""
                stderr = "no price data for SPY"
            return Proc()

        monkeypatch.setattr(backtest_batch.subprocess, "run", fake_run)
        result = backtest_batch.run_one(spec, 10)
        assert "exit 1" in result["error"]
        assert "no price data" in result["error"]


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

    def test_no_specs_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest_batch, "SPECS", tmp_path / "empty")
        assert backtest_batch.main_with([]) == 1
