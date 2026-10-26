# tests/test_codegen_compare.py
"""Tests for codegen --compare-models fan-out and --pick promotion."""
from __future__ import annotations

import codegen
import llm
import pytest

VALID_MODULE = (
    "import pandas as pd\n"
    "def signal(df, **params):\n"
    "    return (df.close > df.close.rolling(5).mean()).astype(float)\n"
)


def setup_specs(tmp_path, n=1):
    specs = tmp_path / "specs"
    specs.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (specs / f"s{i}.yaml").write_text(
            f"meta:\n  slug: s{i}\nsignal:\n  definition: d\n  lag_bars: 1\n")
    return specs


def wire(tmp_path, monkeypatch, complete):
    monkeypatch.setattr(codegen, "ROOT", tmp_path)
    monkeypatch.setattr(codegen, "SPECS", tmp_path / "specs")
    monkeypatch.setattr(codegen, "STRATEGIES", tmp_path / "strategies")
    monkeypatch.setattr(codegen, "COMPARE", tmp_path / "strategies" / "compare")
    monkeypatch.setattr(codegen.llm, "preflight", lambda: None)
    monkeypatch.setattr(codegen.llm, "preflight_for", lambda t: None)
    monkeypatch.setattr(codegen.llm, "complete", complete)


class TestCompareFanout:
    def test_each_label_gets_its_own_artifact(self, tmp_path, monkeypatch):
        calls = []

        def fake_complete(prompt, **k):
            calls.append(k.get("template"))
            return VALID_MODULE

        wire(tmp_path, monkeypatch, fake_complete)
        setup_specs(tmp_path)
        rc = codegen.main_with(["--jobs", "1", "--limit", "1",
                                "--compare-models", "a=cmd-a,b=cmd-b"])
        assert rc == 0
        strategies = tmp_path / "strategies"
        assert (strategies / "compare" / "a" / "s0.py").exists()
        assert (strategies / "compare" / "b" / "s0.py").exists()
        assert not (strategies / "s0.py").exists()  # never published to trusted name
        assert sorted(calls) == ["cmd-a", "cmd-b"]

    def test_compare_subdir_invisible_to_pending_specs(self, tmp_path):
        specs = setup_specs(tmp_path)
        strategies = tmp_path / "strategies"
        (strategies / "compare" / "a").mkdir(parents=True)
        (strategies / "compare" / "a" / "s0.py").write_text("x = 1\n")
        assert [p.stem for p in codegen.pending_specs(specs, strategies, False)] == ["s0"]

    def test_smoke_fail_writes_per_label_error(self, tmp_path, monkeypatch):
        wire(tmp_path, monkeypatch, lambda prompt, **k: "x = 1\n")
        setup_specs(tmp_path)
        rc = codegen.main_with(["--jobs", "1", "--limit", "1",
                                "--compare-models", "a=cmd-a"])
        assert rc == 0  # comparison never aborts the batch on stage-level errors
        assert (tmp_path / "strategies" / "compare" / "a" / "s0.error").exists()

    def test_summary_table_printed(self, tmp_path, monkeypatch, capsys):
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        setup_specs(tmp_path)
        codegen.main_with(["--jobs", "1", "--limit", "1",
                           "--compare-models", "a=cmd-a"])
        out = capsys.readouterr().out
        assert "label" in out and "a" in out

    def test_invalid_compare_spec_exits(self, tmp_path, monkeypatch):
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        setup_specs(tmp_path)
        with pytest.raises(SystemExit, match="expected 'label=cmd'"):
            codegen.main_with(["--compare-models", "label="])


class TestPick:
    def test_promotes_passed_candidate(self, tmp_path, monkeypatch, capsys):
        strategies = tmp_path / "strategies"
        cand = strategies / "compare" / "a"
        cand.mkdir(parents=True)
        (cand / "s0.py").write_text(VALID_MODULE)
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        rc = codegen.main_with(["--pick", "a", "--slug", "s0"])
        assert rc == 0
        assert (strategies / "s0.py").read_text() == VALID_MODULE
        assert (strategies / "compare" / "a" / "s0.py").exists()  # original kept

    def test_refuses_missing_candidate(self, tmp_path, monkeypatch):
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        with pytest.raises(SystemExit, match="no candidate"):
            codegen.main_with(["--pick", "zzz", "--slug", "s0"])

    def test_refuses_failed_candidate(self, tmp_path, monkeypatch):
        strategies = tmp_path / "strategies"
        cand = strategies / "compare" / "a"
        cand.mkdir(parents=True)
        (cand / "s0.error").write_text('{"error": "boom"}\n')
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        with pytest.raises(SystemExit, match="failed"):
            codegen.main_with(["--pick", "a", "--slug", "s0"])

    def test_pick_clears_stale_error_marker(self, tmp_path, monkeypatch):
        strategies = tmp_path / "strategies"
        cand = strategies / "compare" / "a"
        cand.mkdir(parents=True)
        (cand / "s0.py").write_text(VALID_MODULE)
        (strategies / "s0.error").write_text('{"error": "old"}\n')
        wire(tmp_path, monkeypatch, lambda prompt, **k: VALID_MODULE)
        assert codegen.main_with(["--pick", "a", "--slug", "s0"]) == 0
        assert not (strategies / "s0.error").exists()


class TestPerLabelCircuitBreaker:
    def test_aborted_label_keeps_other_label_working(self, tmp_path, monkeypatch,
                                                     capsys):
        setup_specs(tmp_path, n=5)

        def fake_complete(prompt, *, model=None, extra=None, timeout=600,
                          template=None):
            if template == "cmd-bad":
                raise llm.LLMError("opencode exited 1: auth expired")
            return VALID_MODULE

        wire(tmp_path, monkeypatch, fake_complete)
        rc = codegen.main_with(["--jobs", "1", "--limit", "5",
                                "--compare-models", "bad=cmd-bad,good=cmd-good"])
        assert rc == 1
        assert "aborting" in capsys.readouterr().out
        strategies = tmp_path / "strategies"
        # bad label stopped at the k=3 breaker
        assert len(list((strategies / "compare" / "bad").glob("*.error"))) == 3
        # good label completed all 5
        assert len(list((strategies / "compare" / "good").glob("*.py"))) == 5


def test_compare_preflights_only_selected_commands(tmp_path, monkeypatch):
    wire(tmp_path, monkeypatch, lambda prompt, **kwargs: VALID_MODULE)
    setup_specs(tmp_path)
    checked = []
    def unavailable_default():
        raise llm.LLMError('opencode unavailable')
    monkeypatch.setattr(codegen.llm, 'preflight', unavailable_default)
    monkeypatch.setattr(codegen.llm, 'preflight_for', checked.append)
    assert codegen.main_with(['--jobs', '1', '--compare-models', 'c=claude -p']) == 0
    assert checked == ['claude -p']
    assert (tmp_path / 'strategies' / 'compare' / 'c' / 's0.py').exists()
