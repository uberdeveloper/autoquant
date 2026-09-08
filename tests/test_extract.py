"""Tests for extract.py — spec parsing, validation, publishing."""
from __future__ import annotations

import pytest
import yaml

import extract


VALID_SPEC = {
    "meta": {
        "slug": "test-slug", "title": "T", "url": "https://a.com/x",
        "source": "S", "posted": "2026-08-09", "claim": "C",
        "author_evidence": {"sample": "s", "headline_metrics": "m",
                            "costs_included": False},
    },
    "data": {"universe": ["SPY"], "start": "1993-01-01"},
    "signal": {"definition": "close > sma200", "lag_bars": 1},
    "rules": {"direction": "long_only"},
    "costs": {"commission_bps": 1.0, "slippage_bps": 5.0},
    "validation": {"oos_start": "2026-08-09", "cost_sweep_bps": [0, 5]},
    "ambiguities": [{"id": "a1", "question": "q", "default": "d",
                     "alternatives": ["x"], "material": False}],
    "verdict": {"status": "PENDING", "reason": None},
}


class TestParseSpec:
    def test_bare_yaml(self):
        doc = extract.parse_spec("meta:\n  slug: x\n")
        assert doc == {"meta": {"slug": "x"}}

    def test_fenced_yaml(self):
        doc = extract.parse_spec("```yaml\nmeta:\n  slug: x\n```")
        assert doc == {"meta": {"slug": "x"}}

    def test_prose_around_yaml_raises(self):
        with pytest.raises((ValueError, yaml.YAMLError)):
            extract.parse_spec("here is the spec, hope it helps!")

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="not a YAML mapping"):
            extract.parse_spec("- just\n- a\n- list\n")


class TestValidateSpec:
    def test_valid_spec_has_no_errors(self):
        assert extract.validate_spec(VALID_SPEC) == []

    def test_missing_section_reported(self):
        doc = {k: v for k, v in VALID_SPEC.items() if k != "costs"}
        errors = extract.validate_spec(doc)
        assert any("costs" in e for e in errors)

    def test_zero_costs_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["costs"] = {"commission_bps": 0, "slippage_bps": 0}
        assert any("costs" in e for e in extract.validate_spec(doc))

    def test_zero_lag_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["signal"]["lag_bars"] = 0
        assert any("lag_bars" in e for e in extract.validate_spec(doc))

    def test_empty_ambiguities_rejected(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["ambiguities"] = []
        assert any("ambiguities" in e for e in extract.validate_spec(doc))

    def test_oos_start_must_equal_posted(self):
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["validation"]["oos_start"] = "2020-01-01"
        assert any("oos_start" in e for e in extract.validate_spec(doc))

    def test_untestable_needs_only_reason(self):
        doc = {"meta": {"slug": "x", "url": "u", "posted": "2026-08-09"},
               "verdict": {"status": "UNTESTABLE", "reason": "no deterministic rule"}}
        assert extract.validate_spec(doc) == []

    def test_untestable_without_reason_rejected(self):
        doc = {"meta": {"slug": "x"}, "verdict": {"status": "UNTESTABLE"}}
        assert any("reason" in e for e in extract.validate_spec(doc))


class TestExtractOne:
    ROW = {"url": "https://a.com/x", "title": "T", "source": "S",
           "posted": "2026-08-09", "slug": "test-slug", "page": "p.md"}

    def test_valid_reply_returns_spec(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\narticle body\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        reply = yaml.safe_dump(VALID_SPEC)
        recorded = {}

        def fake_run(cmd, input, capture_output, text, timeout):
            recorded["cmd"] = cmd
            recorded["input"] = input

            class Proc:
                returncode = 0
                stdout = reply
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert result["spec"]["meta"]["slug"] == "test-slug"
        assert "claude" in recorded["cmd"]
        assert "RUBRIC" in recorded["input"]

    def test_invalid_yaml_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = "not yaml at all: ["
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "error" in result

    def test_spec_failing_validation_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        bad = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        bad["signal"]["lag_bars"] = 0

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(bad)
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "invalid spec" in result["error"]
        assert "lag_bars" in result["error"]

    def test_model_flag_passed_through(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        seen = {}

        def fake_run(cmd, **k):
            seen["cmd"] = cmd

            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(VALID_SPEC)
                stderr = ""
            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        extract.extract_one(self.ROW, "RUBRIC", "claude-sonnet-4-6")
        assert "--model" in seen["cmd"]
