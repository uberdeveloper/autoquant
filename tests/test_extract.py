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

    def test_scalar_verdict_rejected_not_crash(self):
        errors = extract.validate_spec({"verdict": "UNTESTABLE"})
        assert any("verdict" in e for e in errors)


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

    def test_timeout_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_run(*a, **k):
            raise extract.subprocess.TimeoutExpired(cmd="claude", timeout=600)

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "timed out" in result["error"]

    def test_nonzero_exit_returns_error(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_run(*a, **k):
            class Proc:
                returncode = 1
                stdout = ""
                stderr = "boom"

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "claude exited 1" in result["error"]

    def test_scalar_verdict_becomes_error_not_crash(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = "verdict: UNTESTABLE\n"
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert "invalid spec" in result["error"]

    def test_string_costs_do_not_crash(self, tmp_path, monkeypatch):
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")
        monkeypatch.setattr(extract, "ROOT", tmp_path)
        doc = yaml.safe_load(yaml.safe_dump(VALID_SPEC))
        doc["costs"] = {"commission_bps": "5", "slippage_bps": "5"}

        def fake_run(*a, **k):
            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(doc)
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)
        result = extract.extract_one(self.ROW, "RUBRIC", None)
        assert ("spec" in result) != ("error" in result)  # outcome, not exception


class TestPublish:
    def test_success_writes_spec_atomically(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "test-slug"}
        result = {"url": row["url"], "spec": yaml.safe_load(yaml.safe_dump(VALID_SPEC))}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "spec"
        assert (tmp_path / "specs" / "test-slug.yaml").exists()
        assert not (tmp_path / "specs" / "test-slug.yaml.tmp").exists()

    def test_untestable_writes_marker_with_reason(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "vix-thing"}
        result = {"url": row["url"], "spec": {
            "meta": {"slug": "vix-thing", "url": "u", "posted": "2026-08-09"},
            "verdict": {"status": "UNTESTABLE", "reason": "discretionary"}}}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "untestable"
        text = (tmp_path / "specs" / "vix-thing.untestable").read_text()
        assert "discretionary" in text
        assert not (tmp_path / "specs" / "vix-thing.yaml").exists()

    def test_error_writes_marker(self, tmp_path):
        row = {"url": "https://a.com/x", "slug": "vix-thing"}
        result = {"url": row["url"], "error": "claude exited 1: boom"}
        stage = extract.publish(row, result, tmp_path / "specs")
        assert stage == "extract_failed"
        assert "claude exited 1" in (tmp_path / "specs" / "vix-thing.error").read_text()

    def test_success_supersedes_prior_error_marker(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir()
        (specs / "test-slug.error").write_text("{}\n")
        row = {"url": "https://a.com/x", "slug": "test-slug"}
        result = {"url": row["url"], "spec": yaml.safe_load(yaml.safe_dump(VALID_SPEC))}
        stage = extract.publish(row, result, specs)
        assert stage == "spec"
        assert (specs / "test-slug.yaml").exists()
        assert not (specs / "test-slug.error").exists()


class TestPending:
    def rows(self):
        return [
            {"url": "https://a.com/a", "slug": "done", "stage": "triaged"},
            {"url": "https://a.com/b", "slug": "failed", "stage": "triaged"},
            {"url": "https://a.com/c", "slug": "fresh", "stage": "triaged"},
            {"url": "https://a.com/d", "slug": "fetched-already", "stage": "fetched"},
        ]

    def setup(self, specs):
        specs.mkdir(parents=True)
        (specs / "done.yaml").write_text("meta:\n  slug: done\n")
        (specs / "failed.error").write_text("{}\n")

    def test_skips_specd_and_errored(self, tmp_path):
        specs = tmp_path / "specs"
        self.setup(specs)
        todo = extract.pending(self.rows(), specs, retry=False)
        assert [r["slug"] for r in todo] == ["fresh"]

    def test_retry_reincludes_errored(self, tmp_path):
        specs = tmp_path / "specs"
        self.setup(specs)
        todo = extract.pending(self.rows(), specs, retry=True)
        assert [r["slug"] for r in todo] == ["failed", "fresh"]

    def test_untestable_is_final_even_with_retry(self, tmp_path):
        specs = tmp_path / "specs"
        specs.mkdir(parents=True)
        (specs / "done.untestable").write_text("{}\n")
        rows = [{"url": "u", "slug": "done", "stage": "triaged"}]
        assert extract.pending(rows, specs, retry=True) == []


class TestMain:
    def test_publishes_without_touching_articles_jsonl(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "data").mkdir()
        articles = tmp_path / "data" / "articles.jsonl"
        articles.write_text(extract.json.dumps({
            "url": "https://a.com/x", "title": "T", "source": "S",
            "posted": "2026-08-09", "slug": "test-slug",
            "page": "data/pages/p.md", "stage": "triaged",
        }) + "\n")
        page = tmp_path / "data" / "pages" / "p.md"
        page.parent.mkdir(parents=True)
        page.write_text("---\n\nbody\n")
        before = articles.read_text()

        monkeypatch.setattr(extract, "ROOT", tmp_path)
        monkeypatch.setattr(extract, "ARTICLES", articles)
        monkeypatch.setattr(extract, "SPECS", tmp_path / "specs")
        monkeypatch.setattr(extract, "PROMPT", tmp_path / "prompt.md")
        (tmp_path / "prompt.md").write_text("RUBRIC")

        calls = []

        def fake_run(*a, **k):
            calls.append(1)

            class Proc:
                returncode = 0
                stdout = yaml.safe_dump(VALID_SPEC)
                stderr = ""

            return Proc()

        monkeypatch.setattr(extract.subprocess, "run", fake_run)

        assert extract.main_with(["--jobs", "1"]) == 0
        assert (tmp_path / "specs" / "test-slug.yaml").exists()
        assert articles.read_text() == before  # read-only: never rewritten

        # resume: a second run finds nothing to do and makes zero CLI calls
        assert extract.main_with(["--jobs", "1"]) == 0
        assert len(calls) == 1
