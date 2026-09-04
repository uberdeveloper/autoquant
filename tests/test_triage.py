"""Tests for triage.py — prompt construction and reply parsing."""
from __future__ import annotations

import json

import pytest

import triage


ROW = {
    "title": "Volatility Premium",
    "source": "Blog",
    "url": "https://a.com/x",
    "posted": "2026-08-09",
    "slug": "volatility-premium",
}


class TestStripBody:
    def test_drops_fetch_header(self, tmp_path):
        page = tmp_path / "p.md"
        page.write_text("# Title\n\n- source: S\n\n---\n\nbody line\n")
        assert triage.strip_body(page) == "body line"

    def test_no_separator_returns_whole_text(self, tmp_path):
        page = tmp_path / "p.md"
        page.write_text("just the body\n")
        assert triage.strip_body(page) == "just the body"


class TestBuildPrompt:
    def test_contains_rubric_and_payload_fields(self):
        prompt = triage.build_prompt("RUBRIC", ROW, "article text")
        assert prompt.startswith("RUBRIC\n\n---\n")
        assert '"title": "Volatility Premium"' in prompt
        assert '"text": "article text"' in prompt
        assert "no markdown fences" in prompt

    def test_long_body_is_truncated_with_marker(self):
        prompt = triage.build_prompt("R", ROW, "x" * (triage.MAX_ARTICLE_CHARS + 5000))
        assert "[...truncated for length...]" in prompt
        # truncation marker plus the rest of the prompt bound the text length
        assert len(prompt) < triage.MAX_ARTICLE_CHARS + 500


class TestParseJson:
    def test_bare_json(self):
        assert triage.parse_json('{"score": 4}') == {"score": 4}

    def test_fenced_json(self):
        text = "```json\n{\"score\": 5}\n```"
        assert triage.parse_json(text) == {"score": 5}

    def test_unlabelled_fence(self):
        text = "```\n{\"score\": 3}\n```"
        assert triage.parse_json(text) == {"score": 3}

    def test_prose_wrapped_object(self):
        text = 'Here is my assessment: {"score": 2, "why": "weak"} hope that helps'
        assert triage.parse_json(text) == {"score": 2, "why": "weak"}

    def test_no_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="no JSON object"):
            triage.parse_json("there is nothing here")


class TestScoreOne:
    def test_cli_success_attaches_article_fields(self, tmp_path, monkeypatch):
        row = dict(ROW, page="data/pages/x.md", slug="volatility-premium")
        page = tmp_path / "data" / "pages" / "x.md"
        page.parent.mkdir(parents=True)
        page.write_text("---\n\nbody\n")

        reply = json.dumps({"score": 4, "asset_class": "equity"})
        recorded = {}

        def fake_run(cmd, input, capture_output, text, timeout):
            recorded["cmd"] = cmd
            recorded["input"] = input

            class Proc:
                returncode = 0
                stdout = reply
                stderr = ""

            return Proc()

        monkeypatch.setattr(triage, "ROOT", tmp_path)
        monkeypatch.setattr(triage.subprocess, "run", fake_run)
        result = triage.score_one(row, "RUBRIC", None)
        assert result["score"] == 4
        assert result["url"] == row["url"]
        assert result["slug"] == row["slug"]
        assert "claude" in recorded["cmd"]
        assert "--model" not in recorded["cmd"]

    def test_timeout_returns_error_row(self, tmp_path, monkeypatch):
        row = dict(ROW, page="p.md")
        page = tmp_path / "p.md"
        page.write_text("---\n\nbody\n")

        def fake_run(*a, **k):
            raise triage.subprocess.TimeoutExpired(cmd="claude", timeout=300)

        monkeypatch.setattr(triage, "ROOT", tmp_path)
        monkeypatch.setattr(triage.subprocess, "run", fake_run)
        result = triage.score_one(row, "RUBRIC", None)
        assert "timed out" in result["error"]
