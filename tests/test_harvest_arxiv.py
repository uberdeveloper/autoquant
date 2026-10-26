# tests/test_harvest_arxiv.py
"""Tests for harvest_arxiv.py — arXiv API harvesting with abstract-as-body."""
from __future__ import annotations

import json

import harvest_arxiv
import pytest

ATOM_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2610.02834v1</id>
    <updated>2026-10-01T17:59:59Z</updated>
    <published>2026-10-01T17:59:59Z</published>
    <title>
      Momentum Crashes in
      Equity Factors
    </title>
    <summary>  We study momentum crashes
in equity factor portfolios.
      Our results show drawdowns of 40%.
    </summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom"
        term="q-fin.TR" />
    <category term="q-fin.TR" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2610.02835v2</id>
    <updated>2026-10-02T10:00:00Z</updated>
    <published>2026-10-02T09:00:00Z</published>
    <title>Volatility Forecasting with Realized Measures</title>
    <summary>A short one.</summary>
    <author><name>A. Author</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom"
        term="econ.EM" />
    <category term="econ.EM" />
  </entry>
</feed>
"""


class TestParseAtom:
    def test_rows_use_canonical_versionless_abs_url(self):
        rows = harvest_arxiv.parse_atom(ATOM_PAGE)
        assert rows[0]["url"] == "https://arxiv.org/abs/2610.02834"
        assert rows[1]["url"] == "https://arxiv.org/abs/2610.02835"

    def test_slugs_prefixed_and_whitespace_normalized(self):
        rows = harvest_arxiv.parse_atom(ATOM_PAGE)
        assert rows[0]["slug"] == "arxiv-momentum-crashes-in-equity-factors"

    def test_source_posted_and_blurb(self):
        rows = harvest_arxiv.parse_atom(ATOM_PAGE)
        r = rows[0]
        assert r["source"] == "arXiv"
        assert r["stage"] == "harvested"  # main() flips to fetched after cache write
        assert r["posted"] == "2026-10-01"
        assert "momentum crashes" in r["blurb"]
        assert "40%" in r["blurb"]
        assert r["arxiv_id"] == "2610.02834"

    def test_whitespace_collapsed_in_title_and_summary(self):
        rows = harvest_arxiv.parse_atom(ATOM_PAGE)
        assert rows[0]["title"] == "Momentum Crashes in Equity Factors"
        assert "\n" not in rows[0]["blurb"]


class TestCategories:
    def test_qfin_star_expands_to_all_subcategories(self):
        cats = harvest_arxiv.expand_categories(["q-fin*"])
        assert set(cats) == {"q-fin.CP", "q-fin.GN", "q-fin.MF", "q-fin.PM",
                             "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR"}

    def test_concrete_categories_pass_through(self):
        assert harvest_arxiv.expand_categories(["econ.EM", "q-fin.TR"]) == \
            ["econ.EM", "q-fin.TR"]

    def test_search_query_ors_categories(self):
        q = harvest_arxiv.search_query(["econ.EM", "econ.GN"])
        assert q == "(cat:econ.EM OR cat:econ.GN)"


class TestBodyCache:
    def test_abstract_written_in_fetch_page_format(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        row = {"url": "https://arxiv.org/abs/2610.02834",
               "title": "Momentum Crashes", "source": "arXiv",
               "posted": "2026-10-01"}
        harvest_arxiv.write_body_cache(row, "The abstract text.", pages)
        from fetch import page_path
        dest = pages / page_path(row["url"]).name
        assert dest.exists()
        assert "The abstract text." in dest.read_text()
        assert "- source: arXiv" in dest.read_text()
        assert row["stage"] == "fetched"
        assert row["page"] == str(dest)
        assert row["page_chars"] == dest.stat().st_size


class TestLimitCap:
    def test_limit_hard_capped_at_1000(self):
        assert harvest_arxiv.effective_limit(5000) == 1000
        assert harvest_arxiv.effective_limit(50) == 50
        assert harvest_arxiv.effective_limit(0) == 100  # default


class TestMain:
    def _wire(self, tmp_path, monkeypatch):
        out = tmp_path / "articles.jsonl"
        pages = tmp_path / "pages"
        pages.mkdir()
        calls = []

        class FakeResp:
            text = ATOM_PAGE

            def raise_for_status(self):
                pass

        def fake_get(self, url, **k):
            calls.append(url)
            if "start=2" in url:
                return type("EmptyResp", (), {"text": "<feed />",
                                                "raise_for_status": lambda self: None})()
            return FakeResp()

        monkeypatch.setattr(harvest_arxiv.requests.Session, "get", fake_get)
        return out, pages, calls

    def test_harvest_writes_fetched_rows_and_dedupes(self, tmp_path, monkeypatch):
        out, pages, calls = self._wire(tmp_path, monkeypatch)
        argv = ["--out", str(out), "--pages", str(pages), "--sleep", "0",
                "--limit", "2"]
        assert harvest_arxiv.main_with(argv) == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 2
        assert all(r["stage"] == "fetched" for r in rows)
        assert all((pages / f.split("/")[-1]).exists() for r in rows
                   for f in [r["page"]] if f)
        assert any("sortBy=submittedDate" in c for c in calls)
        # rerun: all urls already known, nothing appended
        assert harvest_arxiv.main_with(argv) == 0
        assert len(out.read_text().splitlines()) == 2

    def test_pagination_stops_on_short_page(self, tmp_path, monkeypatch):
        out, pages, calls = self._wire(tmp_path, monkeypatch)

        class FakeResp:
            text = ATOM_PAGE

            def raise_for_status(self):
                pass

        def fake_get(self, url, **k):
            calls.append(url)
            return FakeResp()

        monkeypatch.setattr(harvest_arxiv.requests.Session, "get", fake_get)
        # The short first page proves the API is exhausted immediately.
        harvest_arxiv.main_with(["--out", str(out), "--pages", str(pages),
                                 "--sleep", "0", "--limit", "4"])
        assert len(calls) == 1
        assert len(out.read_text().splitlines()) == 2

    def test_known_page_does_not_hide_later_new_results(self, tmp_path, monkeypatch):
        out, pages, calls = self._wire(tmp_path, monkeypatch)
        monkeypatch.setattr(harvest_arxiv, "PAGE_SIZE", 2)
        known = harvest_arxiv.parse_atom(ATOM_PAGE)
        out.write_text("".join(json.dumps(row) + "\n" for row in known))
        later_page = ATOM_PAGE.replace("02834", "09999").replace("02835", "09998")

        class FakeResp:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        def fake_get(self, url, **kwargs):
            calls.append(url)
            return FakeResp(later_page if "start=2" in url else ATOM_PAGE)

        monkeypatch.setattr(harvest_arxiv.requests.Session, "get", fake_get)
        harvest_arxiv.main_with(["--out", str(out), "--pages", str(pages),
                                 "--sleep", "0", "--limit", "2"])
        assert any("start=2" in url for url in calls)
        assert len(out.read_text().splitlines()) == 4

    def test_stops_after_bounded_known_pages(self, tmp_path, monkeypatch):
        out, pages, calls = self._wire(tmp_path, monkeypatch)
        monkeypatch.setattr(harvest_arxiv, "PAGE_SIZE", 2)
        monkeypatch.setattr(harvest_arxiv, "MAX_CONSECUTIVE_KNOWN_PAGES", 2)
        known = harvest_arxiv.parse_atom(ATOM_PAGE)
        out.write_text("".join(json.dumps(row) + "\n" for row in known))

        harvest_arxiv.main_with(
            ["--out", str(out), "--pages", str(pages), "--sleep", "0", "--limit", "2"]
        )

        assert len(calls) == 2
        assert len(out.read_text().splitlines()) == 2
