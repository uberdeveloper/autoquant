"""Tests for fetch.py — article text extraction and the fetch_one state machine."""
from __future__ import annotations

import json

import pytest
import requests

import fetch


@pytest.fixture
def pages_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch, "PAGES", tmp_path / "pages")
    return tmp_path / "pages"


def make_row(url="https://blog.example.com/post"):
    return {
        "url": url,
        "title": "A Title",
        "source": "Src",
        "posted": "2026-08-09",
        "stage": "harvested",
    }


class StubRobots:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.checked = []

    def allows(self, url):
        self.checked.append(url)
        return self.allowed


class TestPagePath:
    def test_deterministic_and_suffix(self, pages_dir):
        p1 = fetch.page_path("https://a.com/x")
        p2 = fetch.page_path("https://a.com/x")
        assert p1 == p2
        assert p1.suffix == ".md"
        assert p1.parent == fetch.PAGES


class TestLoadSave:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert fetch.load(tmp_path / "nope.jsonl") == []

    def test_save_load_roundtrip(self, tmp_path):
        rows = [{"url": "a"}, {"url": "b"}]
        path = tmp_path / "q.jsonl"
        fetch.save(path, rows)
        assert fetch.load(path) == rows

    def test_save_is_atomic_no_tmp_left_behind(self, tmp_path):
        path = tmp_path / "q.jsonl"
        fetch.save(path, [{"url": "a"}])
        assert not path.with_suffix(path.suffix + ".tmp").exists()
        assert json.loads(path.read_text().splitlines()[0]) == {"url": "a"}


class TestExtract:
    def test_title_and_article_body(self):
        html = (
            "<html><head><title>Page Title</title></head><body>"
            "<nav>menu junk</nav>"
            "<article><p>" + ("word " * 200) + "</p></article>"
            "</body></html>"
        )
        title, text = fetch.extract(html)
        assert title == "Page Title"
        assert "word" in text
        assert "menu junk" not in text

    def test_script_and_style_stripped(self):
        html = (
            "<html><body><article>"
            "<script>var x = 1;</script><style>p{}</style>"
            "<p>" + ("real text " * 150) + "</p>"
            "</article></body></html>"
        )
        _, text = fetch.extract(html)
        assert "var x" not in text
        assert "real text" in text

    def test_falls_back_to_body_when_container_too_short(self):
        html = (
            "<html><body><article>tiny</article>"
            "<p>" + ("body words " * 200) + "</p></body></html>"
        )
        _, text = fetch.extract(html)
        assert "body words" in text

    def test_blank_lines_collapsed(self):
        html = "<html><body><article><p>a</p>\n\n\n<p>b</p></article></body></html>"
        html += "<p>" + "filler " * 300 + "</p>"
        _, text = fetch.extract(html)
        assert "\n\n" not in text


LONG_HTML = "<html><body><article>" + ("long body text " * 200) + "</article></body></html>"


class OkSession:
    """Session stub serving LONG_HTML without touching the network."""

    def __init__(self, content=LONG_HTML):
        self.content = content

    def get(self, url, timeout):
        resp = requests.Response()
        resp.status_code = 200
        resp._content = self.content.encode()
        return resp


class TestFetchOne:
    def test_success_writes_page_and_advances_stage(self, pages_dir):
        row = fetch.fetch_one(OkSession(), StubRobots(True), make_row())
        assert row["stage"] == "fetched"
        assert "fetch_error" not in row
        page = pages_dir / row["page"].split("/")[-1]
        assert page.exists()
        assert row["page_chars"] == page.stat().st_size
        # header format that triage.strip_body partitions on
        assert "\n---\n\n" in page.read_text()

    def test_robots_disallow_records_failure(self, pages_dir):
        stub = StubRobots(False)
        row = fetch.fetch_one(OkSession(), stub, make_row())
        assert row["stage"] == "fetch_failed"
        assert row["fetch_error"] == "robots.txt disallows"
        assert stub.checked == [row["url"]]
        assert not pages_dir.exists() or not any(pages_dir.iterdir())

    def test_http_error_records_failure(self, pages_dir):
        class FailingSession:
            def get(self, url, timeout):
                raise requests.ConnectionError("boom")

        row = fetch.fetch_one(FailingSession(), StubRobots(True), make_row())
        assert row["stage"] == "fetch_failed"
        assert "ConnectionError" in row["fetch_error"]

    def test_short_body_is_paywall_failure_not_a_page(self, pages_dir):
        row = fetch.fetch_one(OkSession("<html><body><p>too short</p></body></html>"),
                              StubRobots(True), make_row())
        assert row["stage"] == "fetch_failed"
        assert "paywall or JS-rendered" in row["fetch_error"]
        assert not pages_dir.exists() or not any(pages_dir.iterdir())

    def test_cached_page_never_hits_the_network(self, pages_dir):
        dest = fetch.page_path(make_row()["url"])
        dest.parent.mkdir(parents=True)
        dest.write_text("# cached\n\n" + "x" * (fetch.MIN_CHARS + 10))

        class MustNotFetch:
            def get(self, *a, **k):
                raise AssertionError("network hit on cached page")

        row = fetch.fetch_one(MustNotFetch(), StubRobots(True), make_row())
        assert row["stage"] == "fetched"
        assert row["page_chars"] >= fetch.MIN_CHARS

    def test_cached_page_below_min_chars_is_refetched(self, pages_dir):
        dest = fetch.page_path(make_row()["url"])
        dest.parent.mkdir(parents=True)
        dest.write_text("tiny")

        row = fetch.fetch_one(OkSession(), StubRobots(True), make_row())
        assert row["stage"] == "fetched"
