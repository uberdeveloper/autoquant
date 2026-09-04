"""Tests for harvest.py — link parsing from Quantocracy pages."""
from __future__ import annotations

import harvest


class TestSlugify:
    def test_basic(self):
        assert harvest.slugify("Some Post About Volatility") == "some-post-about-volatility"

    def test_collapses_punctuation(self):
        assert harvest.slugify("Sharpe!! ratio???  (part 2)") == "sharpe-ratio-part-2"

    def test_truncates_and_strips_trailing_dash(self):
        s = harvest.slugify("a" * 100, maxlen=10)
        assert len(s) == 10
        assert not s.endswith("-")

    def test_trailing_punctuation_does_not_leave_dash(self):
        assert harvest.slugify("title!") == "title"


MASHUP_HTML = """
<html><body>
<article class="qo-entry">
  <a class="qo-title" href="https://blog.example.com/post-1">Momentum Works [Alpha Blog]</a>
  <summary class="qo-description">A study of momentum.</summary>
  <footer>- 16 hours ago, 9 Aug 2026, 08:08pm -</footer>
</article>
<article class="qo-entry">
  <a class="qo-title" href="https://blog.example.com/post-2">No Source Bracket Here</a>
  <footer>no date in footer</footer>
</article>
<article class="qo-entry">
  <p>no anchor link at all</p>
</article>
</body></html>
"""


class TestParsePage:
    def test_parses_full_entry(self):
        rows = harvest.parse_page(MASHUP_HTML)
        assert len(rows) == 2  # entry without an anchor is skipped
        first = rows[0]
        assert first["url"] == "https://blog.example.com/post-1"
        assert first["title"] == "Momentum Works"
        assert first["source"] == "Alpha Blog"
        assert first["blurb"] == "A study of momentum."
        assert first["posted"] == "2026-08-09"
        assert first["stage"] == "harvested"
        assert first["slug"] == "momentum-works"

    def test_missing_source_and_date_are_none(self):
        row = harvest.parse_page(MASHUP_HTML)[1]
        assert row["title"] == "No Source Bracket Here"
        assert row["source"] is None
        assert row["posted"] is None

    def test_whitespace_in_title_is_normalised(self):
        rows = harvest.parse_page(
            '<article class="qo-entry"><a class="qo-title" href="https://x.com/a">'
            "  Spaced   Out  [Src] </a></article>"
        )
        assert rows[0]["title"] == "Spaced Out"
        assert rows[0]["source"] == "Src"


WRAP_HTML = """
<html><body>
<div class="entry-content">
  <a href="https://quantocracy.com/category/daily-wraps/">wrap nav link</a>
  <a href="https://blog.example.com/a">Vol Targeting [Fund Blog]</a>
  <a href="https://blog.example.com/b">Truncated Title [</a>
  <a href="https://blog.example.com/c"></a>
</div>
</body></html>
"""


class TestParseWrap:
    def test_parses_links_and_skips_self_links(self):
        rows = harvest.parse_wrap(WRAP_HTML, "2026-08-09")
        urls = [r["url"] for r in rows]
        # self-links and empty-text anchors are both dropped
        assert urls == [
            "https://blog.example.com/a",
            "https://blog.example.com/b",
        ]

    def test_bracketed_source_extracted(self):
        row = harvest.parse_wrap(WRAP_HTML, "2026-08-09")[0]
        assert row["title"] == "Vol Targeting"
        assert row["source"] == "Fund Blog"

    def test_truncated_bracket_falls_back_to_raw_title(self):
        row = harvest.parse_wrap(WRAP_HTML, "2026-08-09")[1]
        assert row["title"] == "Truncated Title"
        assert row["source"] is None

    def test_wrap_date_is_the_posted_date(self):
        rows = harvest.parse_wrap(WRAP_HTML, "2026-08-09")
        assert all(r["posted"] == "2026-08-09" for r in rows)
        assert all(r["posted_raw"] == "2026-08-09" for r in rows)

    def test_empty_title_is_dropped(self):
        rows = harvest.parse_wrap(WRAP_HTML, "2026-08-09")
        assert all(r["title"] for r in rows)

    def test_no_entry_content_returns_empty(self):
        assert harvest.parse_wrap("<html><body></body></html>", None) == []
