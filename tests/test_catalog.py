"""Tests for catalog.py — series search, resolution, and multi-provider loading."""
from __future__ import annotations

import pytest

import catalog


class TestSearch:
    def test_match_on_id_substring(self):
        hits = catalog.search("DGS10")
        assert len(hits) == 1
        assert hits[0]["id"] == "fred:DGS10"

    def test_match_on_title_case_insensitive(self):
        hits = catalog.search("treasury")
        assert all("treasury" in (h["title"] + " " + h["id"]).lower() for h in hits)
        assert hits  # the seed has at least one treasury series

    def test_match_on_tag(self):
        hits = catalog.search("rates")
        assert any(h["id"] == "fred:DGS10" for h in hits)

    def test_no_match_returns_empty(self):
        assert catalog.search("definitely-not-a-series-xyz") == []


class TestParseId:
    def test_prefixed_id_splits(self):
        assert catalog.parse_id("fred:DGS10") == ("fred", "DGS10")

    def test_bare_id_defaults_to_yahoo(self):
        assert catalog.parse_id("SPY") == ("yahoo", "SPY")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="unknown source"):
            catalog.parse_id("bogus:XYZ")

    def test_resolve_unknown_series_raises(self):
        with pytest.raises(ValueError, match="not in catalog"):
            catalog.resolve("fred:NOPE_NOPE")
