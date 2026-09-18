"""Tests for catalog.py — series search, resolution, and multi-provider loading."""
from __future__ import annotations

import pandas as pd
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


def _stub_read_csv(monkeypatch, frame_by_url):
    """Stub pandas.read_csv for URLs only; local cache reads pass through."""
    calls = []
    real = pd.read_csv

    def fake_read_csv(url_or_path, *a, **k):
        path = str(url_or_path)
        if not path.startswith("http"):
            return real(url_or_path, *a, **k)  # local cache file
        calls.append(path)
        for key, frame in frame_by_url.items():
            if key in path:
                return frame.copy()
        raise OSError(f"network down for {url_or_path}")

    monkeypatch.setattr(catalog.pd, "read_csv", fake_read_csv)
    return calls

def _stub_yahoo(monkeypatch, frame):
    def fake_download(ticker, **k):
        return frame.copy()

    monkeypatch.setattr(catalog, "_yahoo_download", fake_download)


class TestLoad:
    def test_fred_maps_value_to_close_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)

        class FakeFredReader:
            def __init__(self, series, start=None, end=None):
                assert series == "DGS10"

            def read(self):
                idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="DATE")
                return pd.DataFrame({"DGS10": ["4.0", "4.1"]}, index=idx)

        monkeypatch.setattr(catalog, "FredReader", FakeFredReader)

        df = catalog.load("fred:DGS10", None, None)

        assert list(df.columns) == ["close"]
        assert df["close"].iloc[0] == 4.0        # string -> float
        assert isinstance(df.index, pd.DatetimeIndex)
        assert (tmp_path / "fred_DGS10.csv").exists()  # cache written

    def test_cached_read_makes_no_network_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-02,4.0\n2024-01-03,4.1\n")
        calls = _stub_read_csv(monkeypatch, {})

        df = catalog.load("fred:DGS10", None, None)

        assert calls == []                        # zero network
        assert len(df) == 2

    def test_yahoo_provider(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        yf = pd.DataFrame({"Close": [100.0, 101.0],
                           "Open": [99.0, 100.0],
                           "High": [101.0, 102.0],
                           "Low": [98.0, 99.5],
                           "Volume": [1000, 1100]},
                          index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
        _stub_yahoo(monkeypatch, yf)

        df = catalog.load("yahoo:SPY", None, None)

        assert sorted(df.columns) == ["close", "high", "low", "open", "volume"]
        assert df["close"].iloc[1] == 101.0

    def test_fallback_used_when_primary_fails(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        # yahoo download raises -> QQQ falls back to stooq
        def boom(ticker, **k):
            raise OSError("yahoo down")

        monkeypatch.setattr(catalog, "_yahoo_download", boom)
        stooq = pd.DataFrame({"Date": ["2024-01-02"], "Open": [99.0],
                              "High": [101.0], "Low": [98.0], "Close": [100.0],
                              "Volume": [1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        calls = _stub_read_csv(monkeypatch, {url: stooq})

        df = catalog.load("yahoo:QQQ", None, None)

        assert calls and "stooq.com" in calls[0]
        assert df["close"].iloc[0] == 100.0
        assert "falling back" in capsys.readouterr().out.lower()

    def test_start_end_filters(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2020-01-01,1.0\n2024-01-01,2.0\n")

        df = catalog.load("fred:DGS10", "2023-01-01", None)

        assert len(df) == 1 and df["close"].iloc[0] == 2.0

    def test_unresolvable_series_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        with pytest.raises(ValueError, match="not in catalog"):
            catalog.load("fred:NOPE_NOPE", None, None)

    def test_empty_fetch_is_not_cached(self, tmp_path, monkeypatch):
        # a failed download must not poison the cache with a header-only file
        # (TLT: no fallback, so the failure surfaces instead of recursing)
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        _stub_yahoo(monkeypatch, pd.DataFrame())

        with pytest.raises(ValueError, match="returned no data"):
            catalog.load("yahoo:TLT", None, None)

        assert not (tmp_path / "yahoo_TLT.csv").exists()


class TestMain:
    def test_search_command_prints_hits(self, capsys):
        assert catalog.main_with(["search", "treasury"]) == 0
        out = capsys.readouterr().out
        assert "fred:DGS10" in out

    def test_search_no_match_exits_zero_with_message(self, capsys):
        assert catalog.main_with(["search", "zzzznotathing"]) == 0
        assert "no matches" in capsys.readouterr().out.lower()

    def test_show_prints_entry_details(self, capsys):
        assert catalog.main_with(["show", "fred:DGS10"]) == 0
        out = capsys.readouterr().out
        assert "fred:DGS10" in out and "daily" in out

    def test_show_unknown_series_fails_cleanly(self, capsys):
        with pytest.raises(SystemExit):
            catalog.main_with(["show", "fred:NOPE_NOPE"])

    def test_fetch_warms_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        seen = {}

        def fake_load(series_id, start, end):
            seen["id"], seen["start"], seen["end"] = series_id, start, end
            return pd.DataFrame({"close": [1.0]},
                                index=pd.DatetimeIndex(["2024-01-02"]))

        monkeypatch.setattr(catalog, "load", fake_load)
        assert catalog.main_with(["fetch", "fred:DGS10",
                                  "--start", "2000-01-01"]) == 0
        assert seen == {"id": "fred:DGS10", "start": "2000-01-01", "end": None}


class TestDataIntegrity:
    def test_stooq_newest_first_csv_comes_back_sorted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        stooq = pd.DataFrame({
            "Date": ["2024-01-03", "2024-01-02"],   # Stooq serves newest-first
            "Open": [100.0, 99.0], "High": [101.0, 100.0],
            "Low": [99.0, 98.0], "Close": [100.5, 100.0], "Volume": [1100, 1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        _stub_read_csv(monkeypatch, {url: stooq})

        df = catalog.load("stooq:qqq.us", None, None)

        assert df.index.is_monotonic_increasing
        assert df["close"].iloc[0] == 100.0   # the older bar first

    def test_header_only_cache_is_deleted_and_refetched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text("close\n")           # truncated/interrupted write

        class FakeFredReader:
            def __init__(self, series, start=None, end=None):
                pass

            def read(self):
                idx = pd.DatetimeIndex(["2024-01-02"])
                return pd.DataFrame({"DGS10": [4.0]}, index=idx)

        monkeypatch.setattr(catalog, "FredReader",
                            lambda *a, **k: FakeFredReader(*a))

        df = catalog.load("fred:DGS10", None, None)

        assert len(df) == 1
        assert len(cache.read_text().strip().splitlines()) == 2  # cache rewritten

    def test_unsorted_poisoned_cache_self_heals(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-03,4.1\n2024-01-02,4.0\n")  # descending
        _stub_read_csv(monkeypatch, {})  # network must NOT be needed

        df = catalog.load("fred:DGS10", None, None)

        assert df.index.is_monotonic_increasing

    def test_no_tmp_files_left_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-02,4.0\n")

        catalog.load("fred:DGS10", None, None)

        assert list(tmp_path.glob("*.tmp")) == []

    def test_fetch_refresh_forces_refetch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-02,4.0\n")

        class FakeFredReader:
            def __init__(self, series, start=None, end=None):
                pass

            def read(self):
                idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
                return pd.DataFrame({"DGS10": [4.0, 4.1]}, index=idx)

        fetched = []
        monkeypatch.setattr(catalog, "FredReader",
                            lambda *a, **k: (fetched.append(a), FakeFredReader(*a))[1])

        assert catalog.main_with(["fetch", "fred:DGS10", "--refresh"]) == 0

        assert fetched                        # --refresh bypassed the warm cache
        assert len(pd.read_csv(cache)) == 2


class TestProvenance:
    def test_primary_fetch_records_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(catalog, "PROVENANCE", {})
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        fred = pd.DataFrame({"observation_date": ["2024-01-02"], "DGS10": ["4.0"]})
        _stub_read_csv(monkeypatch, {url: fred})

        catalog.load("fred:DGS10", None, None)

        assert catalog.PROVENANCE["fred:DGS10"] == "fred"

    def test_fallback_records_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(catalog, "PROVENANCE", {})

        def boom(ticker, **k):
            raise OSError("yahoo down")

        monkeypatch.setattr(catalog, "_yahoo_download", boom)
        stooq = pd.DataFrame({"Date": ["2024-01-02"], "Open": [99.0],
                              "High": [101.0], "Low": [98.0], "Close": [100.0],
                              "Volume": [1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        _stub_read_csv(monkeypatch, {url: stooq})

        catalog.load("yahoo:QQQ", None, None)

        assert catalog.PROVENANCE["yahoo:QQQ"] == "fallback:stooq:qqq.us"

    def test_spy_declares_stooq_fallback(self):
        entry = catalog.resolve("yahoo:SPY")
        assert entry["fallback"] == "stooq:spy.us"   # the index's own note says it is SPY's mirror

    def test_provenance_flags_warn_on_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(catalog, "PROVENANCE", {})

        def boom(ticker, **k):
            raise OSError("yahoo down")

        monkeypatch.setattr(catalog, "_yahoo_download", boom)
        stooq = pd.DataFrame({"Date": ["2024-01-02"], "Open": [99.0],
                              "High": [101.0], "Low": [98.0], "Close": [100.0],
                              "Volume": [1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        _stub_read_csv(monkeypatch, {url: stooq})
        catalog.load("yahoo:QQQ", None, None)

        flags = catalog.provenance_flags(["yahoo:QQQ", "yahoo:SPY"])

        assert len(flags) == 1
        assert "fallback stooq:qqq.us" in flags[0]


class TestFrequency:
    def test_monthly_series_warns_once_on_load(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        catalog._WARNED.clear()
        cache = tmp_path / "fred_CPIAUCSL.csv"
        cache.write_text(",close\n2024-01-01,300.0\n2024-02-01,301.0\n")

        catalog.load("fred:CPIAUCSL", None, None)
        first = capsys.readouterr().out
        catalog.load("fred:CPIAUCSL", None, None)
        second = capsys.readouterr().out

        assert "WARNING" in first and "monthly" in first
        assert "WARNING" not in second          # once per series, not per call

    def test_daily_series_does_not_warn(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        catalog._WARNED.clear()
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-02,4.0\n")

        catalog.load("fred:DGS10", None, None)

        assert "WARNING" not in capsys.readouterr().out

    def test_frequency_flags_catch_monthly_in_daily_panel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        flags = catalog.frequency_flags(["SPY", "fred:CPIAUCSL"], "1d")
        assert len(flags) == 1
        assert "fred:CPIAUCSL" in flags[0] and "monthly" in flags[0]

    def test_frequency_flags_pass_matching_bar(self, tmp_path, monkeypatch):
        flags = catalog.frequency_flags(["SPY"], "1d")
        assert flags == []


class TestUnseededYahoo:
    def test_unseeded_yahoo_id_fetches_like_bare_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(catalog, "PROVENANCE", {})
        yf = pd.DataFrame({"Close": [250.0], "Open": [249.0], "High": [251.0],
                           "Low": [248.0], "Volume": [900]},
                          index=pd.DatetimeIndex(["2024-01-02"]))
        _stub_yahoo(monkeypatch, yf)

        df = catalog.load("yahoo:AAPL", None, None)   # not in catalog.json

        assert df["close"].iloc[0] == 250.0
        assert (tmp_path / "yahoo_AAPL.csv").exists()

    def test_unseeded_fred_id_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        with pytest.raises(ValueError, match="not in catalog"):
            catalog.load("fred:NOPE_NOPE", None, None)

    def test_unknown_source_still_raises(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="unknown source"):
            catalog.load("bogus:XYZ", None, None)


class TestMinors:
    def test_multi_word_search_requires_all_terms(self):
        hits = catalog.search("treasury yield")
        ids = {h["id"] for h in hits}
        assert "fred:DGS10" in ids and "fred:DGS2" in ids
        assert "yahoo:TLT" not in ids          # has treasury, lacks yield

    def test_normalize_yahoo_dedupes_legacy_block(self):
        yf = pd.DataFrame({"Close": [100.0], "Open": [99.0], "High": [101.0],
                           "Low": [98.0], "Volume": [1000]},
                          index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"))

        df = catalog.normalize_yahoo(yf)

        assert sorted(df.columns) == ["close", "high", "low", "open", "volume"]
        assert df.index.tz is None

    def test_every_entry_has_a_license(self):
        for entry in catalog._index()["series"]:
            assert entry.get("license"), f"{entry['id']} missing license"
