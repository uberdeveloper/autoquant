# Data Catalog with Cross-Source Search (Issue #4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A committed, machine-readable catalog of data series across providers (yahoo, fred, stooq) with one search interface, cached loading with Yahoo fallback, and spec/backtest integration via `source:id` universe entries.

**Architecture:** New module `catalog.py` owns three things: the seed index (`catalog.json`, committed, flat like `_TEMPLATE.yaml`), a pure search/resolve layer over it, and `load()` which dispatches to a per-provider fetch function and caches under `data/catalog/`. Provider access uses direct CSV endpoints (FRED `fredgraph.csv`, Stooq CSV) — no `pandas-datareader` dependency (the issue flags its maintenance status as an open concern; the provider registry makes it a one-function addition later). `backtest.load_prices` dispatches any universe entry containing `:` to the catalog, so `fred:DGS10` and `yahoo:SPY` work beside plain legacy tickers. The harness contract is untouched: every provider normalizes to a lowercase-column OHLCV-shaped frame with a `close` column.

**Tech Stack:** Python 3.11 stdlib (`urllib` via `pandas.read_csv(url)`), `yfinance` (existing dep), pytest via `uv run pytest`. All network access is stubbed in tests.

**Repo:** `/Users/nehapriya/Desktop/autoquant`, branch `issue4-data-catalog` off `main`.

**Open-question resolutions (from issue #4):**
1. pandas-datareader maintenance risk -> NOT used in v1; direct CSV endpoints instead. Provider registry keeps the door open.
2. Point-in-time coverage -> each entry records `vintage_available` (ALFRED has vintages for FRED macro series) and the honest `point_in_time: false` note that loaded data is current-vintage. Wiring ALFRED vintages is explicitly future work.
3. Search backend -> local curated seed list; live provider search APIs are future work.

**Key existing facts the implementer must know:**

- `backtest.load_prices` (lines ~38-56) is the single data entry point: cache-first (`data/prices/<ticker>.csv`), then `yfinance`. `load_panel` builds on it. The catalog hooks into `load_prices` only — panels inherit it.
- Specs reference `data.universe` as bare tickers today (`[SPY]`); bare tickers must keep working (legacy path unchanged).
- Commit style: lowercase `area: summary`. TDD throughout. `uv run pytest` is the test runner.

---

### Task 1: Seed index + search/resolve

**Files:**
- Create: `catalog.json`
- Create: `catalog.py` (search/resolve only in this task)
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_catalog.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'catalog'`

- [ ] **Step 3: Create the seed index `catalog.json`**

```json
{
  "version": 1,
  "series": [
    {"id": "yahoo:SPY", "title": "S&P 500 ETF (SPY)", "tags": ["equity", "etf", "us"],
     "frequency": "daily", "coverage_start": "1993-01-29", "vintage_available": false,
     "notes": "Split/dividend adjusted via yfinance auto_adjust.", "fallback": null},
    {"id": "yahoo:QQQ", "title": "Nasdaq-100 ETF (QQQ)", "tags": ["equity", "etf", "nasdaq"],
     "frequency": "daily", "coverage_start": "1999-03-10", "vintage_available": false,
     "notes": "Split/dividend adjusted.", "fallback": "stooq:qqq.us"},
    {"id": "yahoo:TLT", "title": "20+ Year Treasury Bond ETF (TLT)", "tags": ["rates", "etf", "treasury"],
     "frequency": "daily", "coverage_start": "2002-07-22", "vintage_available": false,
     "notes": null, "fallback": null},
    {"id": "yahoo:GLD", "title": "Gold Shares ETF (GLD)", "tags": ["commodity", "gold", "etf"],
     "frequency": "daily", "coverage_start": "2004-11-18", "vintage_available": false,
     "notes": null, "fallback": "stooq:gld.us"},
    {"id": "yahoo:^VIX", "title": "CBOE Volatility Index (VIX)", "tags": ["volatility", "index"],
     "frequency": "daily", "coverage_start": "1990-01-02", "vintage_available": false,
     "notes": "Index level, not tradable.", "fallback": null},

    {"id": "fred:DGS10", "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
     "tags": ["rates", "treasury", "yield"], "frequency": "daily",
     "coverage_start": "1962-01-02", "vintage_available": true,
     "notes": "Loaded series is current vintage; ALFRED vintages not wired (future work).",
     "fallback": null},
    {"id": "fred:DGS2", "title": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
     "tags": ["rates", "treasury", "yield"], "frequency": "daily",
     "coverage_start": "1976-06-01", "vintage_available": true, "notes": null, "fallback": null},
    {"id": "fred:DGS3MO", "title": "Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity",
     "tags": ["rates", "treasury", "tbill"], "frequency": "daily",
     "coverage_start": "1982-01-04", "vintage_available": true, "notes": null, "fallback": null},
    {"id": "fred:T10Y2Y", "title": "10-Year Treasury Minus 2-Year Treasury (term spread)",
     "tags": ["rates", "curve", "spread"], "frequency": "daily",
     "coverage_start": "1976-06-01", "vintage_available": true, "notes": null, "fallback": null},
    {"id": "fred:CPIAUCSL", "title": "Consumer Price Index for All Urban Consumers (CPI, SA)",
     "tags": ["macro", "inflation", "cpi"], "frequency": "monthly",
     "coverage_start": "1947-01-01", "vintage_available": true,
     "notes": "Monthly; release lag ~2 weeks. Use pct_change(12) for YoY.",
     "fallback": null},
    {"id": "fred:UNRATE", "title": "Unemployment Rate (SA)",
     "tags": ["macro", "labor", "unemployment"], "frequency": "monthly",
     "coverage_start": "1948-01-01", "vintage_available": true,
     "notes": "Monthly; release lag ~5 weeks.", "fallback": null},
    {"id": "fred:FEDFUNDS", "title": "Effective Federal Funds Rate",
     "tags": ["rates", "macro", "fed"], "frequency": "monthly",
     "coverage_start": "1954-07-01", "vintage_available": true, "notes": null, "fallback": null},
    {"id": "fred:VIXCLS", "title": "VIX Close (CBOE Volatility Index, daily close)",
     "tags": ["volatility", "index"], "frequency": "daily",
     "coverage_start": "1990-01-02", "vintage_available": true,
     "notes": "Official CBOE close, vs yahoo:^VIX intraday proxy.", "fallback": null},
    {"id": "fred:T10YIE", "title": "10-Year Breakeven Inflation Rate",
     "tags": ["inflation", "breakeven", "rates"], "frequency": "daily",
     "coverage_start": "2003-01-02", "vintage_available": true, "notes": null, "fallback": null},

    {"id": "stooq:spy.us", "title": "S&P 500 ETF (SPY) via Stooq", "tags": ["equity", "etf"],
     "frequency": "daily", "coverage_start": "1993-01-29", "vintage_available": false,
     "notes": "Price-only (unadjusted) history; a fallback mirror for yahoo:SPY.",
     "fallback": null}
  ]
}
```

- [ ] **Step 4: Implement search/resolve in `catalog.py`**

```python
"""Data catalog — one search interface over multiple data providers.

Specs reference series as `source:id` (e.g. `fred:DGS10`, `yahoo:SPY`);
a bare id defaults to `yahoo:`. search() finds entries in the committed
seed index (catalog.json) by id, title, or tag. load() fetches through a
per-provider fetch function, caches under data/catalog/, and falls back to
an entry's declared fallback (usually a yahoo/stooq mirror) if the primary
provider fails.

Providers are plain functions registered in PROVIDERS — adding one (World
Bank, OECD, a pandas-datareader adapter) is one function, no call-site
changes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "catalog.json"
CACHE_DIR = ROOT / "data" / "catalog"


def _index() -> dict:
    return json.loads(INDEX.read_text())


def by_id(series_id: str) -> dict | None:
    for entry in _index()["series"]:
        if entry["id"] == series_id:
            return entry
    return None


def search(query: str) -> list[dict]:
    """Case-insensitive substring match on id, title, and tags."""
    q = query.lower()
    out = []
    for entry in _index()["series"]:
        hay = " ".join([entry["id"], entry["title"],
                        " ".join(entry["tags"])]).lower()
        if q in hay:
            out.append(entry)
    return out


SOURCES = ("yahoo", "fred", "stooq")


def parse_id(series_id: str) -> tuple[str, str]:
    """'fred:DGS10' -> ('fred', 'DGS10'); bare ids default to yahoo."""
    if ":" in series_id:
        source, _, rest = series_id.partition(":")
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r} in {series_id!r} "
                             f"(known: {', '.join(SOURCES)})")
        return source, rest
    return "yahoo", series_id


def resolve(series_id: str) -> dict:
    entry = by_id(series_id if ":" in series_id else f"yahoo:{series_id}")
    if entry is None:
        raise ValueError(f"{series_id!r} not in catalog "
                         f"(try: python3 catalog.py search <query>)")
    return entry
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: all PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
git add catalog.json catalog.py tests/test_catalog.py
git commit -m "catalog: seed index with search and id resolution"
```

---

### Task 2: Providers + `load()` with cache and Yahoo fallback

**Files:**
- Modify: `catalog.py`
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
class TestLoad:
    def _stub_read_csv(self, monkeypatch, frame_by_url):
        """Stub pandas.read_csv keyed on URL substring; record calls."""
        calls = []

        def fake_read_csv(url_or_path, *a, **k):
            calls.append(str(url_or_path))
            for key, frame in frame_by_url.items():
                if key in str(url_or_path):
                    return frame.copy()
            raise OSError(f"network down for {url_or_path}")

        monkeypatch.setattr(catalog.pd, "read_csv", fake_read_csv)
        return calls

    def _stub_yahoo(self, monkeypatch, frame):
        def fake_download(ticker, **k):
            return frame.copy()

        monkeypatch.setattr(catalog, "_yahoo_download", fake_download)

    def test_fred_maps_value_to_close_and_caches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        fred = pd.DataFrame({"observation_date": ["2024-01-02", "2024-01-03"],
                             "DGS10": ["4.0", "4.1"]})
        self._stub_read_csv(monkeypatch, {url: fred})

        df = catalog.load("fred:DGS10", None, None)

        assert list(df.columns) == ["close"]
        assert df["close"].iloc[0] == 4.0        # string -> float
        assert df.index.is_all_dates
        assert (tmp_path / "fred_DGS10.csv").exists()  # cache written

    def test_cached_read_makes_no_network_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-02,4.0\n2024-01-03,4.1\n")
        calls = self._stub_read_csv(monkeypatch, {})

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
        self._stub_yahoo(monkeypatch, yf)

        df = catalog.load("yahoo:SPY", None, None)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
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
        calls = self._stub_read_csv(monkeypatch, {url: stooq})

        df = catalog.load("yahoo:QQQ", None, None)

        assert calls and "stooq.com" in calls[0]
        assert df["close"].iloc[0] == 100.0
        assert "fallback" in capsys.readouterr().out.lower()

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
```

Add `import pandas as pd` to the test module imports (after `import catalog`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: FAIL — `AttributeError: module 'catalog' has no attribute 'load'`

- [ ] **Step 3: Implement providers and `load()`**

Append to `catalog.py`:

```python
# ---------------------------------------------------------------- providers

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
STOOQ_URL = "https://stooq.com/q/d/l/?s={id}&i=d"


def _yahoo_download(ticker: str, **kwargs) -> pd.DataFrame:
    """Isolated so tests (and future providers) can stub it."""
    import yfinance

    return yfinance.download(ticker, **kwargs)


def _fetch_yahoo(symbol: str) -> pd.DataFrame:
    df = _yahoo_download(symbol, start="1990-01-01", auto_adjust=True,
                         progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _fetch_fred(series: str) -> pd.DataFrame:
    raw = pd.read_csv(FRED_URL.format(id=series))
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    df = (raw.set_index(date_col)
             .rename(columns={series: "close"})
             .apply(pd.to_numeric, errors="coerce").dropna())
    df.index.name = None
    return df[["close"]]


def _fetch_stooq(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(STOOQ_URL.format(id=symbol))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw.set_index("date").apply(pd.to_numeric, errors="coerce").dropna()
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]]


PROVIDERS = {"yahoo": _fetch_yahoo, "fred": _fetch_fred, "stooq": _fetch_stooq}

# ---------------------------------------------------------------- loading

def load(series_id: str, start, end) -> pd.DataFrame:
    """One series, cached, fallback-aware. Normalized frame out:
    lowercase columns, always with `close`, DatetimeIndex, filtered to
    [start, end]."""
    entry = resolve(series_id)
    source, symbol = parse_id(entry["id"])
    cache = CACHE_DIR / f"{source}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}.csv"

    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        try:
            df = PROVIDERS[source](symbol)
        except Exception as exc:
            if not entry.get("fallback"):
                raise
            print(f"{entry['id']}: {exc} -- falling back to {entry['fallback']}")
            return load(entry["fallback"], start, end)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: all PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add catalog.py tests/test_catalog.py
git commit -m "catalog: provider fetchers, cached load, yahoo fallback"
```

---

### Task 3: CLI — `search`, `show`, `fetch`

**Files:**
- Modify: `catalog.py`
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog.py`:

```python
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
            return pd.DataFrame({"close": [1.0]})

        monkeypatch.setattr(catalog, "load", fake_load)
        assert catalog.main_with(["fetch", "fred:DGS10",
                                  "--start", "2000-01-01"]) == 0
        assert seen == {"id": "fred:DGS10", "start": "2000-01-01", "end": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestMain -q`
Expected: FAIL — `AttributeError: module 'catalog' has no attribute 'main_with'`

- [ ] **Step 3: Implement the CLI**

Append to `catalog.py`:

```python
# ---------------------------------------------------------------- cli

def main_with(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="search the catalog")
    p_search.add_argument("query")

    p_show = sub.add_parser("show", help="show one entry's metadata")
    p_show.add_argument("series_id")

    p_fetch = sub.add_parser("fetch", help="download a series into the local cache")
    p_fetch.add_argument("series_id")
    p_fetch.add_argument("--start", default=None)
    p_fetch.add_argument("--end", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "search":
        hits = search(args.query)
        if not hits:
            print(f"no matches for {args.query!r}")
            return 0
        for e in hits:
            print(f"{e['id']:<18} {e['frequency']:<8} {e['title']}")
        return 0

    if args.cmd == "show":
        e = resolve(args.series_id)   # raises ValueError -> clean exit below
        for key in ("id", "title", "tags", "frequency", "coverage_start",
                    "vintage_available", "notes", "fallback"):
            print(f"{key:>17}: {e.get(key)}")
        return 0

    if args.cmd == "fetch":
        df = load(args.series_id, args.start, args.end)
        print(f"{args.series_id}: {len(df)} bars cached"
              f"{f' ({df.index[0].date()} -> {df.index[-1].date()})' if len(df) else ''}")
        return 0
    return 1


def main() -> int:
    try:
        return main_with()
    except ValueError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
```

(The `import argparse` inside `main_with` avoids touching the module imports; move it to the top import block if preferred.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: all PASS (19 tests)

- [ ] **Step 5: Smoke the CLI**

Run: `uv run python catalog.py search treasury && uv run python catalog.py show fred:DGS10`
Expected: search lists the FRED treasury entries; show prints DGS10 metadata.

- [ ] **Step 6: Commit**

```bash
git add catalog.py tests/test_catalog.py
git commit -m "catalog: search/show/fetch CLI"
```

---

### Task 4: Backtest integration — `source:id` universes

**Files:**
- Modify: `backtest.py` (`load_prices`, ~lines 38-56)
- Modify: `_TEMPLATE.yaml` (data section comment)
- Test: `tests/test_backtest.py` (new class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backtest.py` (append at end; the file already imports `backtest`, `pandas as pd`, and uses `monkeypatch`/`tmp_path`):

```python
class TestCatalogUniverse:
    def test_load_prices_dispatches_prefixed_ids_to_catalog(self, tmp_path, monkeypatch):
        seen = {}

        def fake_load(series_id, start, end):
            seen["id"], seen["start"], seen["end"] = series_id, start, end
            idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
            return pd.DataFrame({"close": [100.0, 101.0]}, index=idx)

        monkeypatch.setattr(backtest.catalog, "load", fake_load)

        df = backtest.load_prices("fred:DGS10", "2024-01-01", None,
                                  tmp_path / "cache")

        assert seen == {"id": "fred:DGS10", "start": "2024-01-01", "end": None}
        assert list(df.columns) == ["close"]

    def test_bare_ticker_still_uses_yfinance_path(self, tmp_path, monkeypatch):
        # a cached data/prices CSV short-circuits before any network
        cache = tmp_path / "SPY.csv"
        cache.write_text(",close\n2024-01-02,100.0\n")

        df = backtest.load_prices("SPY", None, None, tmp_path)

        assert len(df) == 1  # legacy path untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py::TestCatalogUniverse -q`
Expected: FAIL — `AttributeError: module 'backtest' has no attribute 'catalog'`

- [ ] **Step 3: Wire the dispatch into `backtest.py`**

Add `import catalog` to the imports (after `import argparse`/stdlib block, before third-party `numpy`/`pandas`/`yaml` — it is a first-party module; match extract.py's `from triage import ...` placement style). Then replace `load_prices` with:

```python
def load_prices(ticker: str, start, end, cache_dir: Path) -> pd.DataFrame:
    # catalog ids ("fred:DGS10", "yahoo:SPY", ...) go through the catalog;
    # bare tickers keep the legacy yfinance path
    if ":" in ticker:
        return catalog.load(ticker, start, end)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker.replace('/', '_')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df
```

- [ ] **Step 4: Update `_TEMPLATE.yaml` data-section comment**

Replace line 26:

```yaml
  universe: [SPY]                      # explicit tickers, or a rule + point-in-time source
```

with:

```yaml
  universe: [SPY]                      # tickers, or catalog ids: fred:DGS10, yahoo:SPY
                                       # (find them: python3 catalog.py search <query>)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add backtest.py _TEMPLATE.yaml tests/test_backtest.py
git commit -m "backtest: route source:id universe entries through the data catalog"
```

---

### Task 5: README + full validation + live smoke + PR

**Files:**
- Modify: `README.md` (pipeline command list + a short data section)

- [ ] **Step 1: Document in README.md**

After the stage command list (the block containing `python3 codegen.py` etc.), add one line to the code block:

```bash
python3 catalog.py search rates    # find series across providers (yahoo/fred/stooq)
```

and after the "Swapping the LLM provider" section, add:

```markdown
### Data catalog

Universe entries can be bare Yahoo tickers (`SPY`) or catalog ids:
`yahoo:SPY`, `fred:DGS10`, `stooq:spy.us`. Find series with
`python3 catalog.py search <query>`, inspect one with
`python3 catalog.py show <id>`, and pre-warm the cache with
`python3 catalog.py fetch <id>`. The committed seed index is
`catalog.json`; series cache under `data/catalog/`. If a primary provider
fails, an entry's declared fallback (usually a Yahoo/Stooq mirror) is used
automatically. Macro entries note their release lag and whether vintage
(ALFRED) data exists — loading current-vintage data for backtests has
lookahead caveats the spec's `data.caveats` should state.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: all PASS (~217 tests; was 197)

- [ ] **Step 3: Live smoke with a real FRED fetch**

Run: `uv run python catalog.py fetch fred:DGS10 --start 2020-01-01 && uv run python catalog.py search treasury`
Expected: ~1700 daily bars cached; search lists treasury entries. (If the network is unavailable, record that and rely on the stubbed tests.)

- [ ] **Step 4: Commit docs and push**

```bash
git add README.md
git commit -m "docs: data catalog usage"
git push -u fork issue4-data-catalog
```

- [ ] **Step 5: Open the PR**

`gh pr create` against `uberdeveloper/autoquant` `main` from `neha-priyaa:issue4-data-catalog`, title "data catalog: cross-source search, multi-provider loading, source:id universes", body summarizing: seed index + search/resolve/fetch CLI, providers (yahoo/fred/stooq, no pandas-datareader — rationale), cached load with fallback, `source:id` universes in specs, open-question resolutions, test plan.

---

## Self-review notes

- **Spec coverage:** search layer (Task 1), systematic multi-source access (Task 2 — direct CSV instead of pandas-datareader, resolution documented), provider fallback with yahoo (Task 2), machine-readable catalog index with coverage/frequency/vintage/license-notes fields (Task 1, `catalog.json`), spec/backtest integration via catalog ids in `data.universe` with validation via `resolve` errors (Task 4). Non-goals respected: no broker/live trading, no paid sources, no DSL.
- **Placeholder scan:** none — every step carries full code.
- **Type consistency:** `load(series_id: str, start, end) -> pd.DataFrame` used by the CLI, tests, and `backtest.load_prices` identically; `parse_id -> tuple[str, str]` and `resolve -> dict` consumed by `load` and the CLI. FRED frames are `close`-only by design — the harness only requires `close` (and `open` for next_open execution, which price-like FRED specs must not use; noted in the README caveats paragraph).
- **Determinism:** no network in tests — yahoo stubbed via `_yahoo_download`, CSV endpoints via `pd.read_csv` monkeypatch, cache-first ordering exercised explicitly.
