# Issue #10 Catalog Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every review finding from PR #9 (issue #10): silent data-integrity bugs (unsorted Stooq, unvalidated caches, invisible fallback substitution), backtest correctness with catalog series (release-lag lookahead, mixed frequencies, OHLCV contract), feature completion (extract.py rejects catalog ids, unseeded yahoo hard-fail, friendly errors), the pandas-datareader design re-open, and the minors.

**Architecture:** All catalog fixes stay in `catalog.py`/`catalog.json`; backtest-level correctness lands as guards in `backtest.run()`/`backtest.load_prices` and as *flags* (`auto_flags`) so the report and leaderboard carry the warnings without new plumbing. Provenance lives in a module-level `catalog.PROVENANCE` dict populated by `load()` and surfaced via `catalog.provenance_flags()`. FRED moves to `pandas-datareader` (verified live: pdr 0.11.1 + pandas 3.0.5, `FredReader('DGS10').read()` works); Yahoo stays yfinance; Stooq stays direct CSV (datareader has no Stooq reader).

**Tech Stack:** Python 3.11, pandas/pandas-datareader, yfinance, pytest via `uv run pytest`. All network stubbed in tests.

**Repo:** `/Users/nehapriya/Desktop/autoquant`, branch `issue10-catalog-hardening` off `main`.

**Issue-section -> task map:**

| Issue section | Task |
|---|---|
| §1a unsorted Stooq | 1 |
| §1b cache validation, atomic writes, `fetch --refresh` | 1 |
| §1c fallback provenance | 2 |
| §1d SPY fallback field | 2 |
| §2a release-lag lookahead warning | 3 |
| §2b frequency vs data.bar check | 3 |
| §2c OHLCV contract (run guard + source-shaped smoke) | 4 |
| §3 unseeded yahoo + friendly errors | 5 |
| §3 extract.py rejects catalog ids | 6 |
| §4 pandas-datareader for macro | 7 |
| §5a dedupe normalization / §5b license field / §5c multi-word search | 8 |
| docs + validation + PR | 9 |

**Key existing facts:**

- `catalog.load` today: resolve -> parse_id -> cache filename `f"{source}_{re.sub(...)}"` -> cache hit = plain `pd.read_csv` (no validation) -> miss = provider (empty raises) -> fallback recurse -> non-atomic `df.to_csv(cache)` -> start/end filter.
- `backtest.run()` builds `out["auto_flags"] = auto_flags(spec, out)` at backtest.py:333 — new flags append right after.
- `backtest.load_prices` (backtest.py:39) dispatches `":" in ticker` to `catalog.load`; bare tickers keep the yfinance block (backtest.py:49-55) — that block is byte-identical to `catalog._fetch_yahoo`'s normalization (§5a).
- `extract.validate_spec` (extract.py:115-122) requires every universe entry to match `[A-Z0-9\-\.\^=]{1,12}` — rejects `fred:DGS10`.
- `codegen.smoke_frame()` always fabricates all five OHLCV columns (codegen.py:89) and `publish` uses it for every spec (codegen.py:198-200).
- Tests: 209 passing on main. `uv run pytest` is the runner. Commit style: lowercase `area: summary`.

---

### Task 1: Stooq sort + cache validation + atomic writes + `fetch --refresh`

**Files:**
- Modify: `catalog.py` (`_fetch_stooq`, `load`, new `cache_path`/`_write_cache`)
- Modify: `catalog.py` CLI (`fetch --refresh`)
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
class TestDataIntegrity:
    def test_stooq_newest_first_csv_comes_back_sorted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        stooq = pd.DataFrame({
            "Date": ["2024-01-03", "2024-01-02"],   # Stooq serves newest-first
            "Open": [100.0, 99.0], "High": [101.0, 100.0],
            "Low": [99.0, 98.0], "Close": [100.5, 100.0], "Volume": [1100, 1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        self._stub_read_csv(monkeypatch, {url: stooq})

        df = catalog.load("stooq:qqq.us", None, None)

        assert df.index.is_monotonic_increasing
        assert df["close"].iloc[0] == 100.0   # the older bar first

    def test_header_only_cache_is_deleted_and_refetched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text("close\n")           # truncated/interrupted write
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        fred = pd.DataFrame({"observation_date": ["2024-01-02"], "DGS10": ["4.0"]})
        calls = self._stub_read_csv(monkeypatch, {url: fred})

        df = catalog.load("fred:DGS10", None, None)

        assert calls                          # network was hit again
        assert len(df) == 1
        assert len(cache.read_text().strip().splitlines()) == 2  # cache rewritten

    def test_unsorted_poisoned_cache_self_heals(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        cache = tmp_path / "fred_DGS10.csv"
        cache.write_text(",close\n2024-01-03,4.1\n2024-01-02,4.0\n")  # descending
        self._stub_read_csv(monkeypatch, {})  # network must NOT be needed

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
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        fred = pd.DataFrame({"observation_date": ["2024-01-02", "2024-01-03"],
                             "DGS10": ["4.0", "4.1"]})
        calls = self._stub_read_csv(monkeypatch, {url: fred})

        assert catalog.main_with(["fetch", "fred:DGS10", "--refresh"]) == 0

        assert calls                          # --refresh bypassed the warm cache
        assert len(pd.read_csv(cache)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestDataIntegrity -q`
Expected: FAIL — sorted/unsorted asserts fail (Stooq frame is cached descending; poisoned cache returned as-is), `--refresh` is an unrecognized argument.

- [ ] **Step 3: Implement**

In `catalog.py`, make `_fetch_stooq` return a sorted frame:

```python
def _fetch_stooq(symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(STOOQ_URL.format(id=symbol))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw.set_index("date").apply(pd.to_numeric, errors="coerce").dropna()
    df.index.name = None
    return df.sort_index()[["open", "high", "low", "close", "volume"]]
```

Add a cache-name helper and an atomic writer next to `load`:

```python
def cache_path(series_id: str) -> Path:
    source, symbol = parse_id(series_id)
    return CACHE_DIR / f"{source}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}.csv"


def _write_cache(cache: Path, df: pd.DataFrame) -> None:
    """Tmp + replace: an interrupted write can never shadow real data."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".csv.tmp")
    df.to_csv(tmp)
    tmp.replace(cache)
```

Replace `load` with:

```python
def load(series_id: str, start, end) -> pd.DataFrame:
    """One series, cached, fallback-aware. Normalized frame out:
    lowercase columns, always with `close`, ascending DatetimeIndex,
    filtered to [start, end]."""
    entry = resolve(series_id)
    source, symbol = parse_id(entry["id"])
    cache = cache_path(entry["id"])

    df = None
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns:
            cache.unlink()          # truncated/interrupted write -- refetch
            df = None
        else:
            df = df.sort_index()    # self-heal a poisoned (unsorted) cache
    if df is None:
        try:
            df = PROVIDERS[source](symbol)
            if df.empty:
                raise ValueError("returned no data")
        except Exception as exc:
            if not entry.get("fallback"):
                raise ValueError(f"{entry['id']}: {exc}") from exc
            print(f"{entry['id']}: {exc} -- falling back to {entry['fallback']}")
            return load(entry["fallback"], start, end)
        _write_cache(cache, df)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df
```

In `main_with`, add the flag and the unlink:

```python
    p_fetch.add_argument("--refresh", action="store_true",
                         help="delete any cached copy first and refetch")
```

and at the top of the `fetch` branch:

```python
    if args.cmd == "fetch":
        if args.refresh:
            cache_path(args.series_id).unlink(missing_ok=True)
        df = load(args.series_id, args.start, args.end)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: all PASS (25 tests; the existing 20 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add catalog.py tests/test_catalog.py
git commit -m "catalog: sort stooq frames, validate caches, atomic writes, fetch --refresh"
```

---

### Task 2: Fallback provenance + SPY fallback field

**Files:**
- Modify: `catalog.py` (`PROVENANCE`, `provenance()`)
- Modify: `catalog.json` (yahoo:SPY fallback)
- Modify: `backtest.py` (wire provenance flags into `run()`)
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py` (inside/after `TestLoad`; it reuses `_stub_read_csv`/`_stub_yahoo`, so add as a new class with its own stubs):

```python
class TestProvenance:
    def test_primary_fetch_records_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
        fred = pd.DataFrame({"observation_date": ["2024-01-02"], "DGS10": ["4.0"]})
        self._stub_read_csv(monkeypatch, {url: fred})

        catalog.load("fred:DGS10", None, None)

        assert catalog.PROVENANCE["fred:DGS10"] == "fred"

    def test_fallback_records_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)

        def boom(ticker, **k):
            raise OSError("yahoo down")

        monkeypatch.setattr(catalog, "_yahoo_download", boom)
        stooq = pd.DataFrame({"Date": ["2024-01-02"], "Open": [99.0],
                              "High": [101.0], "Low": [98.0], "Close": [100.0],
                              "Volume": [1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        self._stub_read_csv(monkeypatch, {url: stooq})

        catalog.load("yahoo:QQQ", None, None)

        assert catalog.PROVENANCE["yahoo:QQQ"] == "fallback:stooq:qqq.us"

    def test_spy_declares_stooq_fallback(self):
        entry = catalog.resolve("yahoo:SPY")
        assert entry["fallback"] == "stooq:spy.us"   # the index's own note says it is SPY's mirror

    def test_provenance_flags_warn_on_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)

        def boom(ticker, **k):
            raise OSError("yahoo down")

        monkeypatch.setattr(catalog, "_yahoo_download", boom)
        stooq = pd.DataFrame({"Date": ["2024-01-02"], "Open": [99.0],
                              "High": [101.0], "Low": [98.0], "Close": [100.0],
                              "Volume": [1000]})
        url = "https://stooq.com/q/d/l/?s=qqq.us&i=d"
        self._stub_read_csv(monkeypatch, {url: stooq})
        catalog.load("yahoo:QQQ", None, None)

        flags = catalog.provenance_flags(["yahoo:QQQ", "yahoo:SPY"])

        assert len(flags) == 1
        assert "fallback stooq:qqq.us" in flags[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestProvenance -q`
Expected: FAIL — `module 'catalog' has no attribute 'PROVENANCE'` (and SPY fallback is null).

- [ ] **Step 3: Implement**

In `catalog.py`, add after the `SOURCES` definition:

```python
# How each requested series was actually served this process: the provider
# name, "cache", or "fallback:<id>" when a mirror substituted for the
# primary. Surfaced in reports/leaderboards via provenance_flags().
PROVENANCE: dict[str, str] = {}


def provenance_flags(tickers: list[str]) -> list[str]:
    out = []
    for t in tickers:
        prov = PROVENANCE.get(t)
        if prov and prov.startswith("fallback:"):
            out.append(f"WARN: {t} served by fallback {prov.split(':', 1)[1]} "
                       f"(mirror data may differ in adjustment)")
    return out
```

In `load`, record provenance — in the fallback branch before recursing:

```python
            PROVENANCE[entry["id"]] = f"fallback:{entry['fallback']}"
            print(f"{entry['id']}: {exc} -- falling back to {entry['fallback']}")
            return load(entry["fallback"], start, end)
```

and after a successful provider fetch / cache hit:

```python
        _write_cache(cache, df)
    PROVENANCE[entry["id"]] = "fallback:..." if False else ("cache" if df_was_cached else source)
```

— concretely, replace the two `df`-producing branches so the final shape is:

```python
    df = None
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.empty or "close" not in df.columns:
            cache.unlink()          # truncated/interrupted write -- refetch
            df = None
        else:
            df = df.sort_index()    # self-heal a poisoned (unsorted) cache
            PROVENANCE[entry["id"]] = "cache"
    if df is None:
        try:
            df = PROVIDERS[source](symbol)
            if df.empty:
                raise ValueError("returned no data")
        except Exception as exc:
            if not entry.get("fallback"):
                raise ValueError(f"{entry['id']}: {exc}") from exc
            PROVENANCE[entry["id"]] = f"fallback:{entry['fallback']}"
            print(f"{entry['id']}: {exc} -- falling back to {entry['fallback']}")
            return load(entry["fallback"], start, end)
        _write_cache(cache, df)
        PROVENANCE[entry["id"]] = source
```

In `catalog.json`, change the yahoo:SPY entry's fallback from `null` to `"stooq:spy.us"`.

In `backtest.py` `run()`, extend the flags line (currently `out["auto_flags"] = auto_flags(spec, out)` at ~line 333):

```python
    out["auto_flags"] = auto_flags(spec, out)
    out["auto_flags"] += catalog.provenance_flags(tickers)
```

(The leaderboard and report already serialize `auto_flags`, so the substitution becomes visible with no further plumbing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py tests/test_backtest.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catalog.py catalog.json backtest.py tests/test_catalog.py
git commit -m "catalog: record fallback provenance and surface it in run flags; SPY mirror fallback"
```

---

### Task 3: Release-lag warning + frequency-vs-bar flags

**Files:**
- Modify: `catalog.py` (`_warn_frequency`, `frequency_flags`)
- Modify: `backtest.py` (wire `frequency_flags`)
- Test: `tests/test_catalog.py`, `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
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
```

Add to `tests/test_backtest.py` (new class; `BASE_SPEC` already exists in this file):

```python
class TestCatalogFrequencyFlags:
    def test_monthly_series_adds_warning_flag(self, monkeypatch):
        import catalog
        spec = copy.deepcopy(BASE_SPEC)
        spec["data"]["universe"] = ["SPY", "fred:CPIAUCSL"]
        flags = catalog.frequency_flags(spec["data"]["universe"],
                                        spec["data"].get("bar"))
        assert any("fred:CPIAUCSL" in f for f in flags)
```

(`import copy` already exists in test_backtest.py; verify at edit time.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestFrequency tests/test_backtest.py::TestCatalogFrequencyFlags -q`
Expected: FAIL — `_WARNED` and `frequency_flags` do not exist.

- [ ] **Step 3: Implement**

In `catalog.py`, add after `PROVENANCE`:

```python
_WARNED: set[str] = set()


def _warn_frequency(entry: dict) -> None:
    """Monthly/vintage series released after the period they describe give
    generated strategies lookahead bias if used directly -- say so loudly,
    once per series per process."""
    freq = entry.get("frequency")
    if freq and freq != "daily" and entry["id"] not in _WARNED:
        _WARNED.add(entry["id"])
        print(f"WARNING: {entry['id']} is {freq} -- it is released after the period "
              f"it describes, so using it directly gives lookahead bias; lag it in "
              f"the signal or state the caveat in data.caveats")


BAR_FREQUENCY = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}


def frequency_flags(tickers: list[str], bar: str | None) -> list[str]:
    """Spec-level frequency sanity: catalog frequency vs data.bar. A monthly
    series inside a daily panel creates phantom non-trading bars (diluted
    Sharpe) and the annualizer still assumes daily bars."""
    want = BAR_FREQUENCY.get(bar or "1d")
    if want is None:
        return []
    out = []
    for t in tickers:
        entry = by_id(t) or by_id(f"yahoo:{t}")
        if entry and entry.get("frequency") and entry["frequency"] != want:
            out.append(f"WARN: {entry['id']} is {entry['frequency']} but data.bar is "
                       f"{bar or '1d'} -- phantom bars dilute Sharpe and the "
                       f"annualizer assumes daily bars")
    return out
```

Call `_warn_frequency(entry)` in `load` right after `entry = resolve(series_id)`.

In `backtest.py` `run()`, extend the flags block again:

```python
    out["auto_flags"] = auto_flags(spec, out)
    out["auto_flags"] += catalog.frequency_flags(tickers, spec["data"].get("bar"))
    out["auto_flags"] += catalog.provenance_flags(tickers)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py tests/test_backtest.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catalog.py backtest.py tests/test_catalog.py tests/test_backtest.py
git commit -m "catalog: release-lag warning and frequency-vs-bar flags in run()"
```

---

### Task 4: OHLCV contract — next_open guard + source-shaped smoke

**Files:**
- Modify: `backtest.py` (`load_inputs` in `run()`)
- Modify: `catalog.py` (`is_close_only`)
- Modify: `codegen.py` (`smoke_data`, `smoke_test` optional-df param, `publish`)
- Test: `tests/test_backtest.py`, `tests/test_codegen.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest.py`:

```python
class TestOHLCVContract:
    def test_next_open_on_close_only_series_fails_cleanly(self, tmp_path, monkeypatch):
        import catalog
        spec = copy.deepcopy(BASE_SPEC)
        spec["data"]["universe"] = ["fred:DGS10"]
        spec["rules"]["execution_price"] = "next_open"
        idx = pd.bdate_range("2020-01-01", periods=60)
        close = pd.DataFrame({"close": np.linspace(100, 110, 60)}, index=idx)
        monkeypatch.setattr(backtest.catalog, "load",
                            lambda *a, **k: close.copy())

        with pytest.raises(SystemExit, match="no open column"):
            backtest.run_from_spec(spec)   # see Step 3 -- a small refactor of run()
```

Wait — `run()` takes a path. Refactor note (Step 3): split `run()`'s body into `run_spec(spec: dict, n_trials)` and keep `run(path)` a thin wrapper, so tests can drive it with an in-memory spec. Also add a direct `load_inputs`-level test that doesn't need the whole runner:

```python
    def test_close_only_flag_detection(self):
        import catalog
        assert catalog.is_close_only("fred:DGS10")
        assert not catalog.is_close_only("yahoo:SPY")
        assert not catalog.is_close_only("SPY")
```

Add to `tests/test_codegen.py` (imports `codegen` already exist there):

```python
class TestSourceShapedSmoke:
    def test_open_dependent_signal_fails_smoke_on_fred_series(self):
        # a strategy that indexes df["open"] must fail smoke for a close-only
        # catalog series instead of crashing later in run()
        module_src = (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return (df['open'] > df['open'].rolling(5).mean()).astype(float)\n"
        )
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "tmpstrategy.py"
            path.write_text(module_src)
            err = codegen.smoke_test(path, {}, df=catalog_close_only_frame())
        assert err and "open" in err

    def test_close_only_signal_passes_smoke_on_fred_series(self):
        module_src = (
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return (df['close'] > df['close'].rolling(5).mean()).astype(float)\n"
        )
        import pathlib, tempfile
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "tmpstrategy.py"
            path.write_text(module_src)
            err = codegen.smoke_test(path, {}, df=catalog_close_only_frame())
        assert err is None
```

with a module-level helper at the top of that class's file section:

```python
def catalog_close_only_frame():
    import catalog
    frame = codegen.smoke_frame()
    assert catalog.is_close_only("fred:DGS10")
    return frame[["close"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py::TestOHLCVContract tests/test_codegen.py::TestSourceShapedSmoke -q`
Expected: FAIL — `is_close_only` missing, `smoke_test` has no `df` param, `run_from_spec` missing.

- [ ] **Step 3: Implement**

In `catalog.py`:

```python
def is_close_only(series_id: str) -> bool:
    """True when the catalog serves this series as a close-only frame
    (FRED macro prints). The harness's next_open execution needs `open`."""
    entry = by_id(series_id) or by_id(f"yahoo:{series_id}")
    return bool(entry) and entry["id"].startswith("fred:")
```

In `backtest.py`, refactor the orchestration entry: rename the current `run(spec_path: Path, n_trials)` body to accept a loaded spec:

```python
def run(spec_path: Path, n_trials: int | None) -> dict:
    return run_spec(yaml.safe_load(spec_path.read_text()), n_trials)


def run_spec(spec: dict, n_trials: int | None) -> dict:
    slug = spec["meta"]["slug"]
    mod = load_strategy(slug)
    # ... rest of the current run() body unchanged, except load_inputs gains:
```

and inside `run_spec`'s `load_inputs`, guard both paths:

```python
    def load_inputs(spec_variant: dict):
        if len(tickers) == 1:
            df = load_prices(tickers[0], spec_variant["data"]["start"],
                             spec_variant["data"].get("end"), ROOT / "data" / "prices")
            if df.empty:
                sys.exit(f"no price data for {tickers[0]}")
        else:
            data, _ = load_panel(tickers, spec_variant["data"]["start"],
                                 spec_variant["data"].get("end"), ROOT / "data" / "prices")
        if spec_variant["rules"].get("execution_price") == "next_open":
            missing = [t for t in tickers
                       if (df if len(tickers) == 1 else data[t]).columns
                       .isin(["open"]).sum() == 0]
            if missing:
                sys.exit(f"{', '.join(missing)} has no open column (close-only series) "
                         f"— set execution_price: close in the spec's rules")
        return df if len(tickers) == 1 else data
```

In `codegen.py`:

1. `smoke_test` gains an optional frame parameter (multi path unchanged):

```python
def smoke_test(path: Path, params: dict, multi: bool = False,
               df: pd.DataFrame | None = None) -> str | None:
    """Import the module and call signal on synthetic data. None = OK.

    multi=False: params are the signal kwargs; `df` defaults to a full
    OHLCV frame. Pass a source-shaped frame (e.g. close-only for FRED
    series) so a signal that indexes a column the source never provides
    fails here instead of crashing in backtest.run().
    multi=True: params maps ticker -> OHLCV frame and is passed as `data`
    itself; no signal kwargs are forwarded, which is safe because the multi
    contract requires every parameter to have a default."""
    name = f"strategies.{path.stem}"
    spec_ = importlib.util.spec_from_file_location(
        name, path,
        loader=importlib.machinery.SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec_)
    try:
        spec_.loader.exec_module(mod)
        if not hasattr(mod, "signal"):
            return "defines no signal(df, **params)"
        if multi:
            data = params
            out = mod.signal(data)
        else:
            data = df if df is not None else smoke_frame()
            out = mod.signal(data, **params)
    except Exception as exc:  # generated code -- any failure is a codegen failure
        return f"{type(exc).__name__}: {exc}"
    if multi:
        if not isinstance(out, pd.DataFrame):
            return "signal did not return a pd.DataFrame (one column per ticker)"
        if set(out.columns) != set(data):
            return "weight columns do not match the universe"
        if not out.index.equals(next(iter(data.values())).index):
            return "weights index does not match data index"
        finite = np.isfinite(pd.to_numeric(out.stack(), errors="coerce"))
        if not finite.all():
            return "weights contained non-finite values"
        return None
    if not isinstance(out, pd.Series):
        return "signal did not return a pd.Series"
    if not out.index.equals(data.index):
        return "signal index does not match df.index"
    if not np.isfinite(pd.to_numeric(out, errors="coerce")).all():
        return "signal returned non-finite values"
    return None
```

2. Add `import catalog` to codegen.py's import block and the source-shaped helper:

```python
def smoke_data(series_id: str) -> pd.DataFrame:
    """Source-shaped smoke frame: close-only for close-only catalog series
    (FRED), full OHLCV otherwise."""
    frame = smoke_frame()
    if catalog.is_close_only(series_id):
        return frame[["close"]]
    return frame
```

3. In `publish`, replace the smoke calls:

```python
    if len(universe) > 1:
        err = smoke_test(tmp, {t: smoke_data(t) for t in universe}, multi=True)
    else:
        err = smoke_test(tmp, params, df=smoke_data(universe[0]) if universe else None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py tests/test_codegen.py tests/test_catalog.py -q`
Expected: all PASS. (If pre-existing codegen tests assert on `smoke_test`'s positional signature, adjust those call sites to the new optional param — behavior for existing callers is unchanged.)

- [ ] **Step 5: Commit**

```bash
git add backtest.py catalog.py codegen.py tests/test_backtest.py tests/test_codegen.py
git commit -m "backtest/codegen: OHLCV contract guards for close-only catalog series"
```

---

### Task 5: Unseeded yahoo ids pass through + friendly catalog errors

**Files:**
- Modify: `catalog.py` (`_entry_for` replacing the bare `resolve` call in `load`)
- Modify: `backtest.py` (`load_prices` catches `ValueError` -> `sys.exit`)
- Test: `tests/test_catalog.py`, `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py` (new class; stubs as in TestLoad):

```python
class TestUnseededYahoo:
    def test_unseeded_yahoo_id_fetches_like_bare_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(catalog, "CACHE_DIR", tmp_path)
        yf = pd.DataFrame({"Close": [250.0], "Open": [249.0], "High": [251.0],
                           "Low": [248.0], "Volume": [900]},
                          index=pd.DatetimeIndex(["2024-01-02"]))
        self._stub_yahoo(monkeypatch, yf)

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
```

Add to `tests/test_backtest.py::TestCatalogUniverse`:

```python
    def test_catalog_failure_exits_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest.catalog, "load",
                            lambda *a, **k: (_ for _ in ()).throw(
                                ValueError("fred:NOPE_NOPE not in catalog")))

        with pytest.raises(SystemExit, match="not in catalog"):
            backtest.load_prices("fred:NOPE_NOPE", None, None, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestUnseededYahoo tests/test_backtest.py::TestCatalogUniverse -q`
Expected: FAIL — unseeded yahoo raises "not in catalog"; load_prices lets ValueError propagate as a traceback.

- [ ] **Step 3: Implement**

In `catalog.py`, add an entry-lookup helper that lenients yahoo ids:

```python
def _entry_for(series_id: str) -> dict:
    """Seeded entry, or a synthetic one for unseeded yahoo: ids -- they
    fetch exactly like bare tickers always have. fred:/stooq: ids must be
    seeded; a typo should be a loud error, not a silent empty fetch."""
    entry = by_id(series_id) or (by_id(f"yahoo:{series_id}")
                                 if ":" not in series_id else None)
    if entry is not None:
        return entry
    source, symbol = parse_id(series_id)   # raises ValueError for unknown sources
    if source != "yahoo":
        raise ValueError(f"{series_id!r} not in catalog "
                         f"(try: python3 catalog.py search <query>)")
    return {"id": f"yahoo:{symbol}", "title": symbol, "tags": [],
            "frequency": "daily", "coverage_start": None,
            "vintage_available": False, "notes": "unseeded yahoo id",
            "fallback": None}
```

In `load`, replace `entry = resolve(series_id)` with `entry = _entry_for(series_id)`.

In `backtest.py` `load_prices`:

```python
def load_prices(ticker: str, start, end, cache_dir: Path) -> pd.DataFrame:
    # catalog ids ("fred:DGS10", "yahoo:SPY", ...) go through the catalog;
    # bare tickers keep the legacy yfinance path
    if ":" in ticker:
        try:
            return catalog.load(ticker, start, end)
        except ValueError as exc:   # same friendly exit as the legacy path
            sys.exit(f"error: {exc}")
    cache_dir.mkdir(parents=True, exist_ok=True)
```

(legacy block unchanged below).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py tests/test_backtest.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catalog.py backtest.py tests/test_catalog.py tests/test_backtest.py
git commit -m "catalog: unseeded yahoo ids pass through; catalog errors exit cleanly in backtest"
```

---

### Task 6: extract.py accepts catalog ids

**Files:**
- Modify: `extract.py` (`validate_spec` universe rule + import)
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_extract.py` (the file already imports `extract` and has a valid `BASE_SPEC`-style fixture — reuse its valid-spec helper; the new tests below mutate `universe`):

```python
class TestCatalogUniverseValidation:
    def _spec_with_universe(self, make_spec, universe):
        spec = make_spec()
        spec["data"]["universe"] = universe
        return spec

    def test_fred_id_accepted(self, make_spec):
        errs = extract.validate_spec(
            self._spec_with_universe(make_spec, ["fred:DGS10"]), "slug")
        assert not any("universe" in e for e in errs)

    def test_mixed_universe_accepted(self, make_spec):
        errs = extract.validate_spec(
            self._spec_with_universe(make_spec, ["SPY", "fred:DGS10"]), "slug")
        assert not any("universe" in e for e in errs)

    def test_unknown_fred_id_rejected(self, make_spec):
        errs = extract.validate_spec(
            self._spec_with_universe(make_spec, ["fred:NOPE_NOPE"]), "slug")
        assert any("universe" in e for e in errs)

    def test_unseeded_yahoo_id_accepted(self, make_spec):
        errs = extract.validate_spec(
            self._spec_with_universe(make_spec, ["yahoo:AAPL"]), "slug")
        assert not any("universe" in e for e in errs)

    def test_unknown_source_rejected(self, make_spec):
        errs = extract.validate_spec(
            self._spec_with_universe(make_spec, ["bogus:XYZ"]), "slug")
        assert any("universe" in e for e in errs)
```

(Adapt the fixture name `make_spec` to whatever this file already uses for a valid spec — check its existing tests first. If the file builds specs inline, follow that pattern instead.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extract.py::TestCatalogUniverseValidation -q`
Expected: FAIL — `fred:DGS10` fails the Yahoo-format regex.

- [ ] **Step 3: Implement**

In `extract.py`, add `import catalog` to the import block, and replace the universe check in `validate_spec` (lines ~115-123):

```python
    data = doc.get("data")
    uni = data.get("universe") if isinstance(data, dict) else None

    def ticker_ok(t) -> bool:
        if not isinstance(t, str):
            return False
        if re.fullmatch(r"[A-Z0-9\-\.\^=]{1,12}", t):
            return True                       # legacy bare Yahoo ticker
        m = re.fullmatch(r"(yahoo|fred|stooq):(.{1,24})", t)
        if not m:
            return False
        if m.group(1) == "yahoo":
            return True                       # unseeded yahoo ids fetch like bare tickers
        try:
            catalog.resolve(t)
            return True                       # fred:/stooq: ids must be seeded
        except ValueError:
            return False

    if not (isinstance(uni, list) and uni and all(ticker_ok(t) for t in uni)):
        errors.append("data.universe must be explicit tickers or resolvable catalog "
                      "ids (e.g. [SPY, fred:DGS10] — find ids with "
                      "python3 catalog.py search) — rule-based universes are "
                      "unsupported; mark the article UNTESTABLE instead")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extract.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add extract.py tests/test_extract.py
git commit -m "extract: accept resolvable source:id universe entries"
```

---

### Task 7: FRED via pandas-datareader (design re-open, §4)

**Files:**
- Modify: `pyproject.toml` + `uv.lock` (`uv add pandas-datareader`)
- Modify: `catalog.py` (`_fetch_fred` via `FredReader`; drop `FRED_URL`)
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Add the dependency**

```bash
uv add pandas-datareader
```

Then verify in the project env:

```bash
uv run python -c "
from pandas_datareader.fred import FredReader
df = FredReader('DGS10', start='2026-08-01').read()
print(df.tail(2))"
```

Expected: two recent DGS10 rows (verified live in planning: pdr 0.11.1 + pandas 3.0.5 works). If this fails, STOP — report back; do not hand-roll around it.

- [ ] **Step 2: Rewrite the FRED tests first**

In `tests/test_catalog.py`, `TestLoad.test_fred_maps_value_to_close_and_caches` currently stubs `pd.read_csv` on the fredgraph URL. Replace it with a `FredReader` stub (keep the cache/provenance assertions):

```python
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
```

- [ ] **Step 3: Run to verify red**

Run: `uv run pytest tests/test_catalog.py::TestLoad::test_fred_maps_value_to_close_and_caches -q`
Expected: FAIL — the fredgraph URL is still fetched (stub no longer matches).

- [ ] **Step 4: Implement**

In `catalog.py`, replace `import ... FRED_URL` machinery:

- Delete the `FRED_URL` constant.
- Add to the imports: `from pandas_datareader.fred import FredReader`
- Replace `_fetch_fred`:

```python
def _fetch_fred(series: str) -> pd.DataFrame:
    """Documented macro sources go through pandas-datareader (issue #10 §4):
    stable endpoints, maintained reader, and one place to add World Bank /
    OECD / Eurostat later. Yahoo stays yfinance (datareader's Yahoo reader
    is dead); Stooq stays direct CSV (no datareader reader exists)."""
    raw = FredReader(series).read()
    df = raw.apply(pd.to_numeric, errors="coerce").dropna()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df.columns = ["close"]
    df.index.name = None
    return df
```

(`PROVIDERS` unchanged. Existing Stooq `read_csv` stub tests keep passing.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py -q`
Expected: all PASS.

- [ ] **Step 6: Live verify + commit**

```bash
uv run python catalog.py fetch fred:DGS10 --refresh
git add pyproject.toml uv.lock catalog.py tests/test_catalog.py
git commit -m "catalog: FRED via pandas-datareader (design re-open from issue 10)"
```

---

### Task 8: Minors — dedupe normalization, multi-word search, license field

**Files:**
- Modify: `catalog.py` (`normalize_yahoo`, `search`), `backtest.py` (legacy branch uses `catalog.normalize_yahoo`)
- Modify: `catalog.json` (add `license` to every entry)
- Test: `tests/test_catalog.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
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

        assert list(df.columns) == ["close", "open", "high", "low", "volume"]
        assert df.index.tz is None

    def test_every_entry_has_a_license(self):
        for entry in catalog._index()["series"]:
            assert entry.get("license"), f"{entry['id']} missing license"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalog.py::TestMinors -q`
Expected: FAIL — `normalize_yahoo` missing; single-substring search excludes multi-word; entries lack `license`.

- [ ] **Step 3: Implement**

In `catalog.py`:

1. Extract the normalization (public, so backtest.py's legacy branch shares it):

```python
def normalize_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by catalog._fetch_yahoo and backtest.py's legacy bare-ticker
    path -- one normalization, not two diverging ones."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _fetch_yahoo(symbol: str) -> pd.DataFrame:
    df = _yahoo_download(symbol, start="1990-01-01", auto_adjust=True,
                         progress=False)
    return normalize_yahoo(df)
```

2. Multi-word search — every token must match:

```python
def search(query: str) -> list[dict]:
    """Case-insensitive match on id, title, and tags; multi-word queries
    require every term to match (AND)."""
    terms = query.lower().split()
    out = []
    for entry in _index()["series"]:
        hay = " ".join([entry["id"], entry["title"],
                        " ".join(entry["tags"])]).lower()
        if all(t in hay for t in terms):
            out.append(entry)
    return out
```

3. In `backtest.py`'s legacy branch (the `else` under `cache.exists()`), replace the duplicated normalization block:

```python
    else:
        import yfinance as yf

        df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
        df = catalog.normalize_yahoo(df)
        df.to_csv(cache)
```

4. In `catalog.json`, add a `license` field to every entry: `"license": "public domain (FRED)"` for all `fred:` entries, `"license": "Yahoo Finance terms of use"` for `yahoo:` entries, `"license": "Stooq terms of use"` for `stooq:` entries. (coverage_end / update-cadence need live queries — noted as future work in the PR body, not stubbed here.)

5. In `main_with`'s `show` branch, add `"license"` to the printed keys tuple.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalog.py tests/test_backtest.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catalog.py catalog.json backtest.py tests/test_catalog.py
git commit -m "catalog: shared yahoo normalization, AND-token search, license field"
```

---

### Task 9: README + full validation + live smoke + PR

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the README "Data catalog" section**

Amend the section added for PR #9 so it reads (replace the last two sentences about fallback with provenance/refresh guidance):

```markdown
### Data catalog

Universe entries can be bare Yahoo tickers (`SPY`) or catalog ids:
`yahoo:SPY`, `fred:DGS10`, `stooq:spy.us`. Find series with
`python3 catalog.py search <query>` (multi-word queries require every term),
inspect one with `python3 catalog.py show <id>`, and pre-warm the cache with
`python3 catalog.py fetch <id>` (`--refresh` forces a refetch). The committed
seed index is `catalog.json`; series cache under `data/catalog/`. Caches are
validated on load and written atomically — a truncated cache refetches, it
never shadows real data. If a primary provider fails, an entry's declared
fallback (a Yahoo/Stooq mirror) is used and the report's flags record the
substitution. FRED series load through pandas-datareader; Yahoo stays
yfinance; Stooq uses its CSV endpoint.

Two caveats the harness flags automatically (they land in the report's
Automated flags and the leaderboard):

- monthly/vintage series (e.g. `fred:CPIAUCSL`) are released after the
  period they describe — using them directly gives lookahead bias;
- a catalog series whose frequency differs from `data.bar` (e.g. a monthly
  series in a daily panel) distorts Sharpe through phantom bars.

Strategies that run with `execution_price: next_open` require an `open`
column; close-only FRED series are rejected with a clear message, and the
codegen smoke test uses source-shaped frames so such strategies fail there
instead.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: all PASS (~225+ tests; was 209).

- [ ] **Step 3: Live smoke**

```bash
uv run python catalog.py fetch fred:DGS10 --refresh
uv run python catalog.py fetch yahoo:QQQ --refresh
uv run python catalog.py search "treasury yield"
```

Expected: fresh FRED bars via pandas-datareader; fresh Stooq/Yahoo bars sorted ascending; search returns the two yield-curve entries. (If the network is unavailable, record that and rely on the stubbed tests.)

- [ ] **Step 4: Commit docs and push**

```bash
git add README.md
git commit -m "docs: catalog hardening behavior -- provenance flags, refresh, frequency caveats"
git push -u fork issue10-catalog-hardening
```

- [ ] **Step 5: Open the PR**

`gh pr create --repo uberdeveloper/autoquant --base main --head neha-priyaa:issue10-catalog-hardening`, title "catalog hardening: data integrity, backtest correctness guards, catalog-id specs, pandas-datareader FRED", body covering every issue-#10 section with its resolution, the datareader compat verification (pdr 0.11.1 + pandas 3.0.5 live), and the test/live-smoke plan. Closes #10.

---

## Self-review

- **Spec coverage:** §1a Task 1; §1b Task 1; §1c Task 2 (PROVENANCE + provenance_flags -> auto_flags -> report+leaderboard); §1d Task 2; §2a Task 3 (loud once-per-series load warning) + Task 3 flags; §2b Task 3 `frequency_flags` wired into run(); §2c Task 4 (run guard + source-shaped smoke via `smoke_data`/`is_close_only`); §3 pass-through Task 5, extract ids Task 6, friendly errors Task 5; §4 Task 7 (live-verified pdr 0.11.1 + pandas 3.0.5); §5a Task 8 (`normalize_yahoo` shared; caches intentionally stay per-path — full unification would break the panel test fixtures and is out of the minor-item's scope), §5b Task 8 (license field; coverage_end/cadence explicitly deferred with rationale), §5c Task 8.
- **Placeholder scan:** the Task 4 test block contains an inline "see Step 3" narration — Step 3 carries the full `run_spec` refactor code; the test code itself is complete. The Task 2 Step 3 sketch shows a `False` line before the concretely-shaped final code — implementers use the final shape only.
- **Type consistency:** `cache_path(series_id) -> Path` used by `load` and the CLI; `provenance_flags(tickers) -> list[str]` and `frequency_flags(tickers, bar) -> list[str]` both return flag strings appended to `out["auto_flags"]`; `is_close_only(series_id) -> bool` consumed by codegen `smoke_data`; `smoke_test(path, params, multi=False, df=None)` — existing callers unchanged.
- **Determinism:** all tests stub network (`pd.read_csv` for Stooq, `FredReader` for FRED, `_yahoo_download` for Yahoo); `_WARNED`/`PROVENANCE` are reset in tests that depend on their freshness.
