"""Tests for shim.py + external_shim/ -- the offline data shim (#26).

Every test here is offline by construction: the fixture caches live under
``tmp_path`` (``AUTOQUANT_CATALOG_DIR``), the providers are monkeypatched to
recorders, and the only subprocesses launched are local python runs with the
shim on ``PYTHONPATH``.  Nothing here touches the network, the real
``data/catalog`` tree, or ``data/repos/``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import types
from pathlib import Path

import pandas as pd
import pytest

import catalog
import shim
from external_shim import _core
from external_shim import yfinance as yf
from external_shim.pandas_datareader import data as pdr_data
from external_shim.pandas_datareader import fred as pdr_fred

ROOT = Path(shim.__file__).resolve().parent


# ----------------------------------------------------------------- fixtures

def _ohlcv(start: str = "2024-01-02", periods: int = 6, base: float = 100.0):
    """A tiny normalized OHLCV frame (lowercase columns, DatetimeIndex)."""
    index = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {
            "open": [base + i for i in range(periods)],
            "high": [base + i + 1 for i in range(periods)],
            "low": [base + i - 1 for i in range(periods)],
            "close": [base + i + 0.5 for i in range(periods)],
            "volume": [1000 + i for i in range(periods)],
        },
        index=index,
    )


def _recent_ohlcv(periods: int = 60, base: float = 100.0):
    """Business days ending at the last business day on or before today."""
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=periods)
    return pd.DataFrame(
        {
            "open": [base + i for i in range(periods)],
            "high": [base + i + 1 for i in range(periods)],
            "low": [base + i - 1 for i in range(periods)],
            "close": [base + i + 0.5 for i in range(periods)],
            "volume": [1000 + i for i in range(periods)],
        },
        index=index,
    )


def _close_only(start: str = "2024-01-02", periods: int = 4, base: float = 4.0):
    index = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"close": [base + i for i in range(periods)]},
                        index=index)


def _write_cache(root: Path, series_id: str, frame: pd.DataFrame) -> Path:
    """Write a cache file exactly where ``catalog.cache_path`` expects it."""
    source, symbol = series_id.split(":", 1)
    safe = re.sub(r"[^A-Za-z0-9]", "_", symbol)
    path = root / f"{source}_{safe}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="Date")
    return path


@pytest.fixture
def catalog_dir(tmp_path, monkeypatch):
    """An injectable cache root (never the real, empty data/catalog)."""
    root = tmp_path / "catalog"
    root.mkdir()
    monkeypatch.setenv(shim.CATALOG_DIR_ENV, str(root))
    monkeypatch.setenv(shim.OFFLINE_ENV, "1")
    return root


@pytest.fixture
def provider_calls(monkeypatch):
    """A recording stand-in for every provider of the shim's catalog module."""
    cat = _core.catalog_module()
    calls: list[tuple[str, str]] = []

    def recorder(source):
        def _fetch(symbol, *args, **kwargs):
            calls.append((source, symbol))
            return _ohlcv()
        return _fetch

    for name in list(cat.PROVIDERS):
        monkeypatch.setitem(cat.PROVIDERS, name, recorder(name))
    return calls


# ------------------------------------------------------------------- tests

class TestShimEnv:
    def test_pythonpath_is_prefixed_with_the_shim_dir(self):
        env = shim.shim_env(existing="")
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(shim.shim_root())
        assert shim.shim_root().is_dir()
        assert (shim.shim_root() / "yfinance").is_dir()
        assert (shim.shim_root() / "pandas_datareader").is_dir()

    def test_extra_path_and_existing_pythonpath_are_kept(self):
        env = shim.shim_env(extra_path="/opt/extra", existing="/opt/existing")
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts[0] == str(shim.shim_root())
        assert parts[1] == "/opt/extra"
        assert parts[2] == "/opt/existing"

    def test_catalog_dir_is_injectable(self, tmp_path):
        env = shim.shim_env(catalog_root=tmp_path, existing="")
        assert env[shim.CATALOG_DIR_ENV] == str(tmp_path)

    def test_default_catalog_dir_is_the_project_cache(self):
        env = shim.shim_env(existing="")
        assert env[shim.CATALOG_DIR_ENV] == str(shim.DEFAULT_CATALOG_DIR)


class TestImportResolution:
    def test_import_yfinance_resolves_to_the_shim(self, tmp_path):
        env = {**os.environ, **shim.shim_env(catalog_root=tmp_path,
                                             existing="")}
        proc = subprocess.run(
            [sys.executable, "-c",
             "import yfinance, pandas_datareader;"
             "print(yfinance.__file__);"
             "print(pandas_datareader.__file__)"],
            env=env, capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr
        yf_file, pdr_file = proc.stdout.split()
        assert Path(yf_file).is_relative_to(shim.shim_root())
        assert Path(pdr_file).is_relative_to(shim.shim_root())


class TestDownload:
    def test_download_serves_the_fixture_cache_offline(self, catalog_dir):
        frame = _ohlcv()
        _write_cache(catalog_dir, "yahoo:AAPL", frame)
        got = yf.download("AAPL", start="2024-01-03", end="2024-01-05")
        assert isinstance(got, pd.DataFrame)
        assert isinstance(got.index, pd.DatetimeIndex)
        assert got.index.is_monotonic_increasing
        assert list(got.index) == list(frame.index[1:4])
        for column in ("open", "high", "low", "close", "volume"):
            assert column in got.columns
        assert "Close" in got.columns
        pd.testing.assert_series_equal(got["Close"], got["close"],
                                       check_names=False)
        pd.testing.assert_series_equal(
            got["close"], frame["close"].iloc[1:4], check_names=False,
            check_freq=False)

    def test_download_of_an_unseeded_symbol_uses_the_cache(self, catalog_dir):
        # yahoo:AAPL is NOT in catalog.json -- the cache is what makes it
        # servable, not the seed index.
        assert catalog.by_id("yahoo:AAPL") is None
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        got = yf.download("AAPL")
        assert len(got) == 6

    def test_ticker_history_matches_download(self, catalog_dir):
        frame = _ohlcv()
        _write_cache(catalog_dir, "yahoo:SPY", frame)
        hist = yf.Ticker("SPY").history(start="2024-01-02", end="2024-01-05")
        down = yf.download("SPY", start="2024-01-02", end="2024-01-05")
        pd.testing.assert_frame_equal(hist, down)

    def test_the_yahoo_route_does_not_serve_a_fred_series(self, catalog_dir):
        _write_cache(catalog_dir, "fred:DGS10", _close_only())
        with pytest.raises(shim.DataNotAvailable):
            yf.download("DGS10")  # yahoo:DGS10 is not cached


class TestPeriod:
    """A ``period`` must actually narrow the served window (R1)."""

    @pytest.fixture
    def recent(self, catalog_dir):
        frame = _recent_ohlcv(60)
        _write_cache(catalog_dir, "yahoo:SPY", frame)
        return frame

    def test_max_returns_the_whole_series(self, recent):
        got = yf.download("SPY", period="max")
        assert list(got.index) == list(recent.index)

    def test_a_period_narrows_the_window(self, recent):
        full = yf.download("SPY", period="max")
        month = yf.download("SPY", period="1mo")
        today = pd.Timestamp.now().normalize()
        assert len(month) < len(full)
        assert month.index.min() >= today - pd.Timedelta(days=31)
        assert month.index.max() <= today

    def test_5d_returns_the_last_five_trading_days(self, recent):
        got = yf.download("SPY", period="5d")
        assert len(got) == 5
        assert got.index.max() == recent.index.max()

    def test_ytd_returns_the_rows_of_this_year(self, recent):
        got = yf.download("SPY", period="ytd")
        year_start = pd.Timestamp(year=pd.Timestamp.now().year, month=1, day=1)
        expected = recent[recent.index >= year_start]
        assert list(got.index) == list(expected.index)

    def test_ticker_history_honours_the_period(self, recent):
        hist = yf.Ticker("SPY").history(period="5d")
        assert len(hist) == 5

    def test_an_unsupported_period_is_refused(self, recent):
        with pytest.raises(ValueError):
            yf.download("SPY", period="1m")

    def test_a_week_period_is_the_same_however_it_is_spelled(self, recent):
        # 'w' and 'wk' both mean weeks -- the period must not silently
        # widen to a year.
        one_w = yf.download("SPY", period="1w")
        one_wk = yf.download("SPY", period="1wk")
        assert list(one_w.index) == list(one_wk.index)
        assert len(one_w) == len(one_wk)

    def test_a_week_period_is_not_a_year(self, recent):
        one_w = yf.download("SPY", period="1w")
        one_y = yf.download("SPY", period="1y")
        assert list(one_w.index) != list(one_y.index)
        assert len(one_w) < len(one_y)


class TestDataReader:
    def test_fred_and_yahoo_route_to_different_catalog_ids(self, catalog_dir):
        fred = _close_only(base=4.0)
        yahoo = _ohlcv(base=200.0)
        _write_cache(catalog_dir, "fred:DGS10", fred)
        _write_cache(catalog_dir, "yahoo:DGS10", yahoo)
        got_fred = pdr_data.DataReader("DGS10", "fred")
        got_yahoo = pdr_data.DataReader("DGS10", "yahoo")
        assert got_fred["close"].iloc[0] == pytest.approx(4.0)
        assert got_yahoo["close"].iloc[0] == pytest.approx(200.5)
        assert "Close" in got_fred.columns

    def test_unknown_source_raises_data_not_available(self, catalog_dir):
        with pytest.raises(shim.DataNotAvailable):
            pdr_data.DataReader("DGS10", "not-a-source")


class TestOffline:
    def test_unknown_symbol_raises_data_not_available(self, catalog_dir,
                                                       provider_calls):
        with pytest.raises(shim.DataNotAvailable) as excinfo:
            yf.download("ZZZZ_NOT_A_SYMBOL")
        assert "ZZZZ_NOT_A_SYMBOL" in str(excinfo.value)
        assert provider_calls == []

    def test_offline_flag_blocks_the_providers(self, catalog_dir,
                                               provider_calls):
        with pytest.raises(shim.DataNotAvailable):
            yf.download("AAPL")
        assert provider_calls == []

    def test_a_seeded_miss_never_reaches_a_provider(self, catalog_dir,
                                                     provider_calls):
        # yahoo:SPY is seeded but has no cached file: a provider WOULD be
        # hit here if the offline guard were disabled.
        assert catalog.by_id("yahoo:SPY") is not None
        with pytest.raises(shim.DataNotAvailable):
            yf.download("SPY")
        assert provider_calls == []

    def test_the_error_type_is_shared(self):
        assert shim.DataNotAvailable is yf.DataNotAvailable
        assert issubclass(shim.DataNotAvailable, RuntimeError)

    def test_going_online_lets_the_provider_serve_the_request(
            self, catalog_dir, provider_calls, monkeypatch):
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        got = yf.download("AAPL")
        assert provider_calls == [("yahoo", "AAPL")]
        assert "close" in got.columns

    def test_a_cache_hit_never_reaches_a_provider(self, catalog_dir,
                                                   provider_calls):
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        yf.download("AAPL")
        assert provider_calls == []


class TestOnlineMode:
    """Online mode must fail cleanly, never recurse into the shim (R3)."""

    def test_a_miss_without_a_real_yfinance_raises_data_not_available(
            self, catalog_dir, monkeypatch):
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        monkeypatch.setattr(_core, "real_yfinance", lambda: None)
        with pytest.raises(shim.DataNotAvailable) as excinfo:
            yf.download("SPY")
        message = str(excinfo.value)
        assert "SPY" in message
        # a single, clean error -- not a recursion, not a doubled message
        assert "yfinance" in message
        assert "maximum recursion depth" not in message
        assert "yahoo:SPY: yahoo:SPY:" not in message

    def test_a_provider_that_calls_back_into_the_shim_does_not_recurse(
            self, catalog_dir, monkeypatch):
        cat = _core.catalog_module()

        def recursive(symbol, *args, **kwargs):
            return yf.download(symbol)

        monkeypatch.setitem(cat.PROVIDERS, "yahoo", recursive)
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        with pytest.raises(shim.DataNotAvailable) as excinfo:
            yf.download("SPY")
        message = str(excinfo.value)
        # the guard fired -- the failure is not a RecursionError
        assert "already in flight" in message
        assert "maximum recursion depth" not in message

    def test_an_online_fred_miss_raises_data_not_available(self, catalog_dir,
                                                            monkeypatch):
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        with pytest.raises(shim.DataNotAvailable) as excinfo:
            pdr_data.DataReader("DGS10", "fred")
        assert "DGS10" in str(excinfo.value)

    def test_a_warm_cache_is_served_before_any_provider_is_resolved(
            self, catalog_dir, monkeypatch):
        # The shim is cache-first: in online mode a request the cache can
        # serve must not touch the provider machinery at all -- not even to
        # resolve the real yfinance package (the R3 identity gate).
        #
        # This test deliberately does NOT monkeypatch the providers: the
        # catalog's own ``_fetch_yahoo`` stays in place, so the gate is
        # live and a provider resolution would really happen.
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        frame = _ohlcv()
        _write_cache(catalog_dir, "yahoo:AAPL", frame)
        # A provider call loads the real package under this name.
        monkeypatch.delitem(sys.modules, "_shim_real_yfinance", raising=False)
        got = yf.download("AAPL")
        assert len(got) == len(frame)
        assert got["close"].iloc[0] == pytest.approx(frame["close"].iloc[0])
        # Zero provider calls: nothing was resolved, nothing was fetched.
        assert "_shim_real_yfinance" not in sys.modules
        provenance = _core.catalog_module().PROVENANCE
        assert provenance.get("yahoo:AAPL") == "cache"

    def test_a_failing_online_provider_surfaces_as_data_not_available(
            self, catalog_dir, monkeypatch):
        cat = _core.catalog_module()

        def boom(symbol, *args, **kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setitem(cat.PROVIDERS, "yahoo", boom)
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        with pytest.raises(shim.DataNotAvailable) as excinfo:
            yf.download("AAPL")
        assert "provider exploded" in str(excinfo.value)


class TestThreadSafety:
    """The offline guarantee must hold under concurrency (N3).

    ``catalog.PROVIDERS`` and ``catalog.CACHE_DIR`` are module globals of
    the *catalog* module, so two concurrent shim calls interleave their
    block/restore unless the shim serializes the critical section.
    """

    def test_a_concurrent_load_and_miss_never_leak_a_provider_call(
            self, catalog_dir, provider_calls, monkeypatch):
        cat = _core.catalog_module()
        original_provider = cat.PROVIDERS["yahoo"]
        original_cache_dir = cat.CACHE_DIR

        hit_entered = threading.Event()
        miss_entered = threading.Event()
        hit_done = threading.Event()

        real_load = cat.load

        def gated_load(series_id, start, end):
            # Force a deterministic interleaving of the two calls.
            if series_id == "yahoo:SPY":          # the miss
                miss_entered.set()
                hit_done.wait(2.0)
            else:                                 # the cache hit
                hit_entered.set()
                miss_entered.wait(2.0)
            return real_load(series_id, start, end)

        monkeypatch.setattr(cat, "load", gated_load)
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())

        results: dict[str, object] = {}

        def hit():
            try:
                results["frame"] = yf.download("AAPL")
            except BaseException as exc:         # reported below
                results["hit_error"] = exc
            finally:
                hit_done.set()

        def miss():
            hit_entered.wait(2.0)
            try:
                yf.download("SPY")
            except shim.DataNotAvailable:
                results["missed"] = True
            except BaseException as exc:         # reported below
                results["miss_error"] = exc

        first = threading.Thread(target=hit)
        second = threading.Thread(target=miss)
        first.start()
        second.start()
        first.join(10)
        second.join(10)

        assert "hit_error" not in results, results.get("hit_error")
        assert "miss_error" not in results, results.get("miss_error")
        # The hit is served from the cache ...
        assert len(results["frame"]) == 6
        # ... the miss is refused ...
        assert results.get("missed") is True
        # ... and no provider was ever called.
        assert provider_calls == []
        # The global state is left exactly as it was found.
        assert cat.PROVIDERS["yahoo"] is original_provider
        assert cat.CACHE_DIR == original_cache_dir


class TestFred:
    def test_the_fred_reader_refuses_to_fetch(self):
        with pytest.raises(shim.DataNotAvailable):
            pdr_fred.FredReader("DGS10").read()


class TestFallbackMirror:
    def test_a_cached_mirror_serves_the_primary_request(self, catalog_dir,
                                                         provider_calls):
        # yahoo:SPY declares the stooq:spy.us mirror; only the mirror is
        # cached -- the request must be served from it.
        assert catalog.by_id("yahoo:SPY")["fallback"] == "stooq:spy.us"
        _write_cache(catalog_dir, "stooq:spy.us", _ohlcv(base=50.0))
        got = yf.download("SPY")
        assert got["close"].iloc[0] == pytest.approx(50.5)
        assert provider_calls == []

    def test_the_shim_resolves_the_mirror_itself(self, catalog_dir,
                                                 provider_calls, monkeypatch):
        # Same setup, but online: the shim must resolve the mirror itself --
        # the primary's provider must never be consulted (that is what
        # ``catalog.load``'s own fallback would do).
        monkeypatch.setenv(shim.OFFLINE_ENV, "0")
        _write_cache(catalog_dir, "stooq:spy.us", _ohlcv(base=50.0))
        got = yf.download("SPY")
        assert got["close"].iloc[0] == pytest.approx(50.5)
        assert provider_calls == []


class TestInjectableCacheRoot:
    def test_two_cache_roots_serve_different_data(self, tmp_path,
                                                   monkeypatch):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        _write_cache(first, "yahoo:AAPL", _ohlcv(base=100.0))
        _write_cache(second, "yahoo:AAPL", _ohlcv(base=500.0))
        monkeypatch.setenv(shim.OFFLINE_ENV, "1")

        monkeypatch.setenv(shim.CATALOG_DIR_ENV, str(first))
        one = yf.download("AAPL")
        monkeypatch.setenv(shim.CATALOG_DIR_ENV, str(second))
        two = yf.download("AAPL")
        assert one["close"].iloc[0] == pytest.approx(100.5)
        assert two["close"].iloc[0] == pytest.approx(500.5)


class TestIsolation:
    def test_a_download_does_not_change_sys_path(self, catalog_dir):
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        before = list(sys.path)
        yf.download("AAPL")
        assert sys.path == before

    def test_an_external_module_is_not_shadowed_by_the_project(
            self, catalog_dir, tmp_path, monkeypatch):
        external = tmp_path / "external_repo"
        external.mkdir()
        (external / "backtest.py").write_text("VALUE = 'external'\n")
        monkeypatch.syspath_prepend(str(external))
        monkeypatch.delitem(sys.modules, "backtest", raising=False)
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        yf.download("AAPL")
        try:
            import backtest
            assert Path(backtest.__file__).parent == external
        finally:
            sys.modules.pop("backtest", None)

    def test_a_foreign_catalog_module_does_not_break_the_shim(
            self, catalog_dir, tmp_path, monkeypatch):
        foreign = types.ModuleType("catalog")
        foreign.__file__ = str(tmp_path / "catalog.py")
        monkeypatch.setitem(sys.modules, "catalog", foreign)
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        got = yf.download("AAPL")
        assert len(got) == 6

    def test_the_shim_writes_nothing_under_data_repos(self, catalog_dir):
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        repos = ROOT / "data" / "repos"
        before = _snapshot(repos)
        env = {**os.environ, **shim.shim_env(catalog_root=catalog_dir,
                                             existing="")}
        proc = subprocess.run(
            [sys.executable, "-c",
             "import yfinance;"
             "df = yfinance.download('AAPL');"
             "print(len(df))"],
            env=env, capture_output=True, text=True, cwd=str(ROOT))
        assert proc.returncode == 0, proc.stderr
        assert _snapshot(repos) == before


class TestCacheHandling:
    def test_cache_dir_is_restored_after_a_load(self, catalog_dir):
        cat = _core.catalog_module()
        before = cat.CACHE_DIR
        _write_cache(catalog_dir, "yahoo:AAPL", _ohlcv())
        yf.download("AAPL")
        assert cat.CACHE_DIR == before

    def test_a_malformed_cache_file_is_not_deleted(self, catalog_dir):
        path = catalog_dir / "yahoo_AAPL.csv"
        path.write_text("Date,open,high,low,close,volume\n")
        with pytest.raises(shim.DataNotAvailable):
            yf.download("AAPL")
        assert path.exists()


class TestHardOffline:
    @pytest.mark.skipif(not shim.unshare_available(),
                        reason="unshare -n is unavailable in this sandbox")
    def test_unshare_prefix_is_usable(self, monkeypatch):
        monkeypatch.setenv(shim.HARD_OFFLINE_ENV, "1")
        prefix = shim.hard_offline_prefix()
        assert prefix == ["unshare", "-n"]
        proc = subprocess.run([*prefix, "true"], capture_output=True, text=True)
        assert proc.returncode == 0

    def test_hard_offline_is_off_by_default(self, monkeypatch):
        monkeypatch.delenv(shim.HARD_OFFLINE_ENV, raising=False)
        assert shim.hard_offline_prefix() is None


def _snapshot(root: Path):
    """Everything under a directory (or a marker when it does not exist)."""
    if not root.exists():
        return ("absent",)
    return tuple(sorted((str(p.relative_to(root)), p.stat().st_mtime_ns)
                        for p in root.rglob("*")))
