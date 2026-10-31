"""Shared plumbing for the offline data shim (issue #26).

The shim exists so that external strategy-repo code -- which normally calls
``yfinance`` / ``pandas_datareader`` and therefore phones out to Yahoo/FRED --
runs against *our* price cache instead.  It is injected by putting
``external_shim/`` on ``PYTHONPATH``; nothing here is installed.

Two rules govern this module:

1. **Never re-implement the data layer.**  Every load goes through the
   project's own :mod:`catalog` module -- ``catalog.load`` (normalization,
   filtering, fallback, caching), ``catalog.parse_id`` (id parsing) and
   ``catalog.cache_path`` (cache file names).  The shim only decides *which*
   catalog id a request means and *whether* a provider fetch is allowed.
2. **Never phone out by default.**  The shim is offline: a request that
   cannot be served from the local cache raises :class:`DataNotAvailable`
   instead of reaching a provider.  The flag ``AUTOQUANT_SHIM_OFFLINE``
   (default ``1``) turns that guarantee on; ``0`` opts back into live
   fetches (used only for local debugging -- it makes runs
   non-deterministic again).

The cache root is injectable through ``AUTOQUANT_CATALOG_DIR`` (default:
``<project root>/data/catalog``) so a run-copy or a warmed cache can be
targeted without editing code.

Isolation contract (owned by #27, honoured here): this stage never writes
into ``data/repos/`` and never mutates an original clone -- execution happens
in a run-copy, and the shim only ever *reads* the catalog cache (plus the
cache write ``catalog.load`` itself performs on a live fetch).

Isolation from the *external* repo matters just as much: the shim loads the
project's ``catalog.py`` by absolute path under a private module name
(``_autoquant_catalog``) and never touches ``sys.path``.  The external
repo's own ``catalog.py`` / ``backtest.py`` / ``llm.py`` therefore keep
resolving to *its* modules, not to ours.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import pandas as pd

SHIM_DIR = Path(__file__).resolve().parent

CATALOG_DIR_ENV = "AUTOQUANT_CATALOG_DIR"
OFFLINE_ENV = "AUTOQUANT_SHIM_OFFLINE"
PROJECT_ROOT_ENV = "AUTOQUANT_PROJECT_ROOT"

#: Private module name for the project's own ``catalog.py``.  Loading it
#: under a private name keeps it out of the global import namespace, so a
#: foreign module called ``catalog`` (the external repo may ship one) can
#: never shadow it -- and ours never shadows theirs.
CATALOG_MODULE_NAME = "_autoquant_catalog"

#: Coarse lock around the shim's global state.  ``catalog.PROVIDERS`` and
#: ``catalog.CACHE_DIR`` are module globals of the *catalog* module: the
#: shim blocks the providers and redirects the cache root around every
#: load, so two concurrent shim calls would otherwise interleave their
#: block/restore -- one call could then reach a provider while the other
#: believes it is offline, or serve the wrong cache root.
#:
#: It is an ``RLock`` (re-entrant) because the recursion guard re-enters
#: ``load_frame`` from the *same* thread: a provider that calls back into
#: the shim must fail cleanly, not deadlock.  The lock is deliberately
#: coarse -- no per-symbol locking.
_LOAD_LOCK = threading.RLock()


class _ProviderAbort(BaseException):
    """Abort a provider call without letting ``catalog.load`` fall back.

    ``catalog.load`` catches every ``Exception`` raised by a provider and
    then tries the series' fallback (a *live* fetch of the mirror).  That is
    wrong for the shim's recursion guard: a provider that calls back into
    the shim must fail the request, not trigger a mirror fetch.  This
    ``BaseException`` escapes ``catalog.load``'s ``except Exception`` and is
    turned back into :class:`DataNotAvailable` by :func:`_load`.
    """


class DataNotAvailable(RuntimeError):
    """The offline shim could not serve a request from the local cache.

    Raised instead of a live provider fetch.  It is a clean, catchable error
    (never a hang, never a bare traceback): #27 turns it into a verdict row
    for the external artifact.  The message names the symbol and says what to
    do about it.
    """


# ------------------------------------------------------------------ layout

def project_root() -> Path:
    """The project root that owns ``catalog.py``/``catalog.json``."""
    raw = os.environ.get(PROJECT_ROOT_ENV)
    if raw:
        return Path(raw).expanduser()
    for cand in (SHIM_DIR.parent, *SHIM_DIR.parents):
        if (cand / "catalog.py").is_file() and (cand / "catalog.json").is_file():
            return cand
    return SHIM_DIR.parent


def catalog_dir() -> Path:
    """The cache root the shim serves from (injectable, never assumed)."""
    raw = os.environ.get(CATALOG_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return project_root() / "data" / "catalog"


def offline() -> bool:
    """True when provider fetches must raise instead of executing."""
    raw = os.environ.get(OFFLINE_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _fred_reader() -> type:
    """The shim's offline ``FredReader`` -- the shim never fetches FRED.

    The project's ``catalog.py`` imports ``FredReader`` at module load
    time.  In a shimmed process that import already resolves to the stub;
    in the project's own process it resolves to the *real* reader, which
    would phone out to FRED.  The shim therefore always installs its own
    stub.
    """
    try:
        from external_shim.pandas_datareader.fred import FredReader
    except ImportError:  # flat layout: external_shim/ itself is on sys.path
        from pandas_datareader.fred import FredReader
    return FredReader


def _catalog() -> ModuleType:
    """The project's own catalog module -- imported, never re-implemented.

    Loaded by absolute path under a private module name so that (a) a
    foreign module called ``catalog`` in the external repo cannot shadow
    it, and (b) our module cannot shadow the external repo's own modules
    (``sys.path`` is left untouched -- that is the isolation guarantee).
    """
    module = sys.modules.get(CATALOG_MODULE_NAME)
    if module is not None:
        return module
    path = project_root() / "catalog.py"
    if not path.is_file():
        raise DataNotAvailable(
            f"catalog.py not found under {project_root()}: the shim cannot "
            "serve data without the project's own catalog module")
    spec = importlib.util.spec_from_file_location(CATALOG_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise DataNotAvailable(f"cannot load the catalog module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[CATALOG_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(CATALOG_MODULE_NAME, None)
        raise
    # The shim never fetches from FRED -- swap the reader for the stub.
    module.FredReader = _fred_reader()
    return module


def catalog_module() -> ModuleType:
    """The project's catalog module as loaded by the shim (public seam)."""
    return _catalog()


# ------------------------------------------------------------------ errors

def unavailable_message(symbol: str, series_id: str, reason: str | None = None) -> str:
    """Actionable text: names the symbol and says how to fix the miss."""
    msg = (f"{symbol}: no cached data for {series_id!r} in {catalog_dir()}. "
           "The offline shim never calls Yahoo/FRED. "
           f"Warm the cache with `python3 catalog.py fetch {series_id}` on a "
           f"networked machine, or point {CATALOG_DIR_ENV} at a populated cache.")
    if reason:
        msg += f" ({reason})"
    return msg


# ------------------------------------------------------------------ loading

@contextmanager
def _providers_blocked(cat: ModuleType) -> Iterator[None]:
    """Make every provider call raise -- the deterministic offline guarantee."""
    original = cat.PROVIDERS

    def _blocked(source):
        def _raise(symbol, *args, **kwargs):
            raise DataNotAvailable(
                f"provider {source!r} is blocked for {symbol!r}: the shim is "
                f"offline ({OFFLINE_ENV}=1)")
        return _raise

    cat.PROVIDERS = {name: _blocked(name) for name in original}
    try:
        yield
    finally:
        cat.PROVIDERS = original


_FETCH_DEPTH = 0


@contextmanager
def provider_fetch_scope():
    """Mark that a live provider fetch is in flight (recursion guard)."""
    global _FETCH_DEPTH
    _FETCH_DEPTH += 1
    try:
        yield
    finally:
        _FETCH_DEPTH -= 1


def in_provider_fetch() -> bool:
    """True while a provider fetch is in flight."""
    return _FETCH_DEPTH > 0


def real_yfinance() -> ModuleType | None:
    """The installed yfinance package, bypassing this shim's shadow of it.

    Only used when the shim is explicitly put back online: the catalog's
    yahoo provider does ``import yfinance`` and must reach the real package,
    not this stub (which would recurse into the shim forever).  Returns
    ``None`` when no usable real package can be loaded -- the caller then
    fails cleanly instead of recursing.
    """
    shadow = (SHIM_DIR / "yfinance").resolve()
    for entry in sys.path:
        if not entry:
            continue
        pkg = Path(entry) / "yfinance"
        if not (pkg / "__init__.py").is_file():
            continue
        try:
            if pkg.resolve() == shadow:
                continue
        except OSError:
            continue
        module = sys.modules.pop("_shim_real_yfinance", None)
        if module is not None:
            del module
        try:
            spec = importlib.util.spec_from_file_location(
                "_shim_real_yfinance", pkg / "__init__.py",
                submodule_search_locations=[str(pkg)])
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules["_shim_real_yfinance"] = module
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop("_shim_real_yfinance", None)
            continue
    return None


def _candidates(cat: ModuleType, series_id: str) -> list[str]:
    """The catalog ids that may serve this request (primary + its mirror)."""
    out = [series_id]
    entry = cat.by_id(series_id)
    fallback = entry.get("fallback") if entry else None
    if fallback and fallback not in out:
        out.append(fallback)
    return out


def _shape(df: pd.DataFrame) -> pd.DataFrame:
    """The documented frame shape: normalized OHLCV + a ``Close`` alias.

    ``catalog.load`` already returns lowercase columns with ``close`` and an
    ascending DatetimeIndex.  External code overwhelmingly reads ``Close``,
    so the alias is added here; the frame stays flat (no MultiIndex).
    """
    out = df.copy()
    if "close" in out.columns:
        out["Close"] = out["close"]
    return out


def _cache_is_usable(path: Path) -> bool:
    """True when a cache file holds a servable frame (non-empty, a close).

    Checked *before* ``catalog.load`` sees the file: ``catalog.load`` deletes
    a truncated/header-only cache and refetches, which would mutate the
    cache the shim is only supposed to read.
    """
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return False
    return not df.empty and "close" in df.columns


def _guard_provider(fetch):
    """Wrap a provider so a shim-level abort is not swallowed by the catalog.

    ``catalog.load`` catches ``Exception`` from a provider and falls back to
    the series' mirror -- a *live* fetch.  A provider that calls back into
    the shim (the recursion guard) must abort the whole request instead.
    """

    def _wrapped(symbol, *args, **kwargs):
        try:
            return fetch(symbol, *args, **kwargs)
        except DataNotAvailable as exc:
            raise _ProviderAbort(str(exc)) from exc

    return _wrapped


def _load(cat: ModuleType, series_id: str, start: str | None,
          end: str | None) -> pd.DataFrame:
    symbol = series_id.split(":", 1)[-1]
    if offline():
        with _providers_blocked(cat):
            try:
                return _shape(cat.load(series_id, start, end))
            except Exception as exc:
                raise DataNotAvailable(
                    unavailable_message(symbol, series_id, str(exc))) from exc
    # Online mode is *still* cache-first: the real provider is resolved
    # lazily, i.e. only when ``catalog.load`` actually asks for it -- which
    # happens on a cache miss.  A warm cache must never trigger a provider
    # resolution (that is what made an online cache hit fail).
    saved = sys.modules.get("yfinance")
    saved_providers = cat.PROVIDERS
    installed = False

    def _resolve_real_yfinance() -> None:
        """Install the real yfinance package -- at provider-call time only."""
        nonlocal installed
        real = real_yfinance()
        if real is None:
            raise DataNotAvailable(
                f"{symbol}: the installed yfinance package could not be "
                "loaded, so the online provider cannot be used")
        sys.modules["yfinance"] = real
        installed = True

    def _online_provider(name: str,
                         fetch: Callable[..., pd.DataFrame]
                         ) -> Callable[..., pd.DataFrame]:
        def _call(fetch_symbol: str, *args: object,
                  **kwargs: object) -> pd.DataFrame:
            if name == "yahoo" and fetch is cat._fetch_yahoo:
                # The catalog's yahoo provider does ``import yfinance``:
                # without the real package it would call *this* shim and
                # recurse forever.
                _resolve_real_yfinance()
            return fetch(fetch_symbol, *args, **kwargs)
        return _call

    cat.PROVIDERS = {name: _guard_provider(_online_provider(name, fetch))
                     for name, fetch in saved_providers.items()}
    try:
        with provider_fetch_scope():
            return _shape(cat.load(series_id, start, end))
    except DataNotAvailable:
        raise
    except _ProviderAbort as exc:
        raise DataNotAvailable(
            unavailable_message(symbol, series_id, str(exc))) from exc
    except Exception as exc:
        raise DataNotAvailable(
            unavailable_message(symbol, series_id, str(exc))) from exc
    finally:
        cat.PROVIDERS = saved_providers
        if installed:
            if saved is not None:
                sys.modules["yfinance"] = saved
            else:
                sys.modules.pop("yfinance", None)


def load_frame(source: str, symbol: str, start: str | None = None,
               end: str | None = None) -> pd.DataFrame:
    """Serve one series from the local cache. Never phones out when offline.

    ``source`` is a catalog source ("yahoo", "fred", "stooq"); ``symbol`` is
    the bare symbol.  Raises :class:`DataNotAvailable` when the request
    cannot be served locally.

    The whole load -- the provider block, the cache-root redirect and the
    catalog call -- runs under :data:`_LOAD_LOCK`: those are module globals
    of the catalog module, so concurrent shim calls must not interleave
    them.
    """
    cat = _catalog()
    with _LOAD_LOCK:
        previous = cat.CACHE_DIR
        cat.CACHE_DIR = catalog_dir()
        try:
            return _load_series(cat, source, symbol, start, end)
        finally:
            cat.CACHE_DIR = previous


def _load_series(cat: ModuleType, source: str, symbol: str,
                 start: str | None, end: str | None) -> pd.DataFrame:
    symbol = str(symbol)
    try:
        source, symbol = cat.parse_id(f"{source}:{symbol}")
    except ValueError as exc:
        raise DataNotAvailable(
            unavailable_message(symbol, f"{source}:{symbol}", str(exc))) from exc
    series_id = f"{source}:{symbol}"
    if in_provider_fetch():
        # A provider is already fetching: serving this request would call
        # back into the shim and recurse.  Fail cleanly instead.
        raise DataNotAvailable(
            f"{symbol}: a provider fetch is already in flight -- the shim "
            f"refuses to serve {series_id!r} recursively")
    malformed = False
    for candidate in _candidates(cat, series_id):
        path = cat.cache_path(candidate)
        if not path.exists():
            continue
        if not _cache_is_usable(path):
            # A truncated/header-only cache: do not let ``catalog.load``
            # delete it (the shim only reads the cache) -- try the mirror.
            malformed = True
            continue
        return _load(cat, candidate, start, end)
    if offline() and malformed:
        raise DataNotAvailable(
            unavailable_message(symbol, series_id,
                                "the cached file is empty or malformed"))
    return _load(cat, series_id, start, end)
