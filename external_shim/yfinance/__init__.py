"""Stub ``yfinance`` served from the project's own price cache (issue #26).

External strategy repos call ``yf.download(...)`` and
``yf.Ticker(...).history(...)``.  Without a shim those calls hit Yahoo
(rate limits, flakiness, non-deterministic results).  With this package on
``PYTHONPATH`` they are served from our catalog cache instead, offline.

Frame shape (documented, pinned by ``tests/test_shim.py``):

* the normalized frame from ``catalog.load`` -- lowercase OHLCV, always a
  ``close`` column, ascending ``DatetimeIndex``, filtered to
  ``[start, end]``;
* plus a ``Close`` alias column, because external code overwhelmingly reads
  ``Close``;
* flat columns -- deliberately NOT yfinance's MultiIndex, which would break
  the simple ``df["Close"]`` access most notebooks use.

**Daily bars only.**  The catalog holds daily frames; there is no intraday
or weekly data behind it.  The ``interval`` argument is therefore *accepted
and ignored*: a request for ``interval="1wk"`` is served with the same daily
bars as ``interval="1d"``.  Weekly/intraday bars are never synthesized --
that would fabricate data.

A request that the cache cannot serve raises
``external_shim._core.DataNotAvailable`` (re-exported here) instead of
reaching Yahoo.
"""
from __future__ import annotations

import re

import pandas as pd

try:  # package layout: imported as external_shim.yfinance
    from .._core import DataNotAvailable, load_frame
except ImportError:  # flat layout: external_shim/ itself is on sys.path
    from _core import DataNotAvailable, load_frame

__all__ = ["download", "Ticker", "DataNotAvailable"]

#: yfinance-style period units the shim understands.  "mo" is months (as in
#: yfinance); minutes are not supported -- the catalog is daily.
_PERIOD_RE = re.compile(r"^(\d+)\s*(d|wk|w|mo|y)$")


def _period_bounds(period, interval=None) -> tuple[object, object]:
    """(start, end) for a yfinance-style ``period`` argument.

    Supported: ``Nd`` (``5d`` -- the last N *trading* days), ``Nwk``/``Nw``
    (weeks -- the two spellings mean the same window), ``Nmo``, ``Ny``, plus
    ``ytd`` (start of the current year) and ``max`` (the whole cached
    series).  Anything else raises ``ValueError``: an unparseable period
    must not silently widen the window to "everything".

    ``interval`` is accepted for call compatibility and ignored -- only
    daily bars exist in the catalog.
    """
    if period is None:
        return None, None
    text = str(period).strip().lower()
    if text in ("", "max"):
        return None, None
    end = pd.Timestamp.now().normalize()
    if text == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1), end
    match = _PERIOD_RE.match(text)
    if match is None:
        raise ValueError(
            f"unsupported period {period!r}: the offline shim serves daily "
            "bars only -- use one of '1d', '5d', '1w', '1wk', '1mo', '1y', "
            "'ytd', 'max'")
    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"unsupported period {period!r}: count must be > 0")
    unit = match.group(2)
    if unit == "d":
        # The last N trading days (yfinance semantics); bdate_range rolls a
        # weekend/holiday end back to the last business day.
        start = pd.bdate_range(end=end, periods=count)[0]
    elif unit in ("w", "wk"):
        # 'w' and 'wk' both mean weeks -- a value the regex accepts must
        # not fall through to the year branch.
        start = end - pd.Timedelta(weeks=count) + pd.Timedelta(days=1)
    elif unit == "mo":
        start = end - pd.DateOffset(months=count) + pd.Timedelta(days=1)
    else:  # "y"
        start = end - pd.DateOffset(years=count) + pd.Timedelta(days=1)
    return start, end


def download(tickers, start=None, end=None, period=None, interval="1d",
             progress=False, auto_adjust=True, actions=False, group_by="column",
             threads=True, ignore_tz=None, repair=False, keepna=False,
             prepost=False, proxy=None, rounding=False, timeout=10, session=None,
             multi_level_index=True,
             **kwargs) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Serve cached prices for one or more tickers.

    A single ticker returns the flat frame (see the module docstring); a
    list or dict of tickers returns a ``{ticker: frame}`` mapping.
    """
    if isinstance(tickers, dict):
        names = list(tickers)
        single = False
    elif isinstance(tickers, (list, tuple, set)):
        names = list(tickers)
        single = False
    else:
        names = [tickers]
        single = True
    if not names:
        raise ValueError("download() needs at least one ticker")
    if start is None and end is None and period:
        start, end = _period_bounds(period, interval)
    frames = {name: load_frame("yahoo", str(name), start, end) for name in names}
    if single:
        return frames[names[0]]
    return frames


class Ticker:
    """Stub ticker -- ``history()`` is backed by the same cache as download()."""

    def __init__(self, ticker, session=None, proxy=None):
        self.ticker = str(ticker)

    def history(self, period=None, start=None, end=None, interval="1d",
                actions=False, auto_adjust=True,
                **kwargs) -> pd.DataFrame:
        if start is None and end is None and period:
            start, end = _period_bounds(period, interval)
        return load_frame("yahoo", self.ticker, start, end)

    def download(self, period=None, start=None, end=None, interval="1d",
                 **kwargs) -> pd.DataFrame:
        return self.history(period=period, start=start, end=end,
                            interval=interval, **kwargs)
