"""Stub ``pandas_datareader.data.DataReader`` served from our cache (#26).

The returned value is the same normalized frame the catalog serves (lowercase
columns with ``close``, ascending ``DatetimeIndex``, filtered to
``[start, end]``) plus the ``Close`` alias -- a flat frame, not a
MultiIndex.  FRED series therefore come back with a ``close`` column rather
than a column named after the series.
"""
from __future__ import annotations

import pandas as pd

try:  # package layout: imported as external_shim.pandas_datareader.data
    from .._core import DataNotAvailable, load_frame
except ImportError:  # flat layout: external_shim/ itself is on sys.path
    from _core import DataNotAvailable, load_frame

__all__ = ["DataReader", "DataNotAvailable"]


def DataReader(symbols, data_source=None, start=None, end=None,
               **kwargs) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Serve cached prices for ``symbols`` from the given catalog source.

    ``data_source`` is a catalog source ("yahoo", "fred", "stooq").  A list
    of symbols returns a ``{symbol: frame}`` mapping.
    """
    source = str(data_source or "yahoo").strip().lower()
    if isinstance(symbols, (list, tuple, set)):
        return {sym: DataReader(sym, data_source, start, end, **kwargs)
                for sym in symbols}
    return load_frame(source, str(symbols), start, end)
