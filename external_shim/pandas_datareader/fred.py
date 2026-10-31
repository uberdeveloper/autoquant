"""Stub FRED reader -- the shim never fetches from FRED (#26).

``catalog.py`` imports ``FredReader`` from ``pandas_datareader.fred`` at
module load time.  When the shim's ``pandas_datareader`` shadows the
installed package, this module keeps that import working while making sure
no FRED request ever leaves the box: the reader raises
``DataNotAvailable`` instead of downloading.
"""
from __future__ import annotations

try:  # package layout: imported as external_shim.pandas_datareader.fred
    from .._core import DataNotAvailable
except ImportError:  # flat layout: external_shim/ itself is on sys.path
    from _core import DataNotAvailable

__all__ = ["FredReader", "DataNotAvailable"]


class FredReader:
    """Offline stand-in for ``pandas_datareader.fred.FredReader``."""

    def __init__(self, series, start=None, end=None, **kwargs):
        self.series = series
        self.start = start
        self.end = end

    def read(self):
        raise DataNotAvailable(
            f"{self.series}: the offline shim does not fetch FRED series. "
            "Warm the cache with `python3 catalog.py fetch fred:"
            f"{self.series}` on a networked machine, or point "
            "AUTOQUANT_CATALOG_DIR at a populated cache.")
