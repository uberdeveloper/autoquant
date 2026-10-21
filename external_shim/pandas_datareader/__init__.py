"""Stub ``pandas_datareader`` served from the project's own cache (#26).

External repos that pull macro series call
``pandas_datareader.data.DataReader("DGS10", "fred")``; with this package
on ``PYTHONPATH`` the call is routed to the matching catalog id and served
from the local cache, offline.
"""
from __future__ import annotations

from . import data
from .data import DataReader

__all__ = ["data", "DataReader"]
