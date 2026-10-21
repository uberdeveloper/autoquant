"""Offline data shim for external strategy repos (issue #26).

This directory is NOT installed and NOT a package to import from project
code -- it is put on ``PYTHONPATH`` of an external execution so that
``import yfinance`` / ``import pandas_datareader`` resolve to the stubs
here instead of the live providers.  See ``shim.py`` for the seam (#27
launches with it) and ``external_shim/_core.py`` for the shared plumbing.
"""
