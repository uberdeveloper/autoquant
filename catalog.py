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
