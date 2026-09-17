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
        try:
            e = resolve(args.series_id)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
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
