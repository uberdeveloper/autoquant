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
    return df.sort_index()[["open", "high", "low", "close", "volume"]]


PROVIDERS = {"yahoo": _fetch_yahoo, "fred": _fetch_fred, "stooq": _fetch_stooq}

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

# ---------------------------------------------------------------- loading

def cache_path(series_id: str) -> Path:
    source, symbol = parse_id(series_id)
    return CACHE_DIR / f"{source}_{re.sub(r'[^A-Za-z0-9]', '_', symbol)}.csv"


def _write_cache(cache: Path, df: pd.DataFrame) -> None:
    """Tmp + replace: an interrupted write can never shadow real data."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".csv.tmp")
    df.to_csv(tmp)
    tmp.replace(cache)


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
            PROVENANCE[entry["id"]] = "cache"
    if df is None:
        try:
            df = PROVIDERS[source](symbol)
            if df.empty:
                # a header-only frame would shadow real data in the cache forever
                raise ValueError("returned no data")
        except Exception as exc:
            if not entry.get("fallback"):
                raise ValueError(f"{entry['id']}: {exc}") from exc
            PROVENANCE[entry["id"]] = f"fallback:{entry['fallback']}"
            print(f"{entry['id']}: {exc} -- falling back to {entry['fallback']}")
            return load(entry["fallback"], start, end)
        _write_cache(cache, df)
        PROVENANCE[entry["id"]] = source

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
    p_fetch.add_argument("--refresh", action="store_true",
                         help="delete any cached copy first and refetch")

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
        if args.refresh:
            cache_path(args.series_id).unlink(missing_ok=True)
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
