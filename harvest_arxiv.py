#!/usr/bin/env python3
"""Harvest arXiv quant-finance papers into the article pipeline.

Single route: the arXiv API (export.arxiv.org/api/query), newest-first,
pages of 100. The paper abstract IS the article body: it is written
straight into the fetch cache (data/pages/, same format as fetch.py) and
rows are marked stage "fetched", so fetch.py skips them entirely.

Contract mirrors harvest.py/harvest_cboe.py: rows append to the same
data/articles.jsonl; slugs are prefixed `arxiv-`. Dedupe is on the
canonical arXiv id via the abs url.

    python3 harvest_arxiv.py                      # newest 100 from q-fin*,econ.EM,econ.GN
    python3 harvest_arxiv.py --limit 500          # deeper pull (hard cap 1000)
    python3 harvest_arxiv.py --categories stat.ML # any arXiv category list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from fetch import page_path
from harvest import entry_row, slugify

ROOT = Path(__file__).resolve().parent
API = "https://export.arxiv.org/api/query"
# arXiv courtesy limit: at most one request every 3 seconds.
UA = "autoquant-harvest/0.1 (research; contact: atmabuddy99@gmail.com)"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
PAGE_SIZE = 100
HARD_CAP = 1000
DEFAULT_CATEGORIES = "q-fin*,econ.EM,econ.GN"
DEFAULT_LIMIT = 100

# The q-fin archive's subcategories, for prefix expansion.
QFIN_SUBCATS = ["CP", "GN", "MF", "PM", "PR", "RM", "ST", "TR"]


def expand_categories(cats: list[str]) -> list[str]:
    """Expand `q-fin*` to all q-fin subcategories; pass the rest through."""
    out: list[str] = []
    for cat in cats:
        cat = cat.strip()
        if cat == "q-fin*":
            out += [f"q-fin.{s}" for s in QFIN_SUBCATS]
        elif cat:
            out.append(cat)
    return out


def search_query(cats: list[str]) -> str:
    return "(" + " OR ".join(f"cat:{c}" for c in cats) + ")"


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def row_from(entry_url: str, title: str, abstract: str, posted: str | None,
             arxiv_id: str) -> dict:
    row = entry_row(entry_url, normalize_ws(title), "arXiv",
                    normalize_ws(abstract), posted, posted or "")
    row["slug"] = "arxiv-" + slugify(normalize_ws(title))
    row["arxiv_id"] = arxiv_id
    return row


def parse_atom(xml_text: str) -> list[dict]:
    """Rows (without page-cache fields) from one Atom API page."""
    root = ET.fromstring(xml_text)
    rows = []
    for entry in root.iter(f"{ATOM}entry"):
        raw_id = (entry.findtext(f"{ATOM}id") or "").strip()
        if "/abs/" not in raw_id:
            continue
        arxiv_id = raw_id.split("/abs/", 1)[1]
        url = f"https://arxiv.org/abs/{arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id}"
        title = entry.findtext(f"{ATOM}title") or ""
        summary = entry.findtext(f"{ATOM}summary") or ""
        published = (entry.findtext(f"{ATOM}published") or "").strip()
        posted = published[:10] or None
        rows.append(row_from(url, title, summary, posted,
                             arxiv_id.split("v")[0] if "v" in arxiv_id
                             else arxiv_id))
    return rows


def write_body_cache(row: dict, abstract: str, pages_dir: Path) -> None:
    """Write the abstract as the article body, in fetch.py's page format.

    Uses the same file name fetch.page_path() computes, resolved under
    pages_dir (which defaults to fetch.py's data/pages)."""
    dest = pages_dir / page_path(row["url"]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"# {row['title']}\n\n"
        f"- source: {row['source']}\n- url: {row['url']}\n"
        f"- posted: {row['posted']}\n\n---\n\n{abstract.strip()}\n"
    )
    row["stage"] = "fetched"
    row["page"] = str(dest)
    row["page_chars"] = dest.stat().st_size


def effective_limit(limit: int) -> int:
    if limit <= 0:
        return DEFAULT_LIMIT
    return min(limit, HARD_CAP)


def get(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help=f"how many papers to harvest (default {DEFAULT_LIMIT}, "
                         f"hard cap {HARD_CAP})")
    ap.add_argument("--categories", default=DEFAULT_CATEGORIES,
                    help="comma-separated arXiv categories; `q-fin*` expands "
                         "to all q-fin subcategories")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between requests (arXiv courtesy: >= 3)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "articles.jsonl")
    ap.add_argument("--pages", type=Path, default=ROOT / "data" / "pages")
    args = ap.parse_args(argv)

    cats = expand_categories(args.categories.split(","))
    if not cats:
        print("no categories given", file=sys.stderr)
        return 1
    limit = effective_limit(args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if args.out.exists():
        with args.out.open() as fh:
            for line in fh:
                if line.strip():
                    seen.add(json.loads(line)["url"])
    print(f"{len(seen)} urls already known")

    session = requests.Session()
    session.headers["User-Agent"] = UA

    query = search_query(cats)
    new = 0
    with args.out.open("a") as fh:
        start = 0
        while new < limit:
            want = min(PAGE_SIZE, limit - new)
            url = (f"{API}?search_query={requests.utils.quote(query)}"
                   f"&sortBy=submittedDate&sortOrder=descending"
                   f"&start={start}&max_results={want}")
            try:
                response_text = get(session, url)
                raw_entries = len(ET.fromstring(response_text).findall(ATOM + "entry"))
                rows = parse_atom(response_text)
            except Exception as exc:
                print(f"api start={start}: {exc}", file=sys.stderr)
                break
            if raw_entries == 0:
                break  # exhausted
            added = 0
            for row in rows:
                if row["url"] in seen:
                    continue
                seen.add(row["url"])
                # The abstract doubles as the body: cache it so fetch.py
                # never needs to scrape the abs page.
                write_body_cache(row, row["blurb"], args.pages)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                added += 1
                new += 1
            print(f"start={start}: {len(rows)} entries, {added} new")
            # A known page does not mean later pages are known. Continue until
            # the API returns a short page or no entries.
            if raw_entries < want:
                break
            start += want
            time.sleep(args.sleep)

    print(f"\n{new} new articles -> {args.out}")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
