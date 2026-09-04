#!/usr/bin/env python3
"""Harvest article links from the Quantocracy quant mashup.

Two routes, because the site exposes history through only one of them:

  live     https://quantocracy.com/ -- the mashup widget, always the latest ~50.
           /page/N/ serves the SAME widget, so it is useless for history.
  backfill /category/daily-wraps/page/N/ -- 10 "Recent Quant Links as of MM/DD/YYYY"
           wrap posts per page, each listing that day's links. This is the archive.

Incremental: urls already in data/articles.jsonl are never re-added, so a daily
cron run only appends what's new.

    python3 harvest.py                  # latest 50 (daily use)
    python3 harvest.py --backfill 20    # ~200 days of history
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "articles.jsonl"
BASE = "https://quantocracy.com"
UA = "quantocracy-lab/0.1 (research backtester; contact: atmabuddy99@gmail.com)"

# "Some Title [Source Blog]" -> ("Some Title", "Source Blog")
TITLE_RE = re.compile(r"^(?P<title>.*?)\s*\[(?P<source>[^\[\]]+)\]\s*$")
# footer: "- 16 hours ago, 9 Aug 2026, 08:08pm -"
DATE_RE = re.compile(r"(\d{1,2}\s+\w{3}\s+\d{4},\s*\d{1,2}:\d{2}[ap]m)", re.I)


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def entry_row(url: str, title: str, source: str | None, blurb: str,
              posted: str | None, posted_raw: str) -> dict:
    """One harvested article, shared by the mashup and daily-wrap routes."""
    return {
        "url": url,
        "title": title,
        "source": source,
        "blurb": blurb,
        "posted": posted,          # publication date == natural OOS boundary
        "posted_raw": posted_raw,
        "slug": slugify(title),
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "harvested",      # harvested -> triaged -> spec -> backtested
    }


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for art in soup.select("article.qo-entry"):
        a = art.select_one("a.qo-title")
        if not a or not a.get("href"):
            continue
        raw = " ".join(a.get_text(" ", strip=True).split())
        m = TITLE_RE.match(raw)
        title = m.group("title") if m else raw
        source = m.group("source") if m else None

        desc_el = art.select_one("summary.qo-description")
        blurb = desc_el.get_text(" ", strip=True) if desc_el else ""

        foot = art.select_one("footer")
        foot_txt = foot.get_text(" ", strip=True) if foot else ""
        dm = DATE_RE.search(foot_txt)
        posted = None
        if dm:
            try:
                posted = datetime.strptime(dm.group(1), "%d %b %Y, %I:%M%p").date().isoformat()
            except ValueError:
                pass

        rows.append(entry_row(a["href"], title, source, blurb, posted, foot_txt))
    return rows


WRAP_DATE_RE = re.compile(r"as-of-(\d{2})(\d{2})(\d{4})")


def get(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_wrap(html: str, posted: str | None) -> list[dict]:
    """Parse one daily-wrap post: its body lists that day's links as plain <a>s."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.entry-content")
    if not body:
        return []
    rows = []
    for a in body.select("a[href]"):
        href = a["href"]
        if "quantocracy.com" in href:  # nav / self-links
            continue
        raw = " ".join(a.get_text(" ", strip=True).split())
        # wrap posts truncate the title, so the "[Source]" bracket is often cut off
        m = TITLE_RE.match(raw)
        title, source = (m.group("title"), m.group("source")) if m else (raw.rstrip(" ["), None)
        if not title:
            continue
        rows.append(entry_row(href, title, source, "", posted, posted or ""))
    return rows


def wrap_urls(session: requests.Session, page: int) -> list[str]:
    url = f"{BASE}/category/daily-wraps/" if page == 1 else f"{BASE}/category/daily-wraps/page/{page}/"
    soup = BeautifulSoup(get(session, url), "html.parser")
    seen, out = set(), []
    for a in soup.select("a[href]"):
        href = a["href"]
        if "recent-quant-links" in href and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0,
                    help="pages of /category/daily-wraps/ to walk (10 days per page)")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

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

    new = 0
    with args.out.open("a") as fh:

        def emit(rows: list[dict], label: str) -> None:
            nonlocal new
            added = 0
            for row in rows:
                if row["url"] in seen:
                    continue
                seen.add(row["url"])
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                added += 1
            new += added
            print(f"{label}: {len(rows)} entries, {added} new")

        if not args.backfill:
            emit(parse_page(get(session, BASE + "/")), "mashup")
        else:
            for page in range(1, args.backfill + 1):
                try:
                    urls = wrap_urls(session, page)
                except Exception as exc:  # keep long backfills alive
                    print(f"wraps p{page}: {exc}", file=sys.stderr)
                    continue
                if not urls:
                    print(f"wraps p{page}: no wrap posts — stopping")
                    break
                for wu in urls:
                    dm = WRAP_DATE_RE.search(wu)
                    posted = f"{dm.group(3)}-{dm.group(1)}-{dm.group(2)}" if dm else None
                    try:
                        emit(parse_wrap(get(session, wu), posted), f"wrap {posted or wu}")
                    except Exception as exc:
                        print(f"{wu}: {exc}", file=sys.stderr)
                    time.sleep(args.sleep)

    print(f"\n{new} new articles -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
