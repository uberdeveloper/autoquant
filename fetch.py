#!/usr/bin/env python3
"""Stage [2] FETCH -- pull full article text for harvested links.

Triage needs the article body, not the blurb: the rubric explicitly says score
`null` when the text is truncated or paywalled, so guessing from the title is
not an option.

Polite by construction, per the README's scope guardrails:
  - robots.txt is checked per host and cached
  - one request per host at a time, --sleep seconds apart
  - the agent identifies itself in the UA

Cached: a url whose page file already exists is never re-fetched. Articles are
advanced harvested -> fetched (or -> fetch_failed, recorded not deleted).

    python3 fetch.py                # every harvested article
    python3 fetch.py --limit 5      # smoke test
    python3 fetch.py --retry-failed # another go at fetch_failed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.robotparser as robotparser
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
ARTICLES = ROOT / "data" / "articles.jsonl"
PAGES = ROOT / "data" / "pages"
UA = "quantocracy-lab/0.1 (research backtester; contact: atmabuddy99@gmail.com)"

# Chrome/blog furniture that is never article body.
STRIP = ["script", "style", "nav", "header", "footer", "aside", "form",
         "noscript", "iframe", "svg", "button"]
# Tried in order; first hit with enough text wins. Falls back to <body>.
CANDIDATES = ["article", "main", '[role="main"]', ".post-content", ".entry-content",
              ".post-body", ".article-body", ".markdown-body", "#content"]
MIN_CHARS = 600  # below this the extraction almost certainly missed the body


def page_path(url: str) -> Path:
    return PAGES / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.md"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite -- a crash mid-run must not truncate the queue."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


class Robots:
    """robots.txt per host, fetched once. Unreachable robots.txt = allowed."""

    def __init__(self, session: requests.Session):
        self.session = session
        self.cache: dict[str, robotparser.RobotFileParser | None] = {}

    def allows(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self.cache:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(urljoin(url, "/robots.txt"), timeout=15)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:
                    rp = None
            except requests.RequestException:
                rp = None
            self.cache[host] = rp
        rp = self.cache[host]
        return True if rp is None else rp.can_fetch(UA, url)


def extract(html: str) -> tuple[str, str]:
    """(title, body text). Picks the densest plausible container."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    best = ""
    for sel in CANDIDATES:
        for node in soup.select(sel):
            text = node.get_text("\n", strip=True)
            if len(text) > len(best):
                best = text
    if len(best) < MIN_CHARS and soup.body:
        body = soup.body.get_text("\n", strip=True)
        if len(body) > len(best):
            best = body

    # Collapse the blank-line spray that get_text leaves behind.
    lines = [ln.strip() for ln in best.splitlines()]
    return title, "\n".join(ln for ln in lines if ln)


def fetch_one(session: requests.Session, robots: Robots, row: dict) -> dict:
    url = row["url"]
    dest = page_path(url)

    if dest.exists() and dest.stat().st_size > MIN_CHARS:
        row["stage"] = "fetched"
        row["page"] = str(dest.relative_to(ROOT))
        row["page_chars"] = dest.stat().st_size
        row.pop("fetch_error", None)
        return row

    if not robots.allows(url):
        row["stage"] = "fetch_failed"
        row["fetch_error"] = "robots.txt disallows"
        return row

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        row["stage"] = "fetch_failed"
        row["fetch_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row

    title, text = extract(r.text)
    if len(text) < MIN_CHARS:
        # Short body is the paywall / JS-rendered signature. Record it and let
        # triage return score:null rather than pretending we have the article.
        row["stage"] = "fetch_failed"
        row["fetch_error"] = f"extracted only {len(text)} chars (paywall or JS-rendered?)"
        return row

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"# {row['title']}\n\n"
        f"- source: {row['source']}\n- url: {url}\n- posted: {row['posted']}\n"
        f"- page_title: {title}\n\n---\n\n{text}\n"
    )
    row["stage"] = "fetched"
    row["page"] = str(dest.relative_to(ROOT))
    row["page_chars"] = dest.stat().st_size
    row.pop("fetch_error", None)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between requests to the same host")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-attempt rows in stage fetch_failed")
    args = ap.parse_args()

    rows = load(ARTICLES)
    if not rows:
        print(f"no articles in {ARTICLES} -- run harvest.py first", file=sys.stderr)
        return 1

    wanted = {"harvested"} | ({"fetch_failed"} if args.retry_failed else set())
    todo = [r for r in rows if r.get("stage") in wanted]
    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("nothing to fetch")
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = UA
    robots = Robots(session)
    last_hit: dict[str, float] = defaultdict(float)

    ok = failed = 0
    for i, row in enumerate(todo, 1):
        host = urlparse(row["url"]).netloc
        wait = args.sleep - (time.monotonic() - last_hit[host])
        if wait > 0:
            time.sleep(wait)
        last_hit[host] = time.monotonic()

        fetch_one(session, robots, row)
        if row["stage"] == "fetched":
            ok += 1
            print(f"[{i}/{len(todo)}] ok    {row['page_chars']:>7,}c  {row['title'][:60]}")
        else:
            failed += 1
            print(f"[{i}/{len(todo)}] FAIL  {row['fetch_error'][:50]:<50}  {row['title'][:40]}")

        save(ARTICLES, rows)  # checkpoint every article; the run is interruptible

    print(f"\nfetched {ok}, failed {failed} -> {PAGES.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
