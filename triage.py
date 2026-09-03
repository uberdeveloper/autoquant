#!/usr/bin/env python3
"""Stage [3] TRIAGE -- score each fetched article on "can this be backtested?"

Runs triage_prompt.md over every fetched article via the `claude` CLI in
headless mode (-p), so it uses your existing Claude Code auth -- no
ANTHROPIC_API_KEY, no `ant auth login`.

Routing follows the rubric in triage_prompt.md:
  score 4-5  -> triaged     (proceeds to EXTRACT)
  score 3    -> borderline  (human review)
  score 0-2  -> rejected    (recorded, not deleted)
  score null -> untriageable (text unavailable)

Results land in data/triage.jsonl, one object per article, keyed by url.
Re-running only scores articles not already in that file (--rescore overrides).

    python3 triage.py --limit 5          # calibrate against your own judgement
    python3 triage.py                    # the whole fetched queue
    python3 triage.py --rescore          # re-score everything (rubric changed)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTICLES = ROOT / "data" / "articles.jsonl"
TRIAGE = ROOT / "data" / "triage.jsonl"
PROMPT = ROOT / "triage_prompt.md"

MAX_ARTICLE_CHARS = 40_000  # a rubric decision needs the rules, not every word
CLI_TIMEOUT = 300

STAGE_FOR = {5: "triaged", 4: "triaged", 3: "borderline",
             2: "rejected", 1: "rejected", 0: "rejected"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def strip_body(page: Path) -> str:
    """Drop the header fetch.py wrote -- the model gets it as JSON fields."""
    text = page.read_text()
    _, _, body = text.partition("\n---\n\n")
    return (body or text).strip()


def build_prompt(rubric: str, row: dict, body: str) -> str:
    if len(body) > MAX_ARTICLE_CHARS:
        body = body[:MAX_ARTICLE_CHARS] + "\n\n[...truncated for length...]"
    payload = {"title": row["title"], "source": row["source"], "url": row["url"],
               "posted": row["posted"], "text": body}
    return (
        f"{rubric}\n\n"
        "---\n\nHere is the article to triage. Return ONLY the JSON object "
        "described above -- no prose, no markdown fences.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )


def parse_json(text: str) -> dict:
    """The rubric asks for bare JSON; tolerate a fenced or prose-wrapped reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:start + end - start + 1])
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


def score_one(row: dict, rubric: str, model: str | None) -> dict:
    page = ROOT / row["page"]
    cmd = ["claude", "-p"] + (["--model", model] if model else [])
    prompt = build_prompt(rubric, row, strip_body(page))

    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"url": row["url"], "error": f"claude CLI timed out after {CLI_TIMEOUT}s"}
    if proc.returncode != 0:
        return {"url": row["url"], "error": f"claude exited {proc.returncode}: {proc.stderr[:200]}"}

    try:
        result = parse_json(proc.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"url": row["url"], "error": f"unparseable reply: {exc}"[:300]}

    result["url"] = row["url"]
    result["title"] = row["title"]
    result["source"] = row["source"]
    result["posted"] = row["posted"]
    result["slug"] = row["slug"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent claude calls")
    ap.add_argument("--model", default=None,
                    help="override the model (default: your session's)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score articles already in triage.jsonl")
    args = ap.parse_args()

    rubric = PROMPT.read_text()
    rows = load(ARTICLES)
    scored = load(TRIAGE)
    seen = set() if args.rescore else {r["url"] for r in scored}

    todo = [r for r in rows if r.get("stage") == "fetched" and r["url"] not in seen]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to triage -- run fetch.py, or pass --rescore")
        return 0

    print(f"triaging {len(todo)} articles ({args.jobs} at a time)\n")
    by_url = {r["url"]: r for r in rows}
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(score_one, r, rubric, args.model): r for r in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            row = by_url[res["url"]]

            if "error" in res:
                row["stage"] = "triage_failed"
                row["triage_error"] = res["error"]
                print(f"[{i}/{len(todo)}] ERR  {res['error'][:60]}")
                continue

            score = res.get("score")
            row["stage"] = "untriageable" if score is None else STAGE_FOR.get(score, "rejected")
            row["triage_score"] = score
            row.pop("triage_error", None)
            print(f"[{i}/{len(todo)}] {str(score):>4}  {row['stage']:<12} "
                  f"{res.get('asset_class', '?'):<14} {row['title'][:44]}")

    # Replace prior entries for re-scored urls rather than appending duplicates.
    fresh = {r["url"] for r in results}
    save(TRIAGE, [r for r in scored if r["url"] not in fresh] + results)
    save(ARTICLES, rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("stage", "?")] = counts.get(r.get("stage", "?"), 0) + 1
    print("\nstages:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"scores -> {TRIAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
