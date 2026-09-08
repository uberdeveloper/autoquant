#!/usr/bin/env python3
"""Batch backtest runner -- every spec, isolated process, hard timeout.

Each spec runs as its own `backtest.py` subprocess with a timeout cap, so a
hung data download or a pathological spec can never block the batch. A slug
with a leaderboard row is skipped, so re-running resumes where the last run
stopped. Failures are appended to results/failed.jsonl; the batch moves on.

    python3 backtest_batch.py                 # every specs/*.yaml not yet run
    python3 backtest_batch.py --jobs 4
    python3 backtest_batch.py --timeout 300
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SPECS = ROOT / "specs"
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FAILED = ROOT / "results" / "failed.jsonl"
DEFAULT_TIMEOUT = 600


def run_one(spec: Path, timeout: int) -> dict:
    """One spec in an isolated process. Never raises."""
    cmd = ["uv", "run", "python", "backtest.py", str(spec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"spec": str(spec), "error": f"timed out after {timeout}s"}
    if proc.returncode != 0:
        return {"spec": str(spec), "error": f"exit {proc.returncode}: {proc.stderr[-300:]}"}
    return {"spec": str(spec), "output": proc.stdout}


def already_run(leaderboard: Path) -> set[str]:
    """Slugs with a leaderboard row -- running them again is a no-op."""
    if not leaderboard.exists():
        return set()
    return {json.loads(line)["slug"]
            for line in leaderboard.read_text().splitlines() if line.strip()}


def record_failure(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(result) + "\n")


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="concurrent specs")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="per-spec subprocess cap in seconds")
    ap.add_argument("--specs", type=Path, default=SPECS)
    args = ap.parse_args(argv)

    done = already_run(LEADERBOARD)
    specs = []
    for path in sorted(args.specs.glob("*.yaml")):
        slug = yaml.safe_load(path.read_text())["meta"]["slug"]
        if slug not in done:
            specs.append(path)
    if not specs:
        print(f"nothing to backtest in {args.specs} -- all specs have leaderboard rows")
        return 1

    print(f"backtesting {len(specs)} specs "
          f"({args.jobs} at a time, {args.timeout}s cap each)\n")

    ok = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, s, args.timeout): s for s in specs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            if "error" in result:
                failed += 1
                record_failure(result, FAILED)
                print(f"[{i}/{len(specs)}] FAIL  {result['error'][:60]:<60}  {result['spec']}")
            else:
                ok += 1
                lines = result["output"].strip().splitlines()
                print(f"[{i}/{len(specs)}] OK    {lines[0] if lines else ''}")

    print(f"\n{ok} ok, {failed} failed (see {FAILED.relative_to(ROOT)})")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
