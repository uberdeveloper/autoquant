#!/usr/bin/env python3
"""Stage [5] CODEGEN -- write strategies/<slug>.py from specs/<slug>.yaml.

One claude -p call per spec, then an OFFLINE smoke test: the module must
import and signal(df, **params) must return a finite, index-aligned series
on synthetic data. The backtest harness lags the signal -- generated code
must not shift it, and the prompt says so explicitly.

Independent and resumable: reads only specs/, writes only strategies/.
The strategy file itself is the state -- re-running skips coded slugs.

    python3 codegen.py --limit 5     # calibrate
    python3 codegen.py               # every spec without a strategy yet
    python3 codegen.py --retry-failed
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SPECS = ROOT / "specs"
STRATEGIES = ROOT / "strategies"

CLI_TIMEOUT = 600

CONTRACT = """\
Write a Python module implementing the strategy spec below.

Contract:
- define exactly: def signal(df: pd.DataFrame, **params) -> pd.Series
- `df` has lowercase open/high/low/close/volume columns and a DatetimeIndex.
- Return target weights (float series, or boolean treated as 0/1) with the
  SAME index as df.
- The harness lags the signal and applies costs -- do NOT shift, and do NOT
  compute returns or costs.
- Vectorised pandas/numpy only; no network, no file I/O, no prints.
- Every parameter the spec's signal section uses must be a keyword argument
  with a default.
- Output ONLY Python source. No markdown fences, no prose.
"""


def strip_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.+?)```", text, re.S)
    return (fence.group(1) if fence else text).strip()


def build_prompt(spec: dict) -> str:
    return (f"{CONTRACT}\n---\n\nStrategy spec:\n\n```yaml\n"
            f"{yaml.safe_dump(spec, sort_keys=False)}```\n")


def smoke_frame(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + np.linspace(-0.01, 0.01, n)), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(99.0), "high": close * 1.01,
         "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def smoke_test(path: Path, params: dict) -> str | None:
    """Import the module and call signal on synthetic data. None = OK."""
    name = f"strategies.{path.stem}"
    spec_ = importlib.util.spec_from_file_location(
        name, path,
        loader=importlib.machinery.SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec_)
    try:
        spec_.loader.exec_module(mod)
        if not hasattr(mod, "signal"):
            return "defines no signal(df, **params)"
        df = smoke_frame()
        out = mod.signal(df, **params)
    except Exception as exc:  # generated code -- any failure is a codegen failure
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(out, pd.Series):
        return "signal did not return a pd.Series"
    if not out.index.equals(df.index):
        return "signal index does not match df.index"
    if not np.isfinite(pd.to_numeric(out, errors="coerce")).all():
        return "signal returned non-finite values"
    return None


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def pending_specs(specs_dir: Path, strategies_dir: Path, retry: bool) -> list[Path]:
    """Specs with no codegen outcome yet. A .error marker counts as an
    outcome unless retry is set."""
    done = {p.stem for p in strategies_dir.glob("*.py")}
    if not retry:
        done |= {p.stem for p in strategies_dir.glob("*.error")}
    return [p for p in sorted(specs_dir.glob("*.yaml")) if p.stem not in done]


def generate_one(spec_path: Path, model: str | None) -> dict:
    """One claude -p call -> {"slug", "source"} or {"slug", "error"}."""
    try:
        spec = yaml.safe_load(spec_path.read_text())
        slug = spec["meta"]["slug"]
    except (yaml.YAMLError, KeyError, AttributeError, TypeError, OSError) as exc:
        return {"slug": spec_path.stem, "error": f"unreadable spec: {exc}"[:300]}
    if slug != spec_path.stem:
        return {"slug": spec_path.stem,
                "error": "meta.slug does not match spec filename"}
    cmd = ["claude", "-p"] + (["--model", model] if model else [])

    try:
        proc = subprocess.run(cmd, input=build_prompt(spec), capture_output=True,
                              text=True, timeout=CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"slug": slug, "error": f"claude CLI timed out after {CLI_TIMEOUT}s"}
    if proc.returncode != 0:
        return {"slug": slug, "error": f"claude exited {proc.returncode}: {proc.stderr[:200]}"}
    source = strip_fence(proc.stdout)
    if not source:
        return {"slug": slug, "error": "empty reply"}
    return {"slug": slug, "source": source}


def publish(result: dict, spec: dict, strategies_dir: Path) -> str:
    """Smoke-test at a temp path, then publish atomically. The artifact is
    the state -- the trusted <slug>.py only ever appears if smoke passed."""
    slug = result["slug"]
    if "error" in result:
        _write_atomic(strategies_dir / f"{slug}.error",
                      json.dumps({"error": result["error"]}, indent=2))
        return "codegen_failed"

    path = strategies_dir / f"{slug}.py"
    tmp = path.with_suffix(".py.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(result["source"])

    params = {k: v for k, v in spec.get("signal", {}).items()
              if k not in ("definition", "lag_bars")}
    err = smoke_test(tmp, params)
    if err:
        tmp.unlink()  # a module that fails smoke is removed, never published
        _write_atomic(strategies_dir / f"{slug}.error",
                      json.dumps({"error": err}, indent=2))
        return "codegen_failed"
    tmp.replace(path)  # never visible half-written
    (strategies_dir / f"{slug}.error").unlink(missing_ok=True)  # code supersedes error
    return "coded"


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent claude calls")
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry specs with a strategies/<slug>.error marker")
    args = ap.parse_args(argv)

    todo = pending_specs(SPECS, STRATEGIES, args.retry_failed)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to codegen -- run extract.py first, or pass --retry-failed")
        return 0

    print(f"codegen {len(todo)} specs ({args.jobs} at a time)\n")
    stages: dict[str, int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate_one, p, args.model): p for p in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            spec_path = futures[fut]
            try:
                spec = yaml.safe_load(spec_path.read_text())
            except (yaml.YAMLError, OSError):
                spec = None  # error results never reach the spec-using path
            stage = publish(fut.result(), spec, STRATEGIES)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "coded" else "FAIL"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {spec_path.stem}")

    print("\nstages:", ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    print(f"strategies -> {STRATEGIES.relative_to(ROOT)}/")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
