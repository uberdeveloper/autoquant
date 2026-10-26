#!/usr/bin/env python3
"""Stage [5] CODEGEN -- write strategies/<slug>.py from specs/<slug>.yaml.

One LLM call per spec via llm.py (default: `opencode run`), then an OFFLINE smoke test: the module must
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
import llm
import os
import re
import sys
import time
from pathlib import Path

import catalog
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SPECS = ROOT / "specs"
STRATEGIES = ROOT / "strategies"
COMPARE = STRATEGIES / "compare"

CLI_TIMEOUT = 600

CONTRACT_SINGLE = """\
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

CONTRACT_MULTI = """\
Write a Python module implementing the strategy spec below.

Contract:
- define exactly: def signal(data: dict[str, pd.DataFrame], **params) -> pd.DataFrame
- `data` maps every ticker in the spec's data.universe to an OHLCV frame
  (lowercase open/high/low/close/volume, shared DatetimeIndex; closes are
  ffilled but are NaN before the ticker's first quote — weight 0 there).
- Return a weights DataFrame: same index as the frames,
  exactly one column per ticker, float target weights. Rows may be all 0.0 (cash).
- The harness lags the weights per column and applies costs -- do NOT shift,
  do NOT compute returns or costs.
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
    try:
        multi = len(spec["data"]["universe"]) > 1
    except (KeyError, TypeError):
        multi = False  # universe missing/malformed: the single contract is the safe default
    contract = CONTRACT_MULTI if multi else CONTRACT_SINGLE
    return (f"{contract}\n---\n\nStrategy spec:\n\n```yaml\n"
            f"{yaml.safe_dump(spec, sort_keys=False)}```\n")


def smoke_frame(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + np.linspace(-0.01, 0.01, n)), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(99.0), "high": close * 1.01,
         "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def smoke_data(series_id: str) -> pd.DataFrame:
    """Source-shaped smoke frame: close-only for close-only catalog series
    (FRED), full OHLCV otherwise."""
    frame = smoke_frame()
    if catalog.is_close_only(series_id):
        return frame[["close"]]
    return frame


def smoke_test(path: Path, params: dict, multi: bool = False,
               df: pd.DataFrame | None = None,
               data: dict[str, pd.DataFrame] | None = None) -> str | None:
    """Import the module and call signal on synthetic data. None = OK.

    multi=False: params are the signal kwargs; `df` defaults to a full
    OHLCV frame. Pass a source-shaped frame (e.g. close-only for FRED
    series) so a signal that indexes a column the source never provides
    fails here instead of crashing in backtest.run().
    multi=True: data maps ticker to source-shaped frames. In both modes,
    params contains the configured signal keyword arguments."""
    name = f"strategies.{path.stem}"
    spec_ = importlib.util.spec_from_file_location(
        name, path,
        loader=importlib.machinery.SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec_)
    try:
        spec_.loader.exec_module(mod)
        if not hasattr(mod, "signal"):
            return "defines no signal(df, **params)"
        if multi:
            if not data:
                return "portfolio smoke data is empty"
            out = mod.signal(data, **params)
        else:
            data = df if df is not None else smoke_frame()
            out = mod.signal(data, **params)
    except Exception as exc:  # generated code -- any failure is a codegen failure
        return f"{type(exc).__name__}: {exc}"
    if multi:
        if not isinstance(out, pd.DataFrame):
            return "signal did not return a pd.DataFrame (one column per ticker)"
        if set(out.columns) != set(data):
            return "weight columns do not match the universe"
        if not out.index.equals(next(iter(data.values())).index):
            return "weights index does not match data index"
        finite = np.isfinite(pd.to_numeric(out.stack(), errors="coerce"))
        if not finite.all():
            return "weights contained non-finite values"
        return None
    if not isinstance(out, pd.Series):
        return "signal did not return a pd.Series"
    if not out.index.equals(data.index):
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


def generate_one(spec_path: Path, model: str | None,
                 extra: list[str] | None = None,
                 template: str | None = None) -> dict:
    """One LLM call -> {"slug", "source"/"error", "elapsed"}."""
    try:
        spec = yaml.safe_load(spec_path.read_text())
        slug = spec["meta"]["slug"]
    except (yaml.YAMLError, KeyError, AttributeError, TypeError, OSError) as exc:
        return {"slug": spec_path.stem, "error": f"unreadable spec: {exc}"[:300],
                "elapsed": 0.0}
    if slug != spec_path.stem:
        return {"slug": spec_path.stem, "elapsed": 0.0,
                "error": "meta.slug does not match spec filename"}

    t0 = time.monotonic()
    try:
        reply = llm.complete(build_prompt(spec), model=model, extra=extra,
                             timeout=CLI_TIMEOUT, template=template)
    except llm.LLMError as exc:
        return {"slug": slug, "error": str(exc),
                "elapsed": time.monotonic() - t0}
    elapsed = time.monotonic() - t0
    source = strip_fence(reply)
    if not source:
        return {"slug": slug, "error": "empty reply", "elapsed": elapsed}
    return {"slug": slug, "source": source, "elapsed": elapsed}


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
    universe = list((spec.get("data") or {}).get("universe") or [])
    if len(universe) > 1:
        err = smoke_test(tmp, params, multi=True,
                         data={t: smoke_data(t) for t in universe})
    else:
        err = smoke_test(tmp, params, df=smoke_data(universe[0]) if universe else None)
    if err:
        tmp.unlink()  # a module that fails smoke is removed, never published
        _write_atomic(strategies_dir / f"{slug}.error",
                      json.dumps({"error": err}, indent=2))
        return "codegen_failed"
    tmp.replace(path)  # never visible half-written
    (strategies_dir / f"{slug}.error").unlink(missing_ok=True)  # code supersedes error
    return "coded"


def pick(label: str, slug: str) -> None:
    """Promote a compare candidate to the trusted strategies/ namespace."""
    cand_dir = COMPARE / label
    src = cand_dir / f"{slug}.py"
    if (cand_dir / f"{slug}.error").exists():
        sys.exit(f"error: candidate {label}/{slug} failed its smoke test")
    if not src.exists():
        sys.exit(f"error: no candidate {src.relative_to(ROOT)}")
    dest = STRATEGIES / f"{slug}.py"
    tmp = dest.with_suffix(".py.tmp")
    tmp.write_text(src.read_text())
    tmp.replace(dest)
    (STRATEGIES / f"{slug}.error").unlink(missing_ok=True)
    print(f"picked {label}/{slug} -> strategies/{slug}.py")


def run_compare(candidates: list[tuple[str, str]], todo: list[Path],
                args: argparse.Namespace) -> int:
    """Generate each pending spec once per candidate model; artifacts go to
    strategies/compare/<label>/. The circuit breaker trips per label. Print
    a summary table; exit 1 if any label aborted."""
    print(f"compare: {len(candidates)} models x {len(todo)} specs\n")
    stats = {label: {"coded": 0, "smoke_failed": 0, "cli_failed": 0,
                     "times": []} for label, _cmd in candidates}
    aborted: list[str] = []

    for label, cmd in candidates:
        print(f"-- {label} ({cmd})")
        done: list[dict] = []
        abort = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(generate_one, p, args.model,
                                   args.llm_arg, cmd): p for p in todo}
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                spec_path = futures[fut]
                try:
                    spec = yaml.safe_load(spec_path.read_text())
                except (yaml.YAMLError, OSError):
                    spec = None  # error results never reach the spec-using path
                stage = publish(result, spec, COMPARE / label)
                err = result.get("error", "")
                if stage == "coded":
                    stats[label]["coded"] += 1
                    mark = "OK  "
                elif llm.is_cli_error(err):
                    stats[label]["cli_failed"] += 1
                    mark = "FAIL"
                else:
                    stats[label]["smoke_failed"] += 1
                    mark = "FAIL"
                stats[label]["times"].append(result.get("elapsed", 0.0))
                print(f"[{label}] {mark}  {stage:<15} {spec_path.stem}")
                done.append(result)
                abort = llm.circuit_break(done)
                if abort:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
        if abort:
            print(f"\n[{label}] {abort}")
            aborted.append(label)

    print(f"\n{'label':<12} {'coded':>6} {'smoke_failed':>13} "
          f"{'cli_failed':>11} {'median_s':>9}")
    for label, _cmd in candidates:
        s = stats[label]
        times = sorted(s["times"])
        median = times[len(times) // 2] if times else 0.0
        print(f"{label:<12} {s['coded']:>6} {s['smoke_failed']:>13} "
              f"{s['cli_failed']:>11} {median:>9.1f}")

    print(f"\nartifacts -> strategies/compare/<label>/ ; promote one with "
          f"codegen.py --pick <label> --slug <slug>")
    return 1 if aborted else 0


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent LLM calls")
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--llm-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed through to the LLM CLI (repeatable)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry specs with a strategies/<slug>.error marker")
    ap.add_argument("--compare-models", default=None, metavar="SPEC",
                    help="'label=cmd,label=cmd' comparison mode (env fallback: "
                         "AUTOQUANT_LLM_COMPARE); artifacts go to "
                         "strategies/compare/<label>/")
    ap.add_argument("--pick", default=None, metavar="LABEL",
                    help="promote strategies/compare/<label>/<slug>.py to "
                         "strategies/<slug>.py (requires --slug)")
    ap.add_argument("--slug", default=None,
                    help="slug to promote with --pick")
    args = ap.parse_args(argv)

    if args.pick:
        if not args.slug:
            sys.exit("error: --pick requires --slug")
        pick(args.pick, args.slug)
        return 0

    compare_spec = args.compare_models or os.environ.get(
        "AUTOQUANT_LLM_COMPARE", "")
    candidates = None
    if compare_spec:
        try:
            candidates = llm.parse_compare(compare_spec)
        except llm.LLMError as exc:
            sys.exit(f"error: {exc}")

    try:
        if candidates:
            for _label, cmd in candidates:
                llm.preflight_for(cmd)
        else:
            llm.preflight()
    except llm.LLMError as exc:
        sys.exit(f"error: {exc}")

    todo = pending_specs(SPECS, STRATEGIES, args.retry_failed)
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to codegen -- run extract.py first, or pass --retry-failed")
        return 0

    if candidates:
        return run_compare(candidates, todo, args)

    print(f"codegen {len(todo)} specs ({args.jobs} at a time)\n")
    stages: dict[str, int] = {}
    done: list[dict] = []

    abort = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(generate_one, p, args.model,
                               args.llm_arg): p for p in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            spec_path = futures[fut]
            try:
                spec = yaml.safe_load(spec_path.read_text())
            except (yaml.YAMLError, OSError):
                spec = None  # error results never reach the spec-using path
            stage = publish(result, spec, STRATEGIES)
            stages[stage] = stages.get(stage, 0) + 1
            mark = "OK  " if stage == "coded" else "FAIL"
            print(f"[{i}/{len(todo)}] {mark}  {stage:<15} {spec_path.stem}")
            done.append(result)
            abort = llm.circuit_break(done)
            if abort:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    if abort:
        print(f"\n{abort}")
        return 1

    print("\nstages:", ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))
    print(f"strategies -> {STRATEGIES.relative_to(ROOT)}/")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
