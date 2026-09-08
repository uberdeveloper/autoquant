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
    spec_ = importlib.util.spec_from_file_location(f"strategies.{path.stem}", path)
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
