#!/usr/bin/env python3
"""Stages [6][7][8] — one harness for every harvested idea.

    python3 backtest.py specs/example-200dma-timing.yaml

Loads a spec, imports strategies/<slug>.py, runs the backtest and the adversarial
suite (IS/OOS split at the publication date, cost sweep, parameter neighbourhood,
random-entry null), and writes reports/<slug>.md.

The contract for a strategy module is one function:

    signal(df: pd.DataFrame, **params) -> pd.Series   # target weight, index == df.index

`df` has lowercase open/high/low/close/volume. The series may be boolean (treated
as 0/1) or float weights. It is lagged by the harness — do NOT shift it yourself.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


# ---------------------------------------------------------------- data

def load_prices(ticker: str, start, end, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker.replace('/', '_')}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ---------------------------------------------------------------- backtest

def backtest(df: pd.DataFrame, weights: pd.Series, spec: dict) -> pd.DataFrame:
    """Weights are as-of-decision; the harness applies the lag and the costs."""
    rules, costs = spec["rules"], spec["costs"]
    lag = int(spec["signal"].get("lag_bars", 1))

    w = weights.reindex(df.index).astype(float).fillna(0.0)
    w = w.clip(-rules.get("max_leverage", 1.0), rules.get("max_leverage", 1.0))
    pos = w.shift(lag).fillna(0.0)  # the single most important line in this file

    # asset_ret[t] must be the return a position in force during bar t actually earns,
    # using only information available when that position was set.
    #   same_close: decided at close t-1, held close t-1 -> close t.
    #   next_open:  decided at close t-1, filled at open t, held open t -> open t+1.
    # Pairing next_open with a plain open.pct_change() would credit the position with
    # the open t-1 -> close t-1 move that PRECEDED its own signal. That is lookahead,
    # and it inflates Sharpe by roughly 2x on a daily trend overlay.
    if rules.get("execution_price") == "next_open":
        asset_ret = (df["open"].shift(-1) / df["open"] - 1).fillna(0.0)
    else:
        asset_ret = df["close"].pct_change().fillna(0.0)

    turnover = pos.diff().abs().fillna(pos.abs())
    per_side = (costs.get("commission_bps", 0) + costs.get("slippage_bps", 0)) / 1e4
    cost = turnover * per_side
    carry = (
        np.maximum(-pos, 0) * costs.get("borrow_bps_annual", 0) / 1e4 / TRADING_DAYS
        + np.maximum(pos.abs() - 1, 0) * costs.get("financing_bps_annual", 0) / 1e4 / TRADING_DAYS
    )

    gross = pos * asset_ret
    return pd.DataFrame(
        {"pos": pos, "asset_ret": asset_ret, "gross": gross,
         "cost": cost + carry, "net": gross - cost - carry}
    )


def metrics(res: pd.DataFrame, col: str = "net") -> dict:
    r = res[col].dropna()
    if len(r) < 2:
        return {"n_bars": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    trades = int((res["pos"].diff().abs() > 1e-9).sum())
    # t-stat on the mean daily net return: the honest significance number
    tstat = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
    return {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "n_bars": len(r), "years": round(yrs, 2),
        "cagr": round(float(cagr), 4), "vol": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3), "max_dd": round(float(dd), 4),
        "t_stat": round(float(tstat), 2),
        "exposure": round(float(res["pos"].abs().mean()), 3),
        "trades": trades,
        "cost_drag_annual": round(float(res["cost"].mean() * TRADING_DAYS), 4),
    }


def buy_and_hold(res: pd.DataFrame) -> dict:
    bh = res.assign(pos=1.0, net=res["asset_ret"], cost=0.0)
    m = metrics(bh)
    m["trades"] = 1
    return m


# ---------------------------------------------------------------- validation

def random_entry_null(res: pd.DataFrame, spec: dict, n: int = 500, seed: int = 0) -> dict:
    """Same time-in-market, random timing. Where does the real Sharpe rank?"""
    rng = np.random.default_rng(seed)
    exposure = float(res["pos"].abs().mean())
    n_bars = len(res)
    k = int(round(exposure * n_bars))
    if k == 0 or k == n_bars:
        return {"note": "degenerate exposure — null test not meaningful"}
    ret = res["asset_ret"].to_numpy()
    per_side = (spec["costs"].get("commission_bps", 0) + spec["costs"].get("slippage_bps", 0)) / 1e4

    # All trials at once: a (n, n_bars) 0/1 position matrix. argsort of a random
    # matrix gives k distinct bar indices per row, matching rng.choice(replace=False).
    pos = np.zeros((n, n_bars))
    chosen = np.argsort(rng.random((n, n_bars)), axis=1)[:, :k]
    np.put_along_axis(pos, chosen, 1.0, axis=1)

    net = pos * ret - np.abs(np.diff(pos, axis=1, prepend=0.0)) * per_side
    std = net.std(axis=1)
    sharpes = np.where(std > 0, net.mean(axis=1) / np.where(std > 0, std, 1.0) * np.sqrt(TRADING_DAYS), 0.0)

    actual = metrics(res)["sharpe"]
    return {
        "actual_sharpe": actual,
        "null_sharpe_mean": round(float(sharpes.mean()), 3),
        "null_sharpe_p95": round(float(np.percentile(sharpes, 95)), 3),
        "percentile": round(float((sharpes < actual).mean() * 100), 1),
        "beats_null_95": bool(actual > np.percentile(sharpes, 95)),
    }


def deflated_sharpe(sharpe: float, n_bars: int, n_trials: int) -> float:
    """Bailey & Lopez de Prado haircut for having tested many ideas."""
    if not n_trials or n_trials < 2 or n_bars < 2:
        return sharpe
    from math import log, sqrt

    # expected max Sharpe of n_trials independent, truly-zero-edge strategies
    e_max = sqrt(2 * log(n_trials)) - (log(log(n_trials)) + log(4 * np.pi)) / (
        2 * sqrt(2 * log(n_trials))
    )
    return round(float(sharpe - e_max / sqrt(n_bars / TRADING_DAYS)), 3)


# ---------------------------------------------------------------- orchestration

def load_strategy(slug: str):
    path = ROOT / "strategies" / f"{slug}.py"
    if not path.exists():
        sys.exit(f"missing implementation: {path}")
    spec_ = importlib.util.spec_from_file_location(f"strategies.{slug}", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    if not hasattr(mod, "signal"):
        sys.exit(f"{path} defines no signal(df, **params)")
    return mod


def run(spec_path: Path, n_trials: int | None) -> dict:
    spec = yaml.safe_load(spec_path.read_text())
    slug = spec["meta"]["slug"]
    mod = load_strategy(slug)

    tickers = spec["data"]["universe"]
    if len(tickers) != 1:
        sys.exit("this skeleton handles single-asset specs; extend for cross-sectional")
    df = load_prices(tickers[0], spec["data"]["start"], spec["data"].get("end"),
                     ROOT / "data" / "prices")
    if df.empty:
        sys.exit(f"no price data for {tickers[0]}")

    params = {k: v for k, v in spec["signal"].items() if k not in ("definition", "lag_bars")}
    res = backtest(df, mod.signal(df, **params), spec)

    posted = pd.Timestamp(spec["meta"]["posted"])
    out = {
        "slug": slug,
        "full": metrics(res),
        "gross": metrics(res, "gross"),
        "benchmark_bh": buy_and_hold(res),
        "in_sample": metrics(res[res.index < posted]) if (res.index < posted).any() else {},
        "out_of_sample": metrics(res[res.index >= posted]) if (res.index >= posted).any() else {},
    }

    # cost sensitivity — where does the edge die?
    out["cost_sweep"] = {}
    for bps in spec["validation"]["cost_sweep_bps"]:
        s2 = copy.deepcopy(spec)
        s2["costs"]["commission_bps"], s2["costs"]["slippage_bps"] = 0, bps
        out["cost_sweep"][f"{bps}bps"] = metrics(backtest(df, mod.signal(df, **params), s2))["sharpe"]

    # parameter neighbourhood — is the result a spike or a plateau?
    out["param_sweep"] = {}
    for pname, values in (spec["validation"].get("param_sweep") or {}).items():
        out["param_sweep"][pname] = {
            str(v): metrics(backtest(df, mod.signal(df, **{**params, pname: v}), spec))["sharpe"]
            for v in values
        }

    out["null_test"] = random_entry_null(res, spec)
    out["deflated_sharpe"] = deflated_sharpe(
        out["full"]["sharpe"], out["full"]["n_bars"],
        n_trials or spec["validation"]["multiple_testing"].get("n_tested_so_far") or 1,
    )
    out["auto_flags"] = auto_flags(spec, out)
    return out


def auto_flags(spec: dict, out: dict) -> list[str]:
    flags = []
    c = spec["costs"]
    if c.get("commission_bps", 0) + c.get("slippage_bps", 0) <= 0:
        flags.append("FAIL: zero cost model")
    if out["full"]["trades"] < spec["validation"].get("min_trades", 30):
        flags.append(f"FAIL: only {out['full']['trades']} trades — declining to conclude")
    if int(spec["signal"].get("lag_bars", 1)) < 1:
        flags.append("FAIL: lag_bars < 1 — lookahead")
    if out["full"]["sharpe"] <= out["benchmark_bh"]["sharpe"]:
        flags.append("WEAK: does not beat buy-and-hold after costs")
    if out["out_of_sample"] and out["out_of_sample"].get("sharpe", 0) < 0:
        flags.append("WEAK: negative Sharpe since publication")
    if not out["null_test"].get("beats_null_95", True):
        flags.append("WEAK: inside the random-entry null 95th percentile")
    if any(a.get("material") for a in spec.get("ambiguities") or []):
        mats = [a["id"] for a in spec["ambiguities"] if a.get("material")]
        flags.append(f"NOTE: material ambiguities unswept: {', '.join(mats)}")
    return flags


def write_report(spec_path: Path, out: dict) -> Path:
    spec = yaml.safe_load(spec_path.read_text())
    m, bh = out["full"], out["benchmark_bh"]
    lines = [
        f"# {spec['meta']['title']}",
        "",
        f"- source: {spec['meta']['source']} — <{spec['meta']['url']}>",
        f"- published: {spec['meta']['posted']}  (OOS boundary)",
        f"- claim: {spec['meta']['claim'].strip()}",
        f"- spec: `{spec_path.resolve().relative_to(ROOT)}`",
        "",
        "## Verdict",
        "",
        f"**{spec['verdict']['status']}** — {spec['verdict'].get('reason') or 'fill in after review'}",
        "",
        "## Headline (net of costs)",
        "",
        "| | strategy | buy & hold |",
        "|---|---|---|",
        f"| CAGR | {m['cagr']:.2%} | {bh['cagr']:.2%} |",
        f"| Sharpe | {m['sharpe']} | {bh['sharpe']} |",
        f"| max DD | {m['max_dd']:.2%} | {bh['max_dd']:.2%} |",
        f"| t-stat | {m['t_stat']} | {bh['t_stat']} |",
        f"| exposure | {m['exposure']:.0%} | 100% |",
        f"| trades | {m['trades']} | 1 |",
        "",
        f"Gross Sharpe {out['gross']['sharpe']} vs net {m['sharpe']}; "
        f"annual cost drag {m['cost_drag_annual']:.2%}. "
        f"Deflated Sharpe {out['deflated_sharpe']}.",
        "",
        "## In-sample vs since-publication",
        "",
        "| window | period | CAGR | Sharpe | max DD |",
        "|---|---|---|---|---|",
    ]
    for label, key in [("author's era", "in_sample"), ("since publication", "out_of_sample")]:
        w = out.get(key) or {}
        if w.get("cagr") is not None:
            lines.append(
                f"| {label} | {w['start']}→{w['end']} | {w['cagr']:.2%} | {w['sharpe']} | {w['max_dd']:.2%} |"
            )
    lines += [
        "",
        "## Cost sensitivity (Sharpe by per-side slippage)",
        "",
        "| " + " | ".join(out["cost_sweep"]) + " |",
        "|" + "---|" * len(out["cost_sweep"]),
        "| " + " | ".join(str(v) for v in out["cost_sweep"].values()) + " |",
        "",
        "## Parameter neighbourhood (Sharpe)",
        "",
    ]
    for p, sweep in out["param_sweep"].items():
        lines += [f"`{p}`: " + ", ".join(f"{k}={v}" for k, v in sweep.items()), ""]
    lines += [
        "## Random-entry null (matched exposure)",
        "",
        "```json",
        json.dumps(out["null_test"], indent=2),
        "```",
        "",
        "## Automated flags",
        "",
        *([f"- {f}" for f in out["auto_flags"]] or ["- none"]),
        "",
        "## Author's claim vs this run",
        "",
        f"- author reported: `{spec['meta']['author_evidence']['headline_metrics']}` "
        f"over {spec['meta']['author_evidence']['sample']}, "
        f"costs included: {spec['meta']['author_evidence']['costs_included']}",
        f"- this run: CAGR {m['cagr']:.2%}, Sharpe {m['sharpe']}, over {m['start']}→{m['end']}",
        "",
        "## Open ambiguities",
        "",
        *[f"- **{a['id']}** ({'material' if a.get('material') else 'minor'}): {a['question']} "
          f"→ default `{a['default']}`, alternatives {a.get('alternatives')}"
          for a in (spec.get("ambiguities") or [])],
        "",
    ]
    path = ROOT / "reports" / f"{out['slug']}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--n-trials", type=int, default=None,
                    help="ideas tested so far, for the multiple-testing haircut")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = run(args.spec, args.n_trials)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    report = write_report(args.spec, out)
    print(f"\n{out['slug']}: Sharpe {out['full']['sharpe']} net "
          f"(B&H {out['benchmark_bh']['sharpe']}), {out['full']['trades']} trades")
    for f in out["auto_flags"]:
        print(f"  {f}")
    print(f"-> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
