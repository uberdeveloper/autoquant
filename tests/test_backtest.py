"""Tests for backtest.py — lag, costs, metrics, validation, and reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

import backtest


def make_prices(n=60, seed=0):
    """Rising prices with known daily returns; open lags close by 1%."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    open_ = np.roll(close, 1) * 0.99
    open_[0] = 99.0
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": close * 1.01, "low": open_ * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )


BASE_SPEC = {
    "signal": {"definition": "test", "lag_bars": 1},
    "rules": {"max_leverage": 1.0},
    "costs": {"commission_bps": 10, "slippage_bps": 10},
}


class TestBacktest:
    def test_lag_is_applied_no_lookahead(self):
        df = make_prices()
        weights = pd.Series(1.0, index=df.index)
        res = backtest.backtest(df, weights, BASE_SPEC)
        # position on bar t must be the weight decided on bar t-1
        assert res["pos"].iloc[1] == 1.0
        assert res["pos"].iloc[0] == 0.0  # first bar has no prior decision

    def test_gross_equals_position_times_asset_return(self):
        df = make_prices()
        w = pd.Series(1.0, index=df.index)
        res = backtest.backtest(df, w, BASE_SPEC)
        assert np.allclose(res["gross"], res["pos"] * res["asset_ret"])
        assert np.allclose(res["net"], res["gross"] - res["cost"])

    def test_close_execution_uses_close_to_close(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        expected = df["close"].pct_change().fillna(0.0)
        assert np.allclose(res["asset_ret"], expected)

    def test_next_open_execution_uses_open_to_open(self):
        df = make_prices()
        spec = dict(BASE_SPEC, rules={"execution_price": "next_open"})
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), spec)
        expected = (df["open"].shift(-1) / df["open"] - 1).fillna(0.0)
        assert np.allclose(res["asset_ret"], expected)

    def test_costs_reduce_gross_by_turnover_times_bps(self):
        df = make_prices()
        spec = {"signal": {}, "rules": {}, "costs": {"slippage_bps": 100}}
        w = pd.Series([1.0, 1.0, 0.0, 1.0, 0.0] * 12, index=df.index)
        res = backtest.backtest(df, w, spec)
        # per-side cost is 100bps = 1% of turnover
        expected_cost = res["pos"].diff().abs().fillna(res["pos"].abs()) * 0.01
        assert np.allclose(res["cost"], expected_cost)

    def test_weights_clipped_to_max_leverage(self):
        df = make_prices()
        spec = {"signal": {}, "rules": {"max_leverage": 0.5}, "costs": {}}
        w = pd.Series(3.0, index=df.index)
        res = backtest.backtest(df, w, spec)
        assert res["pos"].max() == 0.5

    def test_missing_dates_filled_with_zero_position(self):
        df = make_prices()
        w = pd.Series([1.0], index=[df.index[0]])  # one decision, then gaps
        res = backtest.backtest(df, w, BASE_SPEC)
        assert len(res) == len(df)
        assert res["pos"].iloc[1] == 1.0  # the one decision takes effect next bar
        assert res["pos"].iloc[2:].eq(0.0).all()

    def test_zero_commission_still_charges_slippage(self):
        df = make_prices()
        spec = {"signal": {}, "rules": {}, "costs": {"slippage_bps": 50}}
        w = pd.Series([0.0, 1.0] * 30, index=df.index)
        res = backtest.backtest(df, w, spec)
        assert res["cost"].sum() > 0


class TestMetrics:
    def test_full_output_keys(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        m = backtest.metrics(res)
        for key in ("start", "end", "n_bars", "years", "cagr", "vol",
                    "sharpe", "max_dd", "t_stat", "exposure", "trades",
                    "cost_drag_annual"):
            assert key in m
        assert m["n_bars"] == len(df)
        assert m["exposure"] == pytest.approx(1.0, abs=0.02)  # lag leaves bar 0 flat
        assert m["max_dd"] <= 0

    def test_tiny_input_returns_n_bars_only(self):
        res = pd.DataFrame({"pos": [1.0], "asset_ret": [0.01], "gross": [0.01],
                            "cost": [0.0], "net": [0.01]},
                           index=pd.bdate_range("2020-01-01", periods=1))
        assert backtest.metrics(res) == {"n_bars": 1}

    def test_trades_counts_position_changes(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        res = pd.DataFrame(
            {"pos": [0.0, 1.0, 1.0, 0.0, 0.0], "asset_ret": 0.0, "gross": 0.0,
             "cost": 0.0, "net": 0.0}, index=idx)
        assert backtest.metrics(res)["trades"] == 2

    def test_gross_sharpe_at_least_net_sharpe(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        assert backtest.metrics(res, "gross")["sharpe"] >= backtest.metrics(res)["sharpe"]


class TestBuyAndHold:
    def test_net_is_asset_ret_and_one_trade(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(0.0, index=df.index), BASE_SPEC)
        m = backtest.buy_and_hold(res)
        assert m["trades"] == 1
        assert m["exposure"] == 1.0


class TestDeflatedSharpe:
    def test_no_haircut_for_single_trial(self):
        assert backtest.deflated_sharpe(1.5, 1000, 1) == 1.5
        assert backtest.deflated_sharpe(1.5, 1000, 0) == 1.5

    def test_haircut_grows_with_trials(self):
        s2 = backtest.deflated_sharpe(1.5, 2520, 2)
        s10 = backtest.deflated_sharpe(1.5, 2520, 10)
        assert 1.5 > s2 > s10

    def test_haircut_shrinks_with_more_bars(self):
        short = backtest.deflated_sharpe(1.5, 252, 10)
        long = backtest.deflated_sharpe(1.5, 2520, 10)
        assert long > short


class TestRandomEntryNull:
    def test_degenerate_all_cash_returns_note(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(0.0, index=df.index), BASE_SPEC)
        out = backtest.random_entry_null(res, BASE_SPEC, n=50)
        assert "note" in out

    def test_real_edge_ranks_against_null(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        out = backtest.random_entry_null(res, BASE_SPEC, n=100, seed=0)
        assert out["actual_sharpe"] == out["actual_sharpe"]  # finite
        assert 0 <= out["percentile"] <= 100
        assert isinstance(out["beats_null_95"], bool)

    def test_deterministic_given_seed(self):
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        a = backtest.random_entry_null(res, BASE_SPEC, n=30, seed=7)
        b = backtest.random_entry_null(res, BASE_SPEC, n=30, seed=7)
        assert a == b


class TestAutoFlags:
    def make_out(self, sharpe=1.0, bh_sharpe=0.5, trades=100, oos_sharpe=0.8,
                 beats_null=True):
        return {
            "full": {"sharpe": sharpe, "trades": trades},
            "benchmark_bh": {"sharpe": bh_sharpe},
            "out_of_sample": {"sharpe": oos_sharpe},
            "null_test": {"beats_null_95": beats_null},
        }

    def test_zero_cost_model_flags(self):
        spec = {"costs": {"commission_bps": 0, "slippage_bps": 0},
                "signal": {"lag_bars": 1}, "validation": {}}
        flags = backtest.auto_flags(spec, self.make_out())
        assert any("zero cost model" in f for f in flags)

    def test_few_trades_flags(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 1},
                "validation": {"min_trades": 30}}
        flags = backtest.auto_flags(spec, self.make_out(trades=3))
        assert any("trades" in f for f in flags)

    def test_zero_lag_flags_lookahead(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 0},
                "validation": {}}
        flags = backtest.auto_flags(spec, self.make_out())
        assert any("lookahead" in f for f in flags)

    def test_underperforming_bh_flags_weak(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 1},
                "validation": {}}
        flags = backtest.auto_flags(spec, self.make_out(sharpe=0.1, bh_sharpe=0.9))
        assert any("buy-and-hold" in f for f in flags)

    def test_negative_oos_flags_weak(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 1},
                "validation": {}}
        flags = backtest.auto_flags(spec, self.make_out(oos_sharpe=-0.4))
        assert any("since publication" in f for f in flags)

    def test_losing_to_null_flags_weak(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 1},
                "validation": {}}
        flags = backtest.auto_flags(spec, self.make_out(beats_null=False))
        assert any("null" in f for f in flags)

    def test_clean_run_has_no_flags(self):
        spec = {"costs": {"commission_bps": 5, "slippage_bps": 5},
                "signal": {"lag_bars": 1}, "validation": {"min_trades": 30},
                "ambiguities": []}
        flags = backtest.auto_flags(spec, self.make_out())
        assert flags == []

    def test_material_ambiguities_listed(self):
        spec = {"costs": {"commission_bps": 5}, "signal": {"lag_bars": 1},
                "validation": {},
                "ambiguities": [{"id": "universe", "material": True}]}
        flags = backtest.auto_flags(spec, self.make_out())
        assert any("universe" in f for f in flags)


class TestWriteReport:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)
        spec = {
            "meta": {
                "title": "200dma Timing", "source": "A Blog",
                "url": "https://a.com/x", "posted": "2010-01-01",
                "claim": "Timing the 200dma beats buy and hold.",
                "author_evidence": {
                    "headline_metrics": "Sharpe 1.1", "sample": "1990-2010",
                    "costs_included": False,
                },
            },
            "verdict": {"status": "supported", "reason": "beats null"},
            "ambiguities": [{"id": "u1", "material": False,
                             "question": "which index?", "default": "SPY",
                             "alternatives": ["QQQ"]}],
        }
        df = make_prices()
        res = backtest.backtest(df, pd.Series(1.0, index=df.index), BASE_SPEC)
        out = {
            "slug": "200dma-timing",
            "full": backtest.metrics(res),
            "gross": backtest.metrics(res, "gross"),
            "benchmark_bh": backtest.buy_and_hold(res),
            "in_sample": {}, "out_of_sample": {},
            "cost_sweep": {"0bps": 1.2, "50bps": 1.0},
            "param_sweep": {"lookback": {"100": 0.9, "200": 1.0}},
            "null_test": backtest.random_entry_null(res, BASE_SPEC, n=20),
            "deflated_sharpe": 0.9,
            "auto_flags": ["WEAK: does not beat buy-and-hold after costs"],
        }
        return tmp_path, spec, out

    def test_report_written_and_contains_sections(self, env):
        tmp_path, spec, out = env
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec))
        path = backtest.write_report(spec_path, out)
        assert path.exists()
        assert path == tmp_path / "reports" / "200dma-timing.md"
        text = path.read_text()
        for expected in ("# 200dma Timing", "**supported**", "## Verdict",
                         "## Headline", "beats null", "Cost sensitivity",
                         "Random-entry null", "## Automated flags"):
            assert expected in text

    def test_report_includes_cost_sweep_values(self, env):
        tmp_path, spec, out = env
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec))
        text = backtest.write_report(spec_path, out).read_text()
        assert "0bps" in text and "50bps" in text


class TestEndToEndRun:
    def test_run_on_cached_prices_and_local_strategy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)

        (tmp_path / "data" / "prices").mkdir(parents=True)
        df = make_prices(80)
        df.to_csv(tmp_path / "data" / "prices" / "SPY_TEST.csv")

        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "always-in.py").write_text(
            "import pandas as pd\n"
            "def signal(df, **params):\n"
            "    return pd.Series(1.0, index=df.index)\n"
        )

        spec = {
            "meta": {
                "slug": "always-in", "title": "Always In",
                "source": "test", "url": "https://a.com", "posted": "2020-03-02",
                "claim": "always long",
                "author_evidence": {"headline_metrics": "n/a", "sample": "n/a",
                                    "costs_included": False},
            },
            "data": {"universe": ["SPY_TEST"], "start": "2020-01-01"},
            "signal": {"definition": "always long", "lag_bars": 1},
            "rules": {"max_leverage": 1.0},
            "costs": {"commission_bps": 5, "slippage_bps": 5},
            "validation": {
                "cost_sweep_bps": [0, 50],
                "param_sweep": None,
                "multiple_testing": {"n_tested_so_far": 3},
                "min_trades": 1,
            },
            "verdict": {"status": "unknown", "reason": None},
            "ambiguities": [],
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec))

        out = backtest.run(spec_path, n_trials=3)
        assert out["slug"] == "always-in"
        assert out["full"]["n_bars"] == 80
        assert set(out["cost_sweep"]) == {"0bps", "50bps"}
        assert out["deflated_sharpe"] < out["full"]["sharpe"]
        assert isinstance(out["auto_flags"], list)

        # in-sample / out-of-sample split honours the posted date
        assert out["in_sample"]["n_bars"] + out["out_of_sample"]["n_bars"] == 80


class TestLoadStrategy:
    def test_missing_module_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)
        with pytest.raises(SystemExit, match="missing implementation"):
            backtest.load_strategy("nope")

    def test_module_without_signal_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "empty.py").write_text("x = 1\n")
        with pytest.raises(SystemExit, match="no signal"):
            backtest.load_strategy("empty")

    def test_valid_module_returns_namespace_with_signal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backtest, "ROOT", tmp_path)
        strategies = tmp_path / "strategies"
        strategies.mkdir()
        (strategies / "ok.py").write_text("def signal(df, **p):\n    return df.close\n")
        mod = backtest.load_strategy("ok")
        assert callable(mod.signal)
