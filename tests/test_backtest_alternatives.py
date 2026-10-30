import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

def make_mock_df():
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="B")
    return pd.DataFrame({
        "open": np.linspace(100, 200, len(dates)),
        "high": np.linspace(101, 201, len(dates)),
        "low": np.linspace(99, 199, len(dates)),
        "close": np.linspace(100, 200, len(dates)),
        "volume": 1000.0,
    }, index=dates)


def test_backtest_cli_replace_data(tmp_path: Path):
    import backtest
    import runs_db

    spec_yaml = """
meta:
  slug: test-strat
  title: "Test Strategy"
  posted: 2022-01-01
  claim: "Holding test asset"
data:
  universe: [SPY]
  bar: 1d
  start: 2020-01-01
signal:
  definition: "raw = close > 0"
  lookback_bars: 10
  lag_bars: 1
rules:
  entry: "raw"
  exit: "not raw"
  execution_price: next_open
costs:
  commission_bps: 1.0
  slippage_bps: 5.0
validation:
  cost_sweep_bps: [0, 5, 10]
  param_sweep: {}
  multiple_testing:
    n_tested_so_far: 1
"""
    spec_file = tmp_path / "test-strat.yaml"
    spec_file.write_text(spec_yaml)

    db_path = tmp_path / "runs.duckdb"

    mock_mod = MagicMock()
    mock_mod.signal.side_effect = lambda df, **params: pd.Series(1.0, index=df.index)

    with patch("backtest.load_strategy", return_value=mock_mod), \
         patch("backtest.load_prices", return_value=make_mock_df()), \
         patch("runs_db.DEFAULT_DB_PATH", db_path):
        
        # Test Case 2: Replace data with NSEI
        res = backtest.run_cli([
            str(spec_file),
            "--replace-data", "yahoo:^NSEI",
            "--description", "Testing on Nifty 50",
            "--db-path", str(db_path)
        ])

        conn = runs_db.get_connection(db_path)
        runs = runs_db.get_strategy_runs(conn, "test-strat")
        assert len(runs) == 1
        assert runs[0]["run_type"] == "replacement"
        assert runs[0]["data_source"] == "yahoo:^NSEI"
        assert runs[0]["change_description"] == "Testing on Nifty 50"
        conn.close()


def test_backtest_cli_cross_test_peers(tmp_path: Path):
    import backtest
    import runs_db

    spec_yaml = """
meta:
  slug: test-cross
  title: "Test Cross"
  posted: 2022-01-01
  claim: "Holding test asset"
data:
  universe: [SPY]
  bar: 1d
  start: 2020-01-01
signal:
  definition: "raw = close > 0"
  lookback_bars: 10
  lag_bars: 1
rules:
  entry: "raw"
  exit: "not raw"
  execution_price: next_open
costs:
  commission_bps: 1.0
  slippage_bps: 5.0
validation:
  cost_sweep_bps: [0, 5, 10]
  param_sweep: {}
  multiple_testing:
    n_tested_so_far: 1
"""
    spec_file = tmp_path / "test-cross.yaml"
    spec_file.write_text(spec_yaml)

    families_yaml = """
families:
  equity_test:
    benchmark: "yahoo:SPY"
    peers:
      - id: "yahoo:SPY"
        label: "SPY"
      - id: "yahoo:^NSEI"
        label: "Nifty 50"
"""
    fam_file = tmp_path / "test_families.yaml"
    fam_file.write_text(families_yaml)

    db_path = tmp_path / "runs.duckdb"

    mock_mod = MagicMock()
    mock_mod.signal.side_effect = lambda df, **params: pd.Series(1.0, index=df.index)

    with patch("backtest.load_strategy", return_value=mock_mod), \
         patch("backtest.load_prices", return_value=make_mock_df()):

        res = backtest.run_cli([
            str(spec_file),
            "--cross-test-peers",
            "--families-file", str(fam_file),
            "--family", "equity_test",
            "--db-path", str(db_path)
        ])

        conn = runs_db.get_connection(db_path)
        runs = runs_db.get_strategy_runs(conn, "test-cross")
        assert len(runs) == 2
        sources = {r["data_source"] for r in runs}
        assert "yahoo:SPY" in sources
        assert "yahoo:^NSEI" in sources
        conn.close()
