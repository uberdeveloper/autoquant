import pytest
from pathlib import Path

def test_init_db(tmp_path: Path):
    import runs_db

    db_path = tmp_path / "test_runs.duckdb"
    conn = runs_db.get_connection(db_path)
    runs_db.init_db(conn)

    # Check table existence
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    assert "runs" in table_names
    conn.close()


def test_record_and_get_runs(tmp_path: Path):
    import runs_db

    db_path = tmp_path / "test_runs.duckdb"
    conn = runs_db.get_connection(db_path)
    runs_db.init_db(conn)

    sample_run = {
        "run_id": "test-strat--yahoo_SPY--1",
        "strategy_slug": "test-strat",
        "run_type": "original",
        "change_description": "Baseline run",
        "data_source": "yahoo:SPY",
        "universe_family": "equity_large_cap",
        "bar_frequency": "1d",
        "start_date": "2020-01-01",
        "end_date": "2023-01-01",
        "params_json": {"lookback": 200},
        "rules_json": {"execution_price": "next_open"},
        "costs_json": {"commission_bps": 1.0},
        "cagr": 0.10,
        "net_sharpe": 0.85,
        "gross_sharpe": 0.90,
        "max_drawdown": -0.15,
        "turnover_annual": 2.5,
        "win_rate": 0.55,
        "is_sharpe": 0.80,
        "oos_sharpe": 0.88,
        "beats_oos": True,
        "beats_null": True,
        "benchmark_cagr": 0.08,
        "benchmark_sharpe": 0.60,
        "verdict": "REPLICATED",
        "raw_metrics_json": {},
        "report_path": "reports/test-strat.md",
    }

    runs_db.record_run(conn, sample_run)

    runs = runs_db.get_strategy_runs(conn, "test-strat")
    assert len(runs) == 1
    assert runs[0]["run_id"] == "test-strat--yahoo_SPY--1"
    assert runs[0]["net_sharpe"] == 0.85
    assert runs[0]["data_source"] == "yahoo:SPY"

    # Test idempotence (upsert/replace)
    sample_run["net_sharpe"] = 0.92
    runs_db.record_run(conn, sample_run)
    updated_runs = runs_db.get_strategy_runs(conn, "test-strat")
    assert len(updated_runs) == 1
    assert updated_runs[0]["net_sharpe"] == 0.92

    conn.close()
