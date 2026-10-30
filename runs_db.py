"""Analytical runs store using DuckDB.

Maintains immutable records of backtest executions across baseline and alternative
data sources linked to their parent strategy spec.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb

ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "data" / "runs.duckdb"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              VARCHAR PRIMARY KEY,
    strategy_slug       VARCHAR NOT NULL,
    run_type            VARCHAR NOT NULL,
    change_description VARCHAR,
    created_at          TIMESTAMP NOT NULL,
    data_source         VARCHAR NOT NULL,
    universe_family     VARCHAR,
    bar_frequency       VARCHAR DEFAULT '1d',
    start_date          VARCHAR,
    end_date            VARCHAR,
    params_json         JSON,
    rules_json          JSON,
    costs_json          JSON,
    cagr                DOUBLE,
    net_sharpe          DOUBLE,
    gross_sharpe        DOUBLE,
    max_drawdown        DOUBLE,
    turnover_annual     DOUBLE,
    win_rate            DOUBLE,
    is_sharpe           DOUBLE,
    oos_sharpe          DOUBLE,
    beats_oos           BOOLEAN,
    beats_null          BOOLEAN,
    benchmark_cagr      DOUBLE,
    benchmark_sharpe    DOUBLE,
    verdict             VARCHAR,
    raw_metrics_json    JSON,
    report_path         VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_runs_slug ON runs (strategy_slug);
CREATE INDEX IF NOT EXISTS idx_runs_type ON runs (run_type);
CREATE INDEX IF NOT EXISTS idx_runs_source ON runs (data_source);
CREATE INDEX IF NOT EXISTS idx_runs_family ON runs (universe_family);
"""


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open or create a DuckDB connection to the runs store."""
    p = Path(db_path) if db_path else DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(p))
    return conn


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize schema and indexes if not present."""
    conn.execute(SCHEMA_SQL)


def record_run(conn: duckdb.DuckDBPyConnection, run: dict[str, Any]) -> None:
    """Insert or update a backtest run record."""
    init_db(conn)
    created_at = run.get("created_at")
    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat()
    elif isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    cols = [
        "run_id", "strategy_slug", "run_type", "change_description", "created_at",
        "data_source", "universe_family", "bar_frequency", "start_date", "end_date",
        "params_json", "rules_json", "costs_json", "cagr", "net_sharpe", "gross_sharpe",
        "max_drawdown", "turnover_annual", "win_rate", "is_sharpe", "oos_sharpe",
        "beats_oos", "beats_null", "benchmark_cagr", "benchmark_sharpe", "verdict",
        "raw_metrics_json", "report_path"
    ]

    params_json = json.dumps(run.get("params_json", {})) if isinstance(run.get("params_json"), (dict, list)) else run.get("params_json")
    rules_json = json.dumps(run.get("rules_json", {})) if isinstance(run.get("rules_json"), (dict, list)) else run.get("rules_json")
    costs_json = json.dumps(run.get("costs_json", {})) if isinstance(run.get("costs_json"), (dict, list)) else run.get("costs_json")
    raw_metrics_json = json.dumps(run.get("raw_metrics_json", {})) if isinstance(run.get("raw_metrics_json"), (dict, list)) else run.get("raw_metrics_json")

    values = [
        run["run_id"],
        run["strategy_slug"],
        run.get("run_type", "original"),
        run.get("change_description"),
        created_at,
        run["data_source"],
        run.get("universe_family"),
        run.get("bar_frequency", "1d"),
        str(run.get("start_date")) if run.get("start_date") else None,
        str(run.get("end_date")) if run.get("end_date") else None,
        params_json,
        rules_json,
        costs_json,
        run.get("cagr"),
        run.get("net_sharpe"),
        run.get("gross_sharpe"),
        run.get("max_drawdown"),
        run.get("turnover_annual"),
        run.get("win_rate"),
        run.get("is_sharpe"),
        run.get("oos_sharpe"),
        run.get("beats_oos"),
        run.get("beats_null"),
        run.get("benchmark_cagr"),
        run.get("benchmark_sharpe"),
        run.get("verdict"),
        raw_metrics_json,
        run.get("report_path")
    ]

    placeholders = ", ".join(["?"] * len(cols))
    sql = f"""
    INSERT OR REPLACE INTO runs ({', '.join(cols)})
    VALUES ({placeholders})
    """
    conn.execute(sql, values)


def get_strategy_runs(conn: duckdb.DuckDBPyConnection, slug: str) -> list[dict[str, Any]]:
    """Retrieve all runs associated with a strategy slug, newest first."""
    init_db(conn)
    cursor = conn.execute(
        "SELECT * FROM runs WHERE strategy_slug = ? ORDER BY created_at DESC", [slug]
    )
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_peer_matrix(conn: duckdb.DuckDBPyConnection, slug: str) -> list[dict[str, Any]]:
    """Retrieve peer cross-runs for a given strategy."""
    init_db(conn)
    cursor = conn.execute(
        """SELECT data_source, universe_family, net_sharpe, cagr, max_drawdown, beats_oos, verdict
           FROM runs 
           WHERE strategy_slug = ? AND run_type IN ('original', 'peer_cross', 'replacement')
           ORDER BY created_at ASC""",
        [slug]
    )
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]
