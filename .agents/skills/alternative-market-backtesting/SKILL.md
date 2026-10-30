---
name: alternative-market-backtesting
description: Standard procedure and protocol for testing quantitative strategies across alternative data sources, new asset universes, and peer market families in AutoQuant. Use whenever a user or workflow asks to evaluate a strategy on another instrument, market, or dataset.
license: Proprietary. Same terms as the repository.
compatibility: Requires Python 3, uv, and AutoQuant pipeline tools.
---

# Alternative Market & Cross-Asset Backtesting Protocol

This protocol guides LLM agents and researchers when applying an existing AutoQuant strategy to **alternative data sources, new markets, or peer universes**.

---

## 1. Context & The 3-Case Model

An AutoQuant strategy module in `strategies/<slug>.py` implements the contract:
`signal(df: pd.DataFrame, **params) -> pd.Series | pd.DataFrame`.

When testing this strategy beyond the author's original article, identify which case applies:

1. **Case 1: Baseline Run** (Author's claim on original asset).
2. **Case 2: Asset / Source Replacement** (Testing on a specific substitute asset, e.g. SPY -> Nifty 50 `yahoo:^NSEI`).
3. **Case 3: Peer Market Cross-Testing** (Evaluating robustness across an entire family of analog assets, e.g. global large-cap equities).

---

## 2. Step-by-Step Agent Workflow

When a user asks: *"Run strategy X on Indian equities / Russell 3000 / Crypto / alternative data"*:

### Step 1: Identify Strategy & Target Tickers
1. Locate the strategy spec: `specs/<slug>.yaml`.
2. Determine target data sources using standard catalog syntax:
   - Yahoo Finance: `yahoo:SPY`, `yahoo:^NSEI`, `yahoo:^RUA`, `yahoo:^STOXX50E`, `yahoo:^N225`.
   - FRED (Macro/Rates): `fred:DGS10`, `fred:DGS2`.
   - Stooq: `stooq:spy.us`.
3. If unsure of the identifier, search the catalog:
   ```bash
   uv run python catalog.py search <query>
   ```

### Step 2: Pre-flight Data Verification
Ensure data is available and contains required price columns (e.g. `open` if `execution_price: next_open` is used):
```bash
uv run python catalog.py fetch <source_id>
```

### Step 3: Execute the Alternative Run

#### Option A: Single Instrument Replacement (Case 2)
```bash
uv run python backtest.py specs/<slug>.yaml \
    --replace-data "<source_id>" \
    --description "<Clear explanation of hypothesis or user intent>"
```
*Example:*
```bash
uv run python backtest.py specs/example-200dma-timing.yaml \
    --replace-data "yahoo:^NSEI" \
    --description "Testing 200dma trend overlay on Indian Nifty 50"
```

#### Option B: Peer Family Cross-Testing (Case 3)
If testing across standard global peers, use the family rulebook in `market_families.yaml`:
```bash
uv run python backtest.py specs/<slug>.yaml --cross-test-peers
```
Or specify a specific family:
```bash
uv run python backtest.py specs/<slug>.yaml \
    --cross-test-peers \
    --family equity_large_cap
```

#### Option C: Custom Universe via Family Rulebook
If the user defines a brand new custom universe (e.g. 5 specific crypto tokens or commodities):
1. Create or edit a family rulebook (e.g., `custom_families.yaml` or append to `market_families.yaml`).
2. Run with the custom rulebook:
   ```bash
   uv run python backtest.py specs/<slug>.yaml \
       --cross-test-peers \
       --families-file custom_families.yaml \
       --family my_custom_family
   ```

### Step 4: Query DuckDB & Report Comparative Findings
All runs are automatically persisted to `data/runs.duckdb`.

Query the comparative results:
```bash
uv run python -c "
import runs_db
conn = runs_db.get_connection()
runs = runs_db.get_strategy_runs(conn, '<slug>')
for r in runs[:5]:
    print(f\"{r['data_source']:<15} {r['run_type']:<12} Sharpe: {r['net_sharpe']} CAGR: {r['cagr']}\")
"
```

Summarize the outcome to the user:
- State whether the edge generalized or failed on the new market.
- Compare Out-of-Sample (OOS) Sharpe and Max Drawdown against the baseline.
- Note any market-specific caveats (e.g. currency, trading hours, liquidity).
