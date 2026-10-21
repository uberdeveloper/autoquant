# Issue #26: offline data shim — serve our price cache/catalog to external code — Implementation Plan

Part of #22 (external strategy-repo lane). Stage `[3d]`, sitting after #25's
`env_repo.py` (which produced per-repo envs + rows in `data/repo_envs.jsonl`) and
before #27's `run_external.py`. This is the stage that makes external repo code
run against **our data** instead of phoning out to Yahoo/FRED.

The lane's contract: external code backtests against our data. Instead of letting
repo code hit `yfinance` / `pandas_datareader` live (rate limits, flakiness,
non-determinism), we inject a stub package via `PYTHONPATH` that serves the
existing `data/catalog/*.csv` cache and `catalog.py` series. The project already
stubs network access in tests (`tests/test_catalog.py::_stub_read_csv`,
`_stub_yahoo`) — this packages that idea for external execution.

## Data reality on this box (read before implementing)

- The cache directory is `data/catalog/` (`catalog.py:26` CACHE_DIR, `cache_path()`
  at `catalog.py:192` → `data/catalog/<source>_<symbol>.csv`). The issue text's
  `data/prices/<ticker>.csv` is **stale** — there is no `data/prices/`; the shim
  serves `data/catalog/`.
- The catalog index is `catalog.json` (`{"version":…, "series":[…]}`, 17 series:
  `yahoo:SPY/QQQ/TLT/GLD/^VIX`, `fred:DGS10/DGS2/…`, each with an optional
  `fallback` mirror id).
- **`data/catalog/` is EMPTY on this machine** (the cache was not migrated with the
  machine move; it is gitignored). So the shim cannot serve *real* cached prices
  here until either (a) the cache is re-warmed via `catalog.load(...)` on a box with
  network, or (b) tests provide their own fixture cache. Design the shim so it works
  against ANY `data/catalog` directory (injectable root), and make the tests build a
  tiny fixture cache under `tmp_path` — never depend on the real (empty) cache. This
  is also exactly what #27 will need (it will run against a warmed cache or a
  run-copy).

## Stage placement and shape

`external_shim/` package at the repo root (importable directory, NOT installed), plus
a thin `shim_repo.py`? — **Decision: keep it a pure package + a builder function, no
new top-level CLI stage.** The shim's job is to be *launched with* `PYTHONPATH`, and
#27 is the stage that launches things. So this issue delivers:

1. `external_shim/` — the injectable package:
   - `external_shim/yfinance/__init__.py` — stub `yfinance` exposing the subset
     external code actually calls: `download(...)` and `Ticker(...).history(...)`.
   - `external_shim/pandas_datareader/__init__.py` + `…/data.py` — stub
     `pandas_datareader.data.DataReader(...)`.
   - Both backed by `catalog.load(...)` (the SAME normalization/fallback/caching the
     project already trusts), served against an injectable cache root.
   - A `DataNotAvailable` error type for cache misses that cannot be satisfied.
2. `shim.py` at repo root — the builder/launcher seam #27 imports:
   - `shim_env(catalog_root, extra_path=None) -> dict` — returns the env mapping
     (`PYTHONPATH=<external_shim>:<extra>…`) a subprocess should be launched with, so
     `import yfinance` / `import pandas_datareader` resolve to the shim, not
     site-packages.
   - `shim_root() -> Path` — the `external_shim/` directory (for PYTHONPATH).
   - A helper to point the shim at a given catalog dir (env var
     `AUTOQUANT_CATALOG_DIR`, defaulting to the repo's `data/catalog`), so a run-copy
     or a warmed cache can be targeted without editing code.

No new dependencies. The shim imports only `pandas` + the project's own `catalog`
(which is already a dependency). argparse/`main_with` not required here (no CLI), but
if a tiny `python -c` smoke is wanted it must not hit the network.

## Symbol / API mapping (the subset external code calls)

- `yf.download("AAPL", start=…, end=…, …)` → resolve the symbol to a catalog id
  (`yahoo:AAPL`), `catalog.load(...)`, return a frame yfinance-shaped enough for
  typical strategy code. Decide and document: return the normalized OHLCV (lowercase
  columns, always `close`) vs yfinance's MultiIndex columns — **prefer the simple
  normalized frame with lowercase OHLCV + a `Close` alias**, documented, because
  external code overwhelmingly reads `Close`/`close`. Pin the exact shape with tests.
- `yf.Ticker(sym).history(period=…, start=…, end=…, interval=…)` → same backing load.
- `pandas_datareader.data.DataReader("DGS10", "fred", …)` / `(sym, "yahoo", …)` →
  route by the `source` argument to the matching catalog id.
- A request whose symbol is **not in the catalog and not in the cache** raises
  `DataNotAvailable` (a clean, catchable error), NOT a hang and NOT a raw traceback —
  #27 captures it into the verdict. This is acceptance item 2.

## Isolation guarantee

`catalog.load` must never reach a provider when the shim is used offline. Two
mechanisms, both documented:
- The shim is launched with the catalog cache warm (or a run-copy that has it); cache
  hits never call a provider.
- When a cache miss would trigger a live fetch, the shim must NOT silently phone out.
  Decide: (a) set an offline flag so `PROVIDERS` calls raise `DataNotAvailable`, or
  (b) rely on the process-level `unshare -n` (see limitation). **Implement (a)** as
  the default deterministic behavior, and document `unshare -n` as the optional hard
  belt-and-suspenders.

## Known limitation (documented, decided)

`unshare -n` (network-namespace hard-offline) is **not available on this box**
(sandboxed; the handoff records it as blocked). So the "optional hard offline"
wrapper from the issue is shipped as a documented, feature-flagged path that is
expected to work on a normal Linux box but is NOT exercised here. The deterministic
offline guarantee is the shim's own `DataNotAvailable`-on-miss behavior (a), which is
fully testable offline.

## Fallback mode `RUN_BUNDLED_DATA`

For repos that ship complete data (triage verdict `RUN_BUNDLED_DATA`): run in a
**copy of the repo dir** and let it read its own bundled files — **no shim injected**.
This issue delivers the *shim*; the run-copy-and-select-mode decision lives in #27
(`run_external.py` chooses shim vs bundled per artifact). Document the contract here so
#27 knows `shim_env()` is only invoked for `RUN_WITH_OUR_DATA` artifacts. The
"original clone never mutated — execution happens in a run-copy" guarantee is enforced
by #27 (it owns the copy step); this stage must not write into `data/repos/`.

## Acceptance mapping (how we prove each checkbox)

- [ ] **A notebook calling `yf.download("AAPL", …)` executes offline and returns our
  cached prices** → test builds a fixture `data/catalog/yahoo_AAPL.csv` under
  `tmp_path`, sets `AUTOQUANT_CATALOG_DIR`, calls the stub `download("AAPL", …)`
  with providers forced to raise (offline), asserts the returned frame matches the
  cached prices (normalized, `close` present). No network.
- [ ] **Cache misses produce a recorded, actionable verdict instead of a hang or
  traceback** → test requests an unknown symbol with offline mode on, asserts
  `DataNotAvailable` is raised (catchable, with the symbol + a hint in the message),
  and that no provider call was attempted (monkeypatched provider records zero calls).
  #27 turns this into the verdict row; here we prove the error type + no-hang.
- [ ] **Original repo clone never mutated — execution in a run-copy** → out of this
  stage's executable scope (owned by #27); documented here as the contract, and this
  stage provably never writes to `data/repos/` (a test asserts the shim build touches
  nothing under the repos dir). Recorded honestly in the PR.

## TDD test list (tests/test_shim.py, mirrors `shim.py` + `external_shim/`)

1. `shim_env()` returns `PYTHONPATH` prefixed with the `external_shim` dir and keeps
   any `extra_path`; the value is a real directory containing `yfinance/` and
   `pandas_datareader/`.
2. With the shim root on `PYTHONPATH`, `import yfinance` resolves to the SHIM module
   (assert `yfinance.__file__` is under `external_shim/`), not site-packages.
3. `yf.download` serves fixture cached prices offline (acceptance 1) — normalized
   frame, lowercase OHLCV + `Close` alias, DatetimeIndex ascending, filtered by
   start/end.
4. `yf.Ticker(sym).history(...)` returns the same backing data as `download`.
5. `pandas_datareader.data.DataReader(sym, "fred"|"yahoo", …)` routes to the right
   catalog id and returns the cached series.
6. Unknown symbol + offline → `DataNotAvailable` raised, message names the symbol,
   zero provider calls (acceptance 2).
7. Offline flag actually blocks providers: with a cache MISS and offline on, no
   `PROVIDERS[source]` call fires (record calls via monkeypatch).
8. The shim honors `AUTOQUANT_CATALOG_DIR` (injectable cache root) — point it at two
   different fixture caches, get different data.
9. The shim build/launch writes NOTHING under `data/repos/` (isolation guard).
10. `unshare -n` path is documented/feature-flagged but skipped when unavailable
    (`pytest.mark.skipif` on the probe), so the suite stays green offline here.

## Hard constraints (carry-over from the lane)

- `uv` for everything; NO new dependencies; no daemons; no executable bits; no
  `# type: ignore`; TDD red-first; tests mirror source (`shim.py` → `tests/test_shim.py`;
  the `external_shim/` package's behavior is covered there too); no `conftest.py`
  (fixtures inside the test file); argparse not Typer.
- Tests never hit the network, real LLM, or mutate the real `data/` tree — all fixture
  caches under `tmp_path`, providers monkeypatched at the single call sites.
- Working artifacts (`data/`, `results/`, `specs/`) stay gitignored; never committed.
- `uv.lock` must stay byte-identical: `ffdd71985af44fe4468be5182c05a44a72c3d208ee139ea6be780682f9a60e23`.
- Reuse, never re-implement: the shim calls `catalog.load` / `catalog.cache_path` /
  `catalog.parse_id` rather than re-deriving normalization or cache paths.
