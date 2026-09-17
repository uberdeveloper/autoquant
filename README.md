# AutoQuant

AutoQuant is a research workspace for turning published quantitative ideas into
clear, falsifiable backtests.

## Quantocracy Lab

[`quantocracy-lab/`](quantocracy-lab/) is the first research pipeline. It
harvests Quantocracy links, fetches article text responsibly, triages ideas for
testability, converts viable claims into explicit strategy specifications, and
backtests them with consistent validation.

Start with the lab's [README](quantocracy-lab/README.md). The reusable project
assets are the Python pipeline, the LLM prompts, strategy-spec template, and
example strategy. Downloaded feeds, article caches, market-price caches, local
tool state, and generated reports stay out of Git.

## Stages

```bash
python3 harvest.py                # [1] latest Quantocracy links -> data/articles.jsonl
python3 fetch.py                  # [2] article text -> data/pages/
python3 triage.py                 # [3] backtestability scores -> data/triage.jsonl
python3 extract.py                # [4] triaged article -> specs/<slug>.yaml
python3 codegen.py                # [5] spec -> strategies/<slug>.py (smoke-tested)
python3 backtest_batch.py         # [6-8] every spec, isolated process, timeout-capped
uv run python backtest.py specs/<slug>.yaml   # [6-8] single spec + report + leaderboard row
python3 catalog.py search rates   # find data series across providers (yahoo/fred/stooq)
```

Every stage is independent and resumable: re-running any command skips work
that already has its output artifact. Specs, strategy code, and
`results/leaderboard.jsonl` are committed; page caches, price caches, and
full reports stay out of Git.

### Swapping the LLM provider

The three LLM stages (triage, extract, codegen) go through `llm.py`. The
provider command defaults to `opencode run` and is overridden with env
vars -- no code edits:

```bash
AUTOQUANT_LLM_CMD="claude -p" python3 triage.py
AUTOQUANT_LLM_CMD="crush run -q {model}" python3 extract.py --model sonnet
```

- `AUTOQUANT_LLM_CMD` -- shlex-split command template; a `{model}` placeholder
  is optional (without it, `--model <model>` is appended when `--model` is set).
- `AUTOQUANT_LLM_EXTRA_ARGS` -- extra CLI arguments, shlex-split, appended to
  every call.
- `--llm-arg ARG` -- per-invocation extra argument, repeatable, supported by
  `triage.py`, `extract.py`, and `codegen.py`.

Any CLI that reads the prompt on stdin (or as a positional arg) and prints the
completion on stdout works; the stages' fence-stripping parsers tolerate
formatting differences. A missing CLI is caught at startup by a preflight
check -- a batch never crashes mid-run because the binary disappeared.

### Data catalog

Universe entries can be bare Yahoo tickers (`SPY`) or catalog ids:
`yahoo:SPY`, `fred:DGS10`, `stooq:spy.us`. Find series with
`python3 catalog.py search <query>`, inspect one with
`python3 catalog.py show <id>`, and pre-warm the cache with
`python3 catalog.py fetch <id>`. The committed seed index is
`catalog.json`; series cache under `data/catalog/`. If a primary provider
fails, an entry's declared fallback (usually a Yahoo/Stooq mirror) is used
automatically. Macro entries note their release lag and whether vintage
(ALFRED) data exists -- loading current-vintage data for backtests has
lookahead caveats the spec's `data.caveats` should state.
