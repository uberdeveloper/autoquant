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
```

Every stage is independent and resumable: re-running any command skips work
that already has its output artifact. Specs, strategy code, and
`results/leaderboard.jsonl` are committed; page caches, price caches, and
full reports stay out of Git.
