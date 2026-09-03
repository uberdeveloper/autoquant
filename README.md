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
