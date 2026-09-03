# Stage 4 — Extract one article into a strategy specification

## Role and output contract

Convert one already-triaged article into a reproducible strategy specification. First read _TEMPLATE.yaml; it is the required schema and its comments define the contract. Produce exactly one valid YAML document with the same keys and nesting as that template. Output YAML only—no Markdown fence, analysis, or text before or after it.

The input is one article record and its full text:

```json
{"title":"...","source":"...","url":"...","posted":"YYYY-MM-DD","text":"full extracted article text"}
```

Use the article text as evidence. The triage score is a routing decision, not evidence for the specification.

## Extraction procedure

Work through the article in this order before writing YAML:

1. Identify the author's exact claim, universe, sample period, and reported metrics. Record reported metrics only in meta.author_evidence.
2. Extract the deterministic signal, entry, exit, direction, sizing, rebalance rule, and execution timing.
3. Distinguish every directly stated rule from every missing rule.
4. Set safe defaults only for missing details, then record each such decision in ambiguities with alternatives and materiality.
5. Check for lookahead, survivorship, unavailable data, fitted parameters, and a mismatch between the author's universe and the available data.
6. Populate every required template field. Use YAML null only where the template permits an unknown or unavailable value.

## Evidence and ambiguity rules

Never invent a core rule. A default is allowed only for an operational detail that the article leaves open, and it must have a corresponding ambiguities entry. Each ambiguity includes:

- a stable snake_case id;
- the unanswered question;
- the chosen default;
- realistic alternatives; and
- material: true when another reasonable choice could change the conclusion.

Use these defaults only when the article is silent:

| Missing detail | Default |
| --- | --- |
| Close-based signal execution | next_open with lag_bars: 1 |
| Trading cost | 1 bp commission and 5 bp slippage per side |
| Flat allocation | cash_zero |
| Sizing | full_notional |
| Direction | long_only |
| Moving-average or volatility variant | simple |
| Universe | exact named tickers only; never expand it |

Copy the article's sample into meta.author_evidence.sample, but use the earliest usable date for data.start. Set validation.oos_start to meta.posted; publication date is the out-of-sample boundary.

## Safety checks

Before output, verify all of the following:

- signal.definition uses only data available at the decision bar.
- lag_bars and rules.execution_price cannot introduce same-bar close lookahead.
- Every non-default value is supported by the article; every default appears in ambiguities.
- meta.claim is one falsifiable statement rather than a description of a chart or a conclusion.
- data.universe matches the article exactly, and data.caveats flags survivorship, point-in-time, adjustment, or data-availability limitations.
- Costs are nonzero unless the strategy genuinely cannot trade.
- The resulting YAML parses and preserves the template's required structure.

## Untestable articles

If the article has no deterministic rule after careful reading, do **not** produce a runnable strategy specification. Instead output only this YAML document:

```yaml
meta:
  slug: "derived-from-title"
  url: "article-url"
  posted: YYYY-MM-DD
verdict:
  status: UNTESTABLE
  reason: "One concise evidence-based sentence."
```

Do not use this exception for a merely incomplete rule: record manageable missing operational details as ambiguities in a full specification.
