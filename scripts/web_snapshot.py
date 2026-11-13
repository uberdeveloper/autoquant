#!/usr/bin/env python3
"""WEB SNAPSHOT — build the frontend's landing and workspace snapshots.

Writes ``frontend/src/content/snapshot.json`` (landing data) and
``frontend/src/content/workspace.json`` (workspace data) from real
artifacts. Keeping them separate means the landing does not eagerly ship
the workspace payload (#39).

Pure readers over the pipeline's output artifacts (jsonl/yaml/md). Never
executes the pipeline, never fabricates content: a fresh install produces a
valid all-empty snapshot and the landing renders its empty states.

    uv run python scripts/web_snapshot.py [--out frontend/src/content/snapshot.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "src" / "content" / "snapshot.json"

MAX_BLURB = 400
MAX_ARTIFACT = 1200
MAX_ARCHIVE = 12
MAX_CHAINS = 3

STAGES = [
    ("01", "Discover research",
     "Pull candidate strategy write-ups from Quantocracy, Cboe Insights and "
     "arXiv q-fin feeds.", "data/articles.jsonl"),
    ("02", "Fetch source material",
     "Extract and cache the full article text behind every discovered link.",
     "data/pages/"),
    ("03", "Evaluate testability",
     "Score every claim: is it a testable hypothesis, and do we have the data?",
     "data/triage.jsonl"),
    ("04", "Extract a specification",
     "Turn a triaged claim into an explicit YAML spec — universe, signal, "
     "rules, costs, validation.", "specs/"),
    ("05", "Generate strategy code",
     "Generate Python for the spec and smoke-test it before anything runs.",
     "strategies/"),
    ("06", "Execute an isolated backtest",
     "Run every spec in a timeout-capped subprocess against the local price "
     "cache.", "results/leaderboard.jsonl"),
    ("07", "Compare and inspect",
     "Rank the evidence on one explainable metric and read the full reports.",
     "reports/"),
]

METRIC_LABELS = re.compile(
    r"^(CAGR|Sharpe|max DD|t-stat|exposure|trades|win rate|profit factor|"
    r"Gross Sharpe|Deflated Sharpe)$", re.I)


def _finite_json(value: object) -> object:
    """Represent undefined numerical evidence as JSON null, including nested fields."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = _finite_json(json.loads(line))
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit - 2].rsplit(" ", 1)[0] + " …"


def report_metrics(report_path: Path) -> list[dict]:
    """Metric rows from a report's Headline table: label / strategy / benchmark."""
    if not report_path.exists():
        return []
    metrics = []
    for line in report_path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and METRIC_LABELS.match(cells[0]):
            metrics.append({"label": cells[0], "strategy": cells[1],
                            "benchmark": cells[2]})
    return metrics


def _artifact_path(root: Path, directory: str, filename: str) -> Path | None:
    """Only read files contained in their artifact directory, including symlinks."""
    base = root.resolve() / directory
    path = (base / filename).resolve()
    return path if path.is_relative_to(base) and path.is_file() else None


def _archive_row_article(row: dict, root: Path) -> dict | None:
    slug = row.get("slug")
    if not slug:
        return None
    source = "research lane"
    spec_path = _artifact_path(root, "specs", f"{slug}.yaml")
    if spec_path is not None:
        try:
            spec = yaml.safe_load(spec_path.read_text())
            meta = spec.get("meta") if isinstance(spec, dict) else None
            if isinstance(meta, dict):
                source = meta.get("source") or source
        except yaml.YAMLError:
            pass
    return {"slug": slug, "source": source, "lane": "article",
            "status": row.get("verdict"), "netSharpe": row.get("net_sharpe"),
            "beatsNull": row.get("beats_null_95"),
            "date": (row.get("run_at") or "")[:10],
            "runAt": row.get("run_at") or ""}


def _external_identity(row: dict) -> dict[str, str]:
    """Accept current runner fields and snapshots from the original runner."""
    path = row.get("path") or row.get("artifact") or ""
    repo = row.get("repo") or ""
    return {"slug": f"{repo}/{path}".strip("/"), "repo": repo,
            "artifactPath": path,
            "artifactKey": row.get("artifact_key") or
            f"{repo}|{row.get('kind') or ''}|{path}"}


def _archive_row_external(row: dict) -> dict | None:
    if not row.get("repo"):
        return None
    return {"slug": _external_identity(row)["slug"],
            "source": row["repo"], "lane": "external",
            "status": row.get("verdict"), "netSharpe": None,
            "beatsNull": None, "date": (row.get("run_at") or "")[:10],
            "runAt": row.get("run_at") or ""}


def _spec_summary(path: Path, has_code: bool) -> dict | None:
    """Structured fields and complete raw YAML for the Strategy Lab."""
    try:
        raw = path.read_text(errors="ignore")
        spec = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(spec, dict) or any(
            not isinstance(spec.get(key, {}), dict)
            for key in ("meta", "data", "signal", "rules", "verdict")):
        return None
    meta = spec.get("meta") or {}
    data = spec.get("data") or {}
    signal = spec.get("signal") or {}
    rules = spec.get("rules") or {}
    verdict = spec.get("verdict") or {}
    universe = data.get("universe")
    if isinstance(universe, str):
        universe = [universe]
    if not isinstance(universe, list):
        universe = []
    return {
        "slug": meta.get("slug") or path.stem,
        "title": meta.get("title"),
        "claim": meta.get("claim"),
        "universe": universe,
        "bar": data.get("bar"),
        "signal": signal.get("definition"),
        "entry": rules.get("entry"),
        "exit": rules.get("exit"),
        "direction": rules.get("direction"),
        "status": verdict.get("status"),
        "hasCode": has_code,
        "raw": raw,
        "rawTruncated": False,
    }


def _workspace(root: Path, articles: list[dict], triage: list[dict],
               lead: list[dict], external: list[dict],
               spec_paths: list[Path]) -> dict:
    has_code = {p.stem for p in
                (root / "strategies").glob("*.py")} if (root / "strategies").exists() else set()
    specs = [s for s in (_spec_summary(p, p.stem in has_code)
                         for p in spec_paths) if s]

    runs = [dict(r, slug=r.get("slug"), lane="article",
                 status=r.get("verdict"), netSharpe=r.get("net_sharpe"),
                 beatsNull=r.get("beats_null_95"),
                 date=(r.get("run_at") or "")[:10],
                 runAt=r.get("run_at") or "",
                 flags=r.get("flags") or [], error=None)
            for r in lead]
    runs += [{**_external_identity(r),
              "lane": "external", "status": r.get("verdict"),
              "netSharpe": None, "beatsNull": None,
              "date": (r.get("run_at") or "")[:10],
              "runAt": r.get("run_at") or "",
              "flags": [], "error": (r.get("reason") or r.get("error") or "")[:300]}
             for r in external]

    reports = {}
    for path in (root / "reports").glob("*.md") if (root / "reports").exists() else []:
        if _artifact_path(root, "reports", path.name) is None:
            continue
        reports[path.stem] = {"metrics": report_metrics(path),
                              "text": path.read_text(errors="ignore")}

    failed = []
    # Live failures = current .error markers. results/failed.jsonl is
    # append-only history — entries there often succeed in later runs, so
    # reporting it directly shows stale failures.
    history = {Path(r.get("spec") or "x").stem.replace(".yaml", ""):
               (r.get("error") or "")[:300]
               for r in load_jsonl(root / "results" / "failed.jsonl")}
    strategies = root / "strategies"
    for marker in (strategies.glob("*.error")
                   if strategies.exists() else []):
        slug = marker.stem
        try:
            message = (json.loads(marker.read_text()).get("error") or "")[:300]
        except (json.JSONDecodeError, OSError):
            message = ""
        failed.append({"slug": slug,
                       "error": message or history.get(slug, "codegen failed")})

    catalog = []
    catalog_path = root / "catalog.json"
    if catalog_path.exists():
        try:
            for e in json.loads(catalog_path.read_text()).get("series", []):
                if not e.get("id"):
                    continue
                catalog.append({
                    "id": e["id"], "title": e.get("title"),
                    "provider": e["id"].split(":")[0],
                    "tags": e.get("tags") or [],
                    "frequency": e.get("frequency"),
                    "coverageStart": e.get("coverage_start"),
                    "license": e.get("license"),
                    "notes": e.get("notes"),
                    "fallback": e.get("fallback"),
                })
        except (json.JSONDecodeError, OSError):
            pass

    triage_by_slug = {t.get("slug"): t for t in triage}
    article_rows = [{
        "slug": a.get("slug"), "title": a.get("title"),
        "source": a.get("source"), "url": a.get("url"),
        "posted": a.get("posted"), "stage": a.get("stage"),
        "blurb": truncate(a.get("blurb") or "", 400),
        "triageScore": a.get("triage_score")
        if a.get("triage_score") is not None
        else (triage_by_slug.get(a.get("slug")) or {}).get("score"),
    } for a in articles]

    triage_rows = [{
        "slug": t.get("slug"), "title": t.get("title"), "url": t.get("url"),
        "claim": truncate(t.get("claim") or "", 300),
        "assetClass": t.get("asset_class"), "score": t.get("score"),
        "redFlags": t.get("red_flags") or [], "missing": t.get("missing") or [],
        "reason": truncate(t.get("reason") or "", 300),
    } for t in triage if t.get("slug")]

    return {"articles": article_rows, "triage": triage_rows, "specs": specs,
            "runs": runs, "reports": reports, "failed": failed,
            "catalog": catalog, "catalogCount": len(catalog)}


def build_snapshot(root: Path) -> dict:
    articles = load_jsonl(root / "data" / "articles.jsonl")
    triage = load_jsonl(root / "data" / "triage.jsonl")
    lead = load_jsonl(root / "results" / "leaderboard.jsonl")
    external = load_jsonl(root / "results" / "external" / "leaderboard.jsonl")
    specs = sorted(p for p in (root / "specs").glob("*.yaml")
                   if _artifact_path(root, "specs", p.name) is not None)
    by_slug = {a.get("slug"): a for a in articles}

    archive = [r for r in (_archive_row_article(r, root) for r in lead) if r]
    archive += [r for r in (_archive_row_external(r) for r in external) if r]
    archive.sort(key=lambda r: r["runAt"], reverse=True)
    archive = archive[:MAX_ARCHIVE]

    chains = []
    for row in lead:
        slug = row.get("slug")
        report_path = _artifact_path(root, "reports", f"{slug}.md")
        if not slug or report_path is None:
            continue
        article = by_slug.get(slug, {})
        spec_path = _artifact_path(root, "specs", f"{slug}.yaml")
        code_path = _artifact_path(root, "strategies", f"{slug}.py")
        chains.append({
            "slug": slug, "title": article.get("title") or slug,
            "stages": [
                {"kind": "article", "title": article.get("title") or slug,
                 "body": truncate(article.get("blurb") or
                                  "(article not in cache)", MAX_BLURB),
                 "repoPath": f"data/articles.jsonl#{slug}"},
                {"kind": "spec", "title": f"specs/{slug}.yaml",
                 "body": truncate(spec_path.read_text(errors="ignore")
                                  if spec_path is not None else "(no spec yet)",
                                  MAX_ARTIFACT),
                 "repoPath": f"specs/{slug}.yaml"},
                {"kind": "code", "title": f"strategies/{slug}.py",
                 "body": truncate(code_path.read_text(errors="ignore")
                                  if code_path is not None
                                  else "(no generated code yet)", MAX_ARTIFACT),
                 "repoPath": f"strategies/{slug}.py"},
                {"kind": "report", "title": f"reports/{slug}.md",
                 "body": truncate(report_path.read_text(errors="ignore"),
                                  MAX_ARTIFACT),
                 "repoPath": f"reports/{slug}.md"},
            ]})
        if len(chains) >= MAX_CHAINS:
            break

    headline = None
    for row in reversed(lead):
        report_path = _artifact_path(root, "reports", f"{row.get('slug')}.md")
        metrics = report_metrics(report_path) if report_path is not None else []
        if metrics:
            headline = {"slug": row["slug"], "metrics": metrics}
            break

    catalog_path = root / "catalog.json"
    providers: list[str] = []
    series = 0
    if catalog_path.exists():
        try:
            entries = json.loads(catalog_path.read_text()).get("series", [])
            series = len(entries)
            providers = sorted({e.get("id", "").split(":")[0]
                                for e in entries if e.get("id")})
        except (json.JSONDecodeError, OSError):
            pass

    stage_counts = [len(articles), len(list((root / "data/pages").glob("*.md"))),
                    len(triage), len(specs),
                    len([p for p in (root / "strategies").glob("*.py")
                         if p.name != "__init__.py"]),
                    len(lead) + len(external),
                    len(list((root / "reports").glob("*.md")))]

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {"articles": len(articles), "triaged": len(triage),
                   "specs": len(specs), "backtests": len(lead),
                   "external": len(external)},
        "stages": [{"index": i, "name": n, "description": d, "artifact": a,
                    "count": count,
                    "doc": "https://github.com/uberdeveloper/autoquant#readme"}
                   for (i, n, d, a), count in zip(STAGES, stage_counts)],
        "chains": chains,
        "archive": archive,
        "headline": headline,
        "catalog": {"series": series, "providers": providers},
        "workspace": _workspace(root, articles, triage, lead, external, specs),
    }


def main_with(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="repo root to read")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workspace-out", default=None,
                    help="workspace payload (default: workspace.json beside --out)")
    args = ap.parse_args(argv)
    snap = build_snapshot(Path(args.root))
    workspace = snap.pop("workspace", {})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    ws_out = (Path(args.workspace_out) if args.workspace_out
              else out.with_name("workspace.json"))
    ws_out.parent.mkdir(parents=True, exist_ok=True)
    ws_out.write_text(json.dumps(workspace, ensure_ascii=False, indent=1,
                                 allow_nan=False) + "\n")
    print(f"snapshot -> {out} ({sum(snap['counts'].values())} artifacts)")
    print(f"workspace -> {ws_out}")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
