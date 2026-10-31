#!/usr/bin/env python3
"""Stage [3b] TRIAGE_REPO -- score harvested repo artifacts on runnability.

Runs triage_repo_prompt.md over every notebook, script, and spec recorded in
data/repos/<owner>-<name>/manifest.json (written by harvest_repo.py) via the
LLM CLI configured in llm.py (default: `opencode run`). Swap the provider
with AUTOQUANT_LLM_CMD.

The lane accepts the repo as fact: this stage scores *runnability* -- what is
worth executing and how -- never whether the strategy is sensible or
profitable. Verdicts follow triage_repo_prompt.md:
  RUN_WITH_OUR_DATA   -> runs against data this pipeline already serves
  RUN_BUNDLED_DATA    -> loads its own bundled files (#26/#27 territory)
  NEEDS_HEAVY_ENV     -> heavy deps or pins conflicting with pyproject.toml
  SPEC_ARTIFACT       -> text/yaml with no executable code (routes to #27)
  INTERACTIVE_ONLY    -> needs a human/UI/broker/notebook session
  SKIP                -> nothing runnable

Results land in data/repo_triage.jsonl, one object per artifact, keyed by
(repo, kind, path). Re-running only scores artifacts not already in that
file (--rescore overrides). Error rows still count as scored for resume.

    python3 triage_repo.py                      # every harvested repo
    python3 triage_repo.py Auquan/Tutorials     # one repo (repeatable, nargs="*")
    python3 triage_repo.py --repos list.txt     # one spec per line
    python3 triage_repo.py --limit 20 --jobs 8  # calibrate first
    python3 triage_repo.py --rescore            # rubric changed: re-score all
"""
from __future__ import annotations

import argparse
import concurrent.futures
import harvest_repo
import json
import llm
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOS = ROOT / "data" / "repos"
TRIAGE = ROOT / "data" / "repo_triage.jsonl"
PROMPT = ROOT / "triage_repo_prompt.md"

MAX_ARTIFACT_CHARS = 40_000  # a verdict needs the code, not every cell
MAX_DEPS_CHARS = 4_000  # dep evidence is a pin list, not a novel
CLI_TIMEOUT = 300

VERDICTS = ("RUN_WITH_OUR_DATA", "RUN_BUNDLED_DATA", "NEEDS_HEAVY_ENV",
            "SPEC_ARTIFACT", "INTERACTIVE_ONLY", "SKIP")
DATA_MODES = ("network", "bundled", "both", "none")
DEP_WEIGHTS = ("ordinary", "heavy")
RUNTIME_CLASSES = ("seconds", "minutes", "hours")
LIST_FIELDS = ("symbols", "data_loading", "heavy_deps")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def _key(row: dict) -> tuple[str | None, str | None, str | None]:
    """Resume key: (repo, kind, path); error rows carry the same triple."""
    return (row.get("repo"), row.get("kind"), row.get("path"))


def artifact_rows(manifest: dict, repo: str) -> list[dict]:
    """Scoreable rows for one manifest: notebooks, scripts, specs.

    `repo` is the owner/name label (from the manifest's repo record, never
    the ambiguous directory name). `data` rows are not executable artifacts
    and are not scored; `deps` text rides along in the prompt instead.
    """
    rows: list[dict] = []
    for r in manifest.get("notebooks", []):
        rows.append({"repo": repo, "kind": "notebook", "path": r["path"],
                     "meta": {"kernel": r.get("kernel", "unknown"),
                              "cells": r.get("cells"),
                              "has_outputs": r.get("has_outputs")}})
    for r in manifest.get("scripts", []):
        rows.append({"repo": repo, "kind": "script", "path": r["path"],
                     "meta": {"has_main": r.get("has_main")}})
    for r in manifest.get("specs", []):
        rows.append({"repo": repo, "kind": "spec", "path": r["path"],
                     "meta": {}})
    return rows


def notebook_source(path: Path) -> str:
    """Join the source of a notebook's code and markdown cells.

    Cell `source` may be a string or a list of lines (both legal nbformat);
    raw cells are dropped. Raises OSError or JSONDecodeError when the
    notebook is unreadable -- the caller turns that into an error row.
    """
    obj = json.loads(path.read_text(errors="replace"))
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []
    for cell in obj.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") not in ("code", "markdown"):
            continue
        src = cell.get("source")
        if isinstance(src, list):
            src = "".join(src)
        elif not isinstance(src, str):
            src = ""
        if src:
            parts.append(src)
    return "\n\n".join(parts)


def artifact_text(dest: Path, row: dict) -> str:
    """Content of one artifact inside the clone dir `dest`, by kind."""
    path = dest / row["path"]
    if row["kind"] == "notebook":
        return notebook_source(path)
    if row["kind"] in ("script", "spec"):
        return path.read_text(errors="replace")
    raise ValueError(f"unknown artifact kind: {row['kind']!r}")


def repo_deps(dest: Path, manifest: dict) -> dict:
    """Dep evidence for the prompt: requirements.txt / environment.yml text.

    Only files the manifest flags as present are read; each is truncated to
    MAX_DEPS_CHARS -- dep weight needs the pins, not the whole file.
    """
    flags = manifest.get("deps")
    if not isinstance(flags, dict):
        flags = {}
    out: dict[str, str] = {}
    for key, name in (("requirements_txt", "requirements.txt"),
                      ("environment_yml", "environment.yml")):
        text = ""
        if flags.get(key):
            try:
                text = (dest / name).read_text(errors="replace")
            except OSError:
                text = ""
        if len(text) > MAX_DEPS_CHARS:
            text = text[:MAX_DEPS_CHARS] + "\n\n[...truncated for length...]"
        out[key] = text
    return out


def parse_json(text: str) -> dict:
    """The rubric asks for bare JSON; tolerate a fenced or prose-wrapped reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        raise ValueError("JSON reply must be an object")
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        result = json.loads(text[start:start + end - start + 1])
        if isinstance(result, dict):
            return result
        raise ValueError("JSON reply must be an object")
    raise ValueError(f"no JSON object in reply: {text[:200]!r}")


def build_prompt(rubric: str, row: dict, text: str, deps: dict) -> str:
    if len(text) > MAX_ARTIFACT_CHARS:
        text = text[:MAX_ARTIFACT_CHARS] + "\n\n[...truncated for length...]"
    payload = {"repo": row["repo"], "kind": row["kind"], "path": row["path"],
               "meta": row["meta"], "repo_deps": deps, "text": text}
    return (
        f"{rubric}\n\n"
        "---\n\nHere is the artifact to triage. Return ONLY the JSON object "
        "described above -- no prose, no markdown fences.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )


def validate(result: dict) -> dict:
    """Normalize the model's reply into the row's verdict fields.

    Raises ValueError (message starting `invalid `) on any field outside its
    contract. Those prefixes keep llm.is_cli_error() False, so a bad reply
    can never trip the circuit breaker.
    """
    verdict = result.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r} "
                         f"(expected one of {', '.join(VERDICTS)})")
    data_mode = result.get("data_mode")
    if data_mode not in DATA_MODES:
        raise ValueError(f"invalid data_mode: {data_mode!r} "
                         f"(expected one of {', '.join(DATA_MODES)})")
    dep_weight = result.get("dep_weight")
    if dep_weight not in DEP_WEIGHTS:
        raise ValueError(f"invalid dep_weight: {dep_weight!r} "
                         f"(expected one of {', '.join(DEP_WEIGHTS)})")
    runtime_class = result.get("runtime_class")
    if runtime_class not in RUNTIME_CLASSES:
        raise ValueError(f"invalid runtime_class: {runtime_class!r} "
                         f"(expected one of {', '.join(RUNTIME_CLASSES)})")
    confidence = result.get("confidence")
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1):
        raise ValueError(f"invalid confidence: {confidence!r} "
                         "(expected a number from 0 to 1)")
    lists: dict[str, list[str]] = {}
    for field in LIST_FIELDS:
        value = result.get(field)
        if value is None:
            lists[field] = []
        elif not isinstance(value, list):
            raise ValueError(f"invalid {field}: expected a list of strings, "
                             f"got {value!r}")
        elif not all(isinstance(v, str) for v in value):
            raise ValueError(f"invalid {field}: entries must be strings, "
                             f"got {value!r}")
        else:
            lists[field] = [v for v in value if v.strip()]
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("invalid reason: expected a non-empty string")
    return {"verdict": verdict,
            "symbols": lists["symbols"],
            "data_loading": lists["data_loading"],
            "data_mode": data_mode,
            "dep_weight": dep_weight,
            "heavy_deps": lists["heavy_deps"],
            "runtime_class": runtime_class,
            "confidence": float(confidence),
            "reason": reason.strip()}


def score_one(dest: Path, row: dict, rubric: str, model: str | None,
              extra: list[str] | None = None) -> dict:
    """Score one artifact; failures become {repo,kind,path,meta,error} rows.

    Stage-level errors (`unparseable reply:`, `invalid ...`,
    `unreadable artifact:`) are deliberately not CLI-level, so llm's circuit
    breaker only ever trips on genuine CLI plumbing failures (LLMError).
    """
    base = {"repo": row["repo"], "kind": row["kind"], "path": row["path"],
            "meta": row["meta"]}
    try:
        text = artifact_text(dest, row)
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "error": f"unreadable artifact: {exc}"[:300]}
    try:
        manifest = json.loads((dest / "manifest.json").read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    prompt = build_prompt(rubric, row, text, repo_deps(dest, manifest))

    try:
        reply = llm.complete(prompt, model=model, extra=extra,
                             timeout=CLI_TIMEOUT)
    except llm.LLMError as exc:
        return {**base, "error": str(exc)}

    try:
        result = parse_json(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        return {**base, "error": f"unparseable reply: {exc}"[:300]}

    try:
        result = validate(result)
    except ValueError as exc:
        return {**base, "error": str(exc)[:300]}

    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds") \
        .replace("+00:00", "Z")
    return {**base, **result, "scored_at": scored_at,
            "model": model or "default"}


def repo_manifests(root: Path, specs: list[str]) -> list[tuple[Path, dict, str]]:
    """Locate (dest, manifest, label) for every repo to score.

    With specs: exactly those repos, deduped, each parsed with
    harvest_repo.parse_repo_spec and required to have a manifest
    (ValueError otherwise). Without specs: every child of <root>/data/repos/
    holding a manifest, sorted. The label always comes from the manifest's
    repo record -- the directory name is ambiguous for colliding specs
    (known #23 limitation) -- and a manifest without owner/name is skipped.
    A corrupt manifest raises JSONDecodeError (a ValueError) for main_with.
    """
    out: list[tuple[Path, dict, str]] = []

    def add(dest: Path, spec: str | None) -> None:
        manifest_path = dest / "manifest.json"
        if spec is not None and not manifest_path.exists():
            raise ValueError(f"no manifest for {spec} -- run harvest_repo.py first")
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        rec = manifest.get("repo") if isinstance(manifest, dict) else None
        if not isinstance(rec, dict):
            return
        owner, name = rec.get("owner"), rec.get("name")
        if not owner or not name:
            return
        out.append((dest, manifest, f"{owner}/{name}"))

    if specs:
        for spec in dict.fromkeys(specs):
            owner, name = harvest_repo.parse_repo_spec(spec)
            add(harvest_repo.repo_dir(owner, name, root), f"{owner}/{name}")
    else:
        repos_dir = Path(root) / "data" / "repos"
        if repos_dir.is_dir():
            for child in sorted(repos_dir.iterdir()):
                if child.is_dir():
                    add(child, None)
    return out


def main_with(argv: list[str] | None = None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="score harvested repo artifacts for runnability (stage [3b])")
    ap.add_argument("specs", nargs="*", metavar="owner/repo",
                    help="harvested repo(s) to score (default: every manifest)")
    ap.add_argument("--repos", dest="repos_file", default=None, metavar="FILE",
                    help="file with one owner/repo spec per line")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent LLM calls")
    ap.add_argument("--model", default=None,
                    help="override the model (default: your session's)")
    ap.add_argument("--llm-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed through to the LLM CLI (repeatable)")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score artifacts already in repo_triage.jsonl")
    args = ap.parse_args(argv)

    try:
        llm.preflight()
    except llm.LLMError as exc:
        sys.exit(f"error: {exc}")

    specs = list(args.specs)
    if args.repos_file:
        try:
            specs.extend(harvest_repo.repos_from_file(Path(args.repos_file)))
        except OSError as exc:
            print(f"error: cannot read {args.repos_file}: {exc}", file=sys.stderr)
            return 1

    try:
        rubric = PROMPT.read_text()
    except OSError as exc:
        print(f"error: cannot read prompt {PROMPT}: {exc}", file=sys.stderr)
        return 1

    try:
        manifests = repo_manifests(ROOT if root is None else Path(root), specs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    todo: list[tuple[Path, dict]] = []
    for dest, manifest, label in manifests:
        todo.extend((dest, row) for row in artifact_rows(manifest, label))

    scored = load(TRIAGE)
    seen = set() if args.rescore else {_key(r) for r in scored}
    todo = [(dest, row) for dest, row in todo if _key(row) not in seen]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to triage -- run harvest_repo.py, or pass --rescore")
        return 0

    print(f"triaging {len(todo)} artifacts ({args.jobs} at a time)\n")
    results: list[dict] = []
    abort = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(score_one, dest, row, rubric, args.model,
                               args.llm_arg): row for dest, row in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if "error" in res:
                print(f"[{i}/{len(todo)}] {'ERR':<17} {res['kind']} "
                      f"{res['repo']}:{res['path']} {res['error'][:60]}")
            else:
                print(f"[{i}/{len(todo)}] {res['verdict']:<17} {res['kind']} "
                      f"{res['repo']}:{res['path']}")
            abort = llm.circuit_break(results)
            if abort:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    # Replace prior entries for re-scored keys rather than appending duplicates.
    fresh = {_key(r) for r in results}
    save(TRIAGE, [r for r in scored if _key(r) not in fresh] + results)

    if abort:
        print(f"\n{abort}")
        return 1

    counts: dict[str, int] = {}
    for r in load(TRIAGE):
        verdict = r.get("verdict", "error")
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\nverdicts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    try:
        shown = TRIAGE.relative_to(ROOT)
    except ValueError:
        shown = TRIAGE
    print(f"scores -> {shown}")
    return 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
