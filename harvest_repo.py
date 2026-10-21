"""Harvest (shallow-clone) strategy repos and write an inventory manifest.

CLI usage:
    python3 harvest_repo.py owner/repo [owner/repo ...]
    python3 harvest_repo.py --repos repos.txt

Manifest schema (data/repos/<owner>-<name>/manifest.json):
    {
      "repo":   {"owner":..., "name":..., "url":..., "sha":..., "cloned_at":...},
      "notebooks": [{"path":..., "kernel":..., "cells":..., "has_outputs":...}],
      "scripts":   [{"path":..., "has_main":...}],
      "specs":     [{"path":...}],
      "data":      [{"path":..., "bytes":...}],
      "deps":      {"requirements_txt":..., "environment_yml":..., "setup_py":...},
      "counts":    {"notebooks":..., "scripts":..., "specs":..., "data":...}
    }

Classification rules (deterministic):
- notebooks: *.ipynb; kernel from metadata.kernelspec.name (default "unknown");
  cells = cell count; has_outputs = any cell with non-empty outputs or non-null
  execution_count.
- scripts: *.py; has_main = file contains an `if __name__` guard.
- specs: *.yaml/*.yml (any), plus *.md whose lowercase filename or content
  contains strateg, signal, or backtest.
- data: extensions csv parquet h5 hdf5 pickle pkl xlsx, with byte size.
- deps: presence of requirements.txt, environment.yml, setup.py at repo root.
- Skip .git/, hidden dirs, __pycache__, .venv.

Resume: a repo is skipped only when the manifest inside its directory records
the same owner/name AND the local clone's HEAD still equals the recorded SHA.
A fresh clone happens when the directory is missing, when the directory holds
a different repo (owner/name mismatch), or when the existing clone is broken.
No fetch is performed: picking up upstream commits requires deleting the
clone. data/ is already gitignored.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_EXTS = {"csv", "parquet", "h5", "hdf5", "pickle", "pkl", "xlsx"}
SKIP_DIRS = {".git", "__pycache__", ".venv"}


def parse_repo_spec(spec: str) -> tuple[str, str]:
    if not isinstance(spec, str) or not spec:
        raise ValueError("bad spec")
    if "/" not in spec:
        raise ValueError("bad spec")
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError("bad spec")
    owner, name = parts
    if not owner or not name:
        raise ValueError("bad spec")
    if owner.startswith(".") or name.startswith("."):
        raise ValueError("bad spec")
    if ".." in (owner, name) or owner.startswith("/") or name.startswith("/"):
        raise ValueError("bad spec")
    return owner, name


def repos_from_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def repo_url(owner: str, name: str) -> str:
    return f"https://github.com/{owner}/{name}.git"


def repo_dir(owner: str, name: str, root: Path) -> Path:
    return Path(root) / "data" / "repos" / f"{owner}-{name}"


def run_git(argv: list[str]) -> str:
    """Run a git command, returning stripped stdout; raise RuntimeError on failure.

    Uses a 600-second timeout so a hung connection cannot block a batch run,
    and GIT_TERMINAL_PROMPT=0 so git never waits for credentials.
    """
    try:
        r = subprocess.run(argv, capture_output=True, text=True, check=True,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                           timeout=600)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"git timed out after 600s: {' '.join(argv)}") from e
    return (r.stdout or "").strip()


def clone_repo(owner: str, name: str, dest) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_git(["git", "clone", "--depth", "1", repo_url(owner, name), str(dest)])


def rev_parse(dest: Path) -> str:
    return run_git(["git", "-C", str(dest), "rev-parse", "HEAD"])


def _has_main(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    return "if __name__" in text


def inventory(tree: Path) -> dict:
    tree = Path(tree)
    notebooks: list[dict] = []
    scripts: list[dict] = []
    specs: list[dict] = []
    data: list[dict] = []
    for p in sorted(tree.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(tree)
        # skip hidden/SKIP_DIRS directories at any depth (all but the last
        # component is a directory), plus hidden files at any depth
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.name.startswith("."):
            continue
        ext = p.suffix.lower().lstrip(".")
        if ext == "ipynb":
            try:
                obj = json.loads(p.read_text())
            except Exception:
                obj = {}
            meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
            ks = meta.get("kernelspec", {}) if isinstance(meta, dict) else {}
            kernel = ks.get("name", "unknown") if isinstance(ks, dict) else "unknown"
            if not kernel:
                kernel = "unknown"
            cells = obj.get("cells", []) if isinstance(obj, dict) else []
            has_outputs = False
            for c in cells:
                if not isinstance(c, dict):
                    continue
                outs = c.get("outputs")
                if outs:
                    has_outputs = True
                    break
                ec = c.get("execution_count")
                if ec is not None:
                    has_outputs = True
                    break
            notebooks.append({"path": str(rel), "kernel": kernel,
                             "cells": len(cells), "has_outputs": has_outputs})
        elif ext == "py":
            scripts.append({"path": str(rel), "has_main": _has_main(p)})
        elif ext in ("yaml", "yml"):
            specs.append({"path": str(rel)})
        elif ext == "md":
            name_l = p.name.lower()
            try:
                text_l = p.read_text(errors="replace").lower()
            except OSError:
                text_l = ""
            if any(k in name_l or k in text_l
                   for k in ("strateg", "signal", "backtest")):
                specs.append({"path": str(rel)})
        elif ext in DATA_EXTS:
            data.append({"path": str(rel), "bytes": p.stat().st_size})
    # sort specs? test expects notes.md before strategy.yaml (sorted by path)
    specs.sort(key=lambda x: x["path"])
    notebooks.sort(key=lambda x: x["path"])
    scripts.sort(key=lambda x: x["path"])
    data.sort(key=lambda x: x["path"])
    deps = {
        "requirements_txt": (tree / "requirements.txt").is_file(),
        "environment_yml": (tree / "environment.yml").is_file(),
        "setup_py": (tree / "setup.py").is_file(),
    }
    return {
        "notebooks": notebooks,
        "scripts": scripts,
        "specs": specs,
        "data": data,
        "deps": deps,
        "counts": {
            "notebooks": len(notebooks),
            "scripts": len(scripts),
            "specs": len(specs),
            "data": len(data),
        },
    }


def publish(manifest: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest))
    except OSError:
        tmp.unlink(missing_ok=True)  # never leave a stray .tmp behind
        raise
    tmp.replace(path)


def main_with(argv=None, root=None) -> int:
    if root is None:
        root = ROOT
    root = Path(root)
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="*", default=[])
    parser.add_argument("--repos", dest="repos_file", default=None)
    parser.add_argument("--root", dest="root_override", default=None)
    args = parser.parse_args(argv)
    if args.root_override is not None:
        root = Path(args.root_override)
    specs: list[str] = list(args.specs)
    if args.repos_file:
        specs.extend(repos_from_file(Path(args.repos_file)))
    failed = 0
    for spec in dict.fromkeys(specs):
        try:
            owner, name = parse_repo_spec(spec)
            dest = repo_dir(owner, name, root)
            manifest_path = dest / "manifest.json"
            skip = False
            reclone = False
            if manifest_path.exists():
                try:
                    existing = json.loads(manifest_path.read_text())
                except (json.JSONDecodeError, OSError):
                    existing = None
                rec = existing.get("repo", {}) if isinstance(existing, dict) else {}
                if rec.get("owner") != owner or rec.get("name") != name:
                    reclone = True  # the directory holds a different repo
                elif isinstance(rec.get("sha"), str):
                    try:
                        head = rev_parse(dest)
                    except RuntimeError:
                        reclone = True  # broken clone: a fresh clone repairs it
                    else:
                        if head == rec["sha"]:
                            skip = True
            if skip:
                print(f"skip {owner}/{name}")
                continue
            if reclone and dest.exists():
                shutil.rmtree(dest)
            if not dest.exists():
                clone_repo(owner, name, dest)
            body = inventory(dest)
            manifest = {"repo": {
                "owner": owner,
                "name": name,
                "url": repo_url(owner, name),
                "sha": rev_parse(dest),
                "cloned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }}
            manifest.update(body)
            publish(manifest, manifest_path)
            print(f"harvested {owner}/{name}: {manifest['counts']}")
        except (ValueError, RuntimeError) as e:
            print(f"error {spec}: {e}", file=sys.stderr)
            failed += 1
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_with())
