# Issue #23: repo ingestion — shallow-clone strategy repos + inventory manifest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New stage-1 script `harvest_repo.py` that shallow-clones a GitHub strategy repo and writes a machine-readable `manifest.json` inventory, mirroring the existing harvesters (append-to-artifact, resumable, no-op on re-run at the recorded SHA).

**Architecture:** One CLI driver plus pure functions. `run_git()` is the only place `subprocess` is invoked (single monkeypatch point for tests — no network in the suite). `inventory()` is a pure walk over a directory tree. `publish()` writes `manifest.json` atomically (tmp + `replace`), matching every other stage's publish contract. The artifact is the state: a repo present at its recorded SHA is skipped; a changed or missing SHA triggers re-clone + re-inventory.

**Tech Stack:** Python 3.11+ stdlib (argparse, json, subprocess, pathlib), pytest. No new dependencies.

**Repo under change:** `/home/pi/autoquant` (branch off `main`, branch `issue23-harvest-repo`). This plan document is committed into the repo at `docs/plans/2026-10-11-issue23-harvest-repo.md` so it is versioned next to the code it changes.

**Upstream context:** part of #22 (external strategy-repo lane). Nothing else in the repo depends on `harvest_repo.py` yet — #24 (triage_repo.py) will consume `manifest.json` and is implemented after this issue is merged.

---

## Manifest schema (documented in the module docstring)

```json
{
  "repo":   {"owner": "LastAncientOne", "name": "Stock_Analysis_For_Quant",
             "url": "https://github.com/LastAncientOne/Stock_Analysis_For_Quant.git",
             "sha": "<40-hex commit>", "cloned_at": "<ISO8601 UTC>"},
  "notebooks": [{"path": "01_intro.ipynb", "kernel": "python3",
                 "cells": 12, "has_outputs": true}],
  "scripts":   [{"path": "backtest.py", "has_main": true}],
  "specs":     [{"path": "strategy.yaml"}],
  "data":      [{"path": "data/spy.csv", "bytes": 12345}],
  "deps":      {"requirements_txt": true, "environment_yml": false, "setup_py": false},
  "counts":    {"notebooks": 1, "scripts": 1, "specs": 1, "data": 1}
}
```

Classification rules (documented, deterministic):
- **notebooks**: `*.ipynb`; `kernel` from `metadata.kernelspec.name` (default `"unknown"`); `cells` = cell count; `has_outputs` = any cell with non-empty `outputs` or non-null `execution_count`.
- **scripts**: `*.py`; `has_main` = file contains an `if __name__` guard.
- **specs**: `*.yaml`/`*.yml` (any), plus `*.md` whose lowercase filename contains `strateg`, `signal`, or `backtest`.
- **data**: extensions `csv parquet h5 hdf5 pickle pkl xlsx`, with byte size.
- **deps**: presence of `requirements.txt`, `environment.yml`, `setup.py` at the repo root.
- Skip `.git/`, hidden dirs, `__pycache__`, `.venv`.

---

### Task 1: `harvest_repo.py` — clone, SHA pinning, inventory, publish

**Files:**
- Create: `harvest_repo.py`
- Test: `tests/test_harvest_repo.py`

- [ ] **Step 1: Write the failing tests** (create `tests/test_harvest_repo.py`)

```python
"""Tests for harvest_repo.py — clone argv, SHA-pinned resume, inventory, publish."""
from __future__ import annotations

import json
import subprocess

import pytest

import harvest_repo


def make_repo(tmp_path):
    """A local stand-in for a cloned repo (tests never touch the network)."""
    repo = tmp_path / "src-repo"
    (repo / "data").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "backtest.py").write_text("if __name__ == '__main__':\n    pass\n")
    (repo / "strategy.yaml").write_text("signal: {}\n")
    (repo / "README.md").write_text("# readme\n")
    (repo / "notes.md").write_text("strategy notes\n")
    nb = {"metadata": {"kernelspec": {"name": "python3"}},
          "cells": [{"cell_type": "code", "outputs": [], "execution_count": None,
                     "source": ["print(1)"]}]}
    (repo / "01.ipynb").write_text(json.dumps(nb))
    (repo / "data" / "spy.csv").write_text("date,close\n2020-01-01,100\n")
    (repo / "requirements.txt").write_text("pandas\n")
    return repo


@pytest.fixture
def cloned(monkeypatch, tmp_path):
    """Redirect git calls: `clone` copies the fixture, `rev-parse` echoes a SHA."""
    repo = make_repo(tmp_path)
    dest_holder = {}
    shas = {"head": "a" * 40}

    def fake_run(argv, **kwargs):
        if argv[:2] == ["git", "clone"]:
            dest = argv[-1]
            subprocess.run(["cp", "-r", str(repo), dest], check=True)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["git", "-C", str(dest_holder.get("dest", ""))] and argv[3] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, shas["head"] + "\n", "")
        raise AssertionError(f"unexpected git argv: {argv}")

    monkeypatch.setattr(harvest_repo.subprocess, "run", fake_run)
    return repo, dest_holder, shas


class TestCloneArgv:
    def test_shallow_clone(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(harvest_repo.subprocess, "run",
                            lambda argv, **kw: seen.setdefault("argv", argv)
                            or subprocess.CompletedProcess(argv, 0, "", ""))
        harvest_repo.clone_repo("o", "r", "/tmp/dest")
        assert seen["argv"][:4] == ["git", "clone", "--depth", "1"]
        assert seen["argv"][-1] == "/tmp/dest"
        assert "github.com/o/r" in seen["argv"][-2]


class TestInventory:
    def test_classifies_artifacts(self, tmp_path):
        m = harvest_repo.inventory(make_repo(tmp_path))
        assert m["notebooks"] == [{"path": "01.ipynb", "kernel": "python3",
                                   "cells": 1, "has_outputs": False}]
        assert m["scripts"] == [{"path": "backtest.py", "has_main": True}]
        assert m["specs"] == [{"path": "notes.md"}, {"path": "strategy.yaml"}]
        assert m["data"] == [{"path": "data/spy.csv", "bytes": 24}]
        assert m["deps"] == {"requirements_txt": True, "environment_yml": False,
                             "setup_py": False}
        assert m["counts"] == {"notebooks": 1, "scripts": 1, "specs": 2, "data": 1}

    def test_notebook_with_outputs(self, tmp_path):
        nb = {"metadata": {"kernelspec": {"name": "python3"}},
              "cells": [{"cell_type": "code", "outputs": [{"text": "1"}],
                         "execution_count": 1}]}
        (tmp_path / "x.ipynb").write_text(json.dumps(nb))
        m = harvest_repo.inventory(tmp_path)
        assert m["notebooks"][0]["has_outputs"] is True

    def test_ignores_git_and_venv(self, tmp_path):
        (tmp_path / ".git" / "objects").mkdir(parents=True)
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / ".venv" / "lib" / "x.py").write_text("x = 1\n")
        assert harvest_repo.inventory(tmp_path)["scripts"] == []


class TestPublish:
    def test_atomic_valid_json(self, tmp_path):
        out = tmp_path / "manifest.json"
        harvest_repo.publish({"a": 1}, out)
        assert json.loads(out.read_text()) == {"a": 1}
        assert not list(tmp_path.glob("*.tmp"))


class TestResume:
    def test_second_run_is_a_noop(self, tmp_path, cloned):
        repo, dest_holder, shas = cloned
        dest = tmp_path / "data" / "repos" / "o-r"
        args = ["o/r"]
        # first run: clones + writes manifest
        harvest_repo.main_with(args, root=tmp_path)
        first = json.loads((dest / "manifest.json").read_text())
        assert first["repo"]["sha"] == "a" * 40
        # second run at the same SHA: no second clone, manifest unchanged
        calls = []
        monkeypatch_calls = None
        before = (dest / "manifest.json").read_text()
        harvest_repo.main_with(args, root=tmp_path)
        assert (dest / "manifest.json").read_text() == before

    def test_changed_sha_reinventories(self, tmp_path, cloned):
        repo, dest_holder, shas = cloned
        dest = tmp_path / "data" / "repos" / "o-r"
        harvest_repo.main_with(["o/r"], root=tmp_path)
        shas["head"] = "b" * 40
        (dest / "backtest.py").write_text("# changed upstream\n")
        harvest_repo.main_with(["o/r"], root=tmp_path)
        after = json.loads((dest / "manifest.json").read_text())
        assert after["repo"]["sha"] == "b" * 40


class TestRepoSpec:
    @pytest.mark.parametrize("bad", ["noslash", "a/b/c", "../escape", "/abs"])
    def test_rejects_bad_spec(self, bad):
        with pytest.raises(ValueError):
            harvest_repo.parse_repo_spec(bad)

    def test_reads_repos_file(self, tmp_path):
        f = tmp_path / "repos.txt"
        f.write_text("# comment\n\nLastAncientOne/Stock_Analysis_For_Quant\nAuquan/Tutorials\n")
        assert harvest_repo.repos_from_file(f) == [
            "LastAncientOne/Stock_Analysis_For_Quant", "Auquan/Tutorials"]
```

- [ ] **Step 2: Run the suite to see the failures**

```
uv run python -m pytest tests/test_harvest_repo.py -q
```

Expected: collection error (`No module named harvest_repo`) — red.

- [ ] **Step 3: Implement `harvest_repo.py`** (create at repo root)

Module docstring documents: CLI usage, the manifest schema above, the
classification rules, resume semantics, and that `data/repos/` is already
covered by the repo's `.gitignore` (`data/` entry — no `.gitignore` change).

Required surface (names matter — the tests import them):

- `parse_repo_spec(spec: str) -> tuple[str, str]` — `owner/repo`; reject
  anything else (extra slash, empty part, `..`, leading `/`) with `ValueError`.
- `repos_from_file(path: Path) -> list[str]` — one spec per line, `#` comments
  and blank lines skipped.
- `repo_url(owner: str, name: str) -> str` — `https://github.com/<owner>/<name>.git`.
- `repo_dir(owner: str, name: str, root: Path) -> Path` — `root/data/repos/<owner>-<name>`.
- `run_git(argv: list[str]) -> str` — `subprocess.run(argv, capture_output=True, text=True, check=True)`, returns `stdout.strip()`; raises `RuntimeError` with stderr on failure.
- `clone_repo(owner, name, dest) -> None` — `run_git(["git", "clone", "--depth", "1", url, str(dest)])`; `dest.parent.mkdir(parents=True, exist_ok=True)` first.
- `rev_parse(dest: Path) -> str` — `run_git(["git", "-C", str(dest), "rev-parse", "HEAD"])`.
- `inventory(tree: Path) -> dict` — the pure walk returning the manifest body (no `repo`/`cloned_at` keys).
- `publish(manifest: dict, path: Path) -> None` — write to `path.with_suffix(path.suffix + ".tmp")` then `Path.replace`.
- `main_with(argv: list[str] | None = None, root: Path | None = None) -> int` — argparse: positional repo specs (nargs="*"), `--repos FILE`, `--root` (default: `harvest_repo.ROOT`). For each spec: parse; skip when `manifest.json` exists and its `repo.sha` equals `rev_parse(dest)`; else clone when `dest` is missing, then inventory + publish with `repo`/`cloned_at` (UTC ISO8601) filled in; print one summary line per repo.

- [ ] **Step 4: Run the suite until green**

```
uv run python -m pytest tests/test_harvest_repo.py -q
```

- [ ] **Step 5: Manual smoke against a real repo (no network assertions)**

```
uv run python harvest_repo.py LastAncientOne/Stock_Analysis_For_Quant
uv run python harvest_repo.py LastAncientOne/Stock_Analysis_For_Quant   # second run: no-op
python3 -c "import json;print(json.load(open('data/repos/LastAncientOne-Stock_Analysis_For_Quant/manifest.json'))['counts'])"
```

Confirm: clones, counts look sane, second run prints a skip and does not re-clone.

- [ ] **Step 6: Full suite + commit**

```
uv run python -m pytest -q
git add harvest_repo.py tests/test_harvest_repo.py docs/plans/2026-10-11-issue23-harvest-repo.md
git commit -m "harvest_repo: shallow-clone strategy repos + inventory manifest (#23)"
```

---

## Acceptance mapping (#23)

| #23 acceptance | Where satisfied |
|---|---|
| All three pilot repos clone + manifest without manual intervention | Step 5 smoke on `LastAncientOne/Stock_Analysis_For_Quant`; same code path for `Auquan/Tutorials`, `rotmanfinhub/gamma-vega-rl-hedging` |
| Manifest schema documented in the module docstring | `harvest_repo.py` docstring |
| Re-run is a no-op for an already-cloned pinned SHA | `TestResume.test_second_run_is_a_noop` |
