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
    """Redirect git calls: `clone` copies the fixture, `rev-parse` echoes a SHA.

    Returns (repo, dest_holder, shas, git) where git holds a `clones` list of
    clone destinations (count invocations) and a `flags` dict; setting
    `flags["broken"]` makes rev-parse raise RuntimeError until a fresh clone
    repairs the repo.
    """
    repo = make_repo(tmp_path)
    dest_holder = {}
    shas = {"head": "a" * 40}
    clones: list[str] = []
    flags = {"broken": False}
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv[:2] == ["git", "clone"]:
            dest = argv[-1]
            dest_holder["dest"] = dest
            clones.append(dest)
            flags["broken"] = False  # a fresh clone repairs the checkout
            real_run(["cp", "-r", str(repo), dest], check=True)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["git", "-C"] and argv[3] == "rev-parse":
            assert argv[2] == dest_holder.get("dest"), (
                f"rev-parse against unexpected dir: {argv[2]!r}")
            if flags["broken"]:
                raise RuntimeError("fatal: not a git repository")
            return subprocess.CompletedProcess(argv, 0, shas["head"] + "\n", "")
        raise AssertionError(f"unexpected git argv: {argv}")

    monkeypatch.setattr(harvest_repo.subprocess, "run", fake_run)
    return repo, dest_holder, shas, {"clones": clones, "flags": flags}


class TestCloneArgv:
    def test_shallow_clone(self, monkeypatch):
        seen = {}

        def fake(argv, **kw):
            seen["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(harvest_repo.subprocess, "run", fake)
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
        assert m["data"] == [{"path": "data/spy.csv", "bytes": 26}]
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

    def test_publish_leaves_no_tmp_on_failure(self, tmp_path, monkeypatch):
        # A mid-write failure (e.g. disk full) must not leave a stray .tmp.
        out = tmp_path / "manifest.json"
        real_write_text = harvest_repo.Path.write_text

        def failing_write_text(self, data, *args, **kwargs):
            real_write_text(self, data, *args, **kwargs)
            raise OSError("disk full")

        monkeypatch.setattr(harvest_repo.Path, "write_text", failing_write_text)
        with pytest.raises(OSError, match="disk full"):
            harvest_repo.publish({"a": 1}, out)
        monkeypatch.undo()
        assert not list(tmp_path.glob("*.tmp"))


class TestResume:
    def test_second_run_is_a_noop(self, tmp_path, cloned, capsys):
        repo, dest_holder, shas, git = cloned
        dest = tmp_path / "data" / "repos" / "o-r"
        args = ["o/r"]
        # first run: clones + writes manifest
        harvest_repo.main_with(args, root=tmp_path)
        first = json.loads((dest / "manifest.json").read_text())
        assert first["repo"]["sha"] == "a" * 40
        assert len(git["clones"]) == 1
        capsys.readouterr()
        # second run at the same SHA: no second clone, manifest unchanged
        before = (dest / "manifest.json").read_text()
        harvest_repo.main_with(args, root=tmp_path)
        assert (dest / "manifest.json").read_text() == before
        assert len(git["clones"]) == 1
        assert "skip o/r" in capsys.readouterr().out

    def test_changed_sha_reinventories(self, tmp_path, cloned):
        repo, dest_holder, shas, git = cloned
        dest = tmp_path / "data" / "repos" / "o-r"
        harvest_repo.main_with(["o/r"], root=tmp_path)
        shas["head"] = "b" * 40
        (dest / "backtest.py").write_text("# changed upstream\n")
        harvest_repo.main_with(["o/r"], root=tmp_path)
        after = json.loads((dest / "manifest.json").read_text())
        assert after["repo"]["sha"] == "b" * 40

    def test_name_collision_reclones(self, tmp_path, cloned):
        # `a/b-c` and `a-b/c` both map to data/repos/a-b-c: the second spec
        # must re-clone, not be skipped as an already-harvested repo.
        repo, dest_holder, shas, git = cloned
        dest = tmp_path / "data" / "repos" / "a-b-c"
        harvest_repo.main_with(["a/b-c", "a-b/c"], root=tmp_path)
        manifest = json.loads((dest / "manifest.json").read_text())
        assert manifest["repo"]["owner"] == "a-b"
        assert manifest["repo"]["name"] == "c"
        assert len(git["clones"]) == 2

    def test_broken_clone_is_recloned(self, tmp_path, cloned):
        # A git failure on an existing clone repairs via re-clone + republish,
        # instead of surfacing a RuntimeError or re-inventorying the stale tree.
        repo, dest_holder, shas, git = cloned
        dest = tmp_path / "data" / "repos" / "o-r"
        harvest_repo.main_with(["o/r"], root=tmp_path)
        assert len(git["clones"]) == 1
        manifest_path = dest / "manifest.json"
        tampered = json.loads(manifest_path.read_text())
        tampered["repo"]["marker"] = "stale"  # a skip would leave this behind
        manifest_path.write_text(json.dumps(tampered))
        git["flags"]["broken"] = True
        harvest_repo.main_with(["o/r"], root=tmp_path)
        after = json.loads(manifest_path.read_text())
        assert len(git["clones"]) == 2
        assert "marker" not in after["repo"]  # republished, not skipped
        assert after["repo"]["sha"] == "a" * 40


class TestRepoSpec:
    @pytest.mark.parametrize("bad", ["noslash", "a/b/c", "../escape", "/abs"])
    def test_rejects_bad_spec(self, bad):
        with pytest.raises(ValueError):
            harvest_repo.parse_repo_spec(bad)

    def test_error_containment_continues(self, tmp_path, cloned, capsys):
        # One bad spec fails with a one-line stderr error (exit 1) while the
        # remaining repos still get harvested.
        repo, dest_holder, shas, git = cloned
        rc = harvest_repo.main_with(["noslash", "o/r"], root=tmp_path)
        assert rc == 1
        assert (tmp_path / "data" / "repos" / "o-r" / "manifest.json").exists()
        err = capsys.readouterr().err
        assert "error noslash:" in err

    def test_duplicate_specs_run_once(self, tmp_path, cloned, capsys):
        # Exact duplicate specs are deduped: no second (noisy) skip pass.
        repo, dest_holder, shas, git = cloned
        rc = harvest_repo.main_with(["o/r", "o/r"], root=tmp_path)
        assert rc == 0
        out = capsys.readouterr().out
        assert "harvested o/r" in out
        assert "skip o/r" not in out
        assert len(git["clones"]) == 1

    def test_reads_repos_file(self, tmp_path):
        f = tmp_path / "repos.txt"
        f.write_text("# comment\n\nLastAncientOne/Stock_Analysis_For_Quant\nAuquan/Tutorials\n")
        assert harvest_repo.repos_from_file(f) == [
            "LastAncientOne/Stock_Analysis_For_Quant", "Auquan/Tutorials"]
