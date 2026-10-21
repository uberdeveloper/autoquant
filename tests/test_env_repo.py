"""Tests for env_repo.py — per-repo venv argv, reuse keys, legacy fallback.

`run_uv` is the module's only subprocess call site, so every test here
monkeypatches it: no test touches the network, and none of them writes
outside `tmp_path`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import env_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_repo(root: Path, owner: str = "Auquan", name: str = "Tutorials",
              sha: str = "abc123", deps: dict | None = None,
              files: dict | None = None) -> Path:
    """Build <root>/data/repos/<owner>-<name>/ with a manifest and files."""
    dest = root / "data" / "repos" / f"{owner}-{name}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo": {"owner": owner, "name": name,
                 "url": f"https://github.com/{owner}/{name}.git",
                 "sha": sha, "cloned_at": "2026-10-21T00:00:00+00:00"},
        "notebooks": [], "scripts": [], "specs": [], "data": [],
        "deps": deps or {"requirements_txt": False, "environment_yml": False,
                         "setup_py": False},
        "counts": {"notebooks": 0, "scripts": 0, "specs": 0, "data": 0},
    }
    for rel, text in (files or {}).items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (dest / "manifest.json").write_text(json.dumps(manifest))
    return dest


def stage(tmp_path, monkeypatch):
    """Redirect the module constants to tmp_path (no real repo is touched)."""
    monkeypatch.setattr(env_repo, "ROOT", tmp_path)
    monkeypatch.setattr(env_repo, "REPOS", tmp_path / "data" / "repos")
    monkeypatch.setattr(env_repo, "ENVS", tmp_path / "data" / "repo_envs.jsonl")
    return tmp_path


class FakeUv:
    """Stand-in for `run_uv`: records argv, fakes the venv it would create.

    `fail_for` holds a path fragment; any uv call mentioning it raises
    RuntimeError, which is how the per-repo containment is exercised.
    """

    def __init__(self, fail_for: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[int | None] = []
        self.fail_for = fail_for

    def __call__(self, argv, timeout=None):
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        if self.fail_for and any(self.fail_for in a for a in argv):
            raise RuntimeError("uv exploded")
        if len(argv) > 1 and argv[1] == "venv":
            venv = Path(argv[-1])
            venv.mkdir(parents=True, exist_ok=True)
            (venv / "pyvenv.cfg").write_text(
                "home = /usr/bin\nimplementation = CPython\n"
                "version_info = 3.11.2\n")
        return ""


def manifest_of(dest: Path) -> dict:
    return json.loads((dest / "manifest.json").read_text())


class TestConstants:
    def test_paths_and_names(self):
        assert env_repo.REPOS == env_repo.ROOT / "data" / "repos"
        assert env_repo.ENVS == env_repo.ROOT / "data" / "repo_envs.jsonl"
        assert env_repo.VENV_DIRNAME == ".venv"
        assert env_repo.LEGACY_VENV_DIRNAME == ".venv-legacy"
        assert env_repo.LEGACY_PYTHON == "3.9"
        assert env_repo.UV_TIMEOUT == 1800
        assert env_repo.DEFAULT_TIMEOUT == 1800

    def test_legacy_pins_are_pandas_1x_compatible(self):
        pins = env_repo.LEGACY_PINS
        assert "pandas==1.3.5" in pins
        assert "numpy==1.22.4" in pins
        assert all(isinstance(p, str) for p in pins)


class TestDepsSource:
    def test_requirements_txt_wins(self, tmp_path):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": True,
                                         "setup_py": False})
        (dest / "requirements.txt").write_text("numpy\n")
        assert env_repo.deps_source(dest, manifest_of(dest)) == "requirements.txt"

    def test_environment_yml_when_no_requirements(self, tmp_path):
        dest = make_repo(tmp_path, deps={"requirements_txt": False,
                                         "environment_yml": True,
                                         "setup_py": False})
        assert env_repo.deps_source(dest, manifest_of(dest)) == "environment.yml"

    def test_setup_py_alone_is_none(self, tmp_path):
        dest = make_repo(tmp_path, deps={"requirements_txt": False,
                                         "environment_yml": False,
                                         "setup_py": True})
        assert env_repo.deps_source(dest, manifest_of(dest)) == "none"

    def test_manifest_flags_beat_the_filesystem(self, tmp_path):
        # The manifest is the contract: a stray requirements.txt on disk that
        # harvest_repo did not record must not become an install source.
        dest = make_repo(tmp_path, deps={"requirements_txt": False,
                                         "environment_yml": False,
                                         "setup_py": False})
        (dest / "requirements.txt").write_text("numpy==1.23.1\n")
        assert env_repo.deps_source(dest, manifest_of(dest)) == "none"

    def test_missing_deps_block_is_none(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        manifest.pop("deps")
        assert env_repo.deps_source(dest, manifest) == "none"


class TestReadRequirements:
    def test_strips_comments_blanks_and_pip_flags(self, tmp_path):
        p = tmp_path / "requirements.txt"
        p.write_text(
            "# a comment\n"
            "\n"
            "numpy==1.23.1\n"
            "  pandas>=1.0   # inline comment\n"
            "-r other.txt\n"
            "--requirement more.txt\n"
            "-e .\n"
            "--editable .\n"
            "--index-url https://example.invalid/simple\n"
            "-f https://example.invalid/wheels\n"
            "scipy==1.8.1\n")
        assert env_repo.read_requirements(p) == ["numpy==1.23.1",
                                                 "pandas>=1.0",
                                                 "scipy==1.8.1"]


class TestEnvironmentYml:
    def test_conda_pins_dropped_pip_sublist_kept(self, tmp_path):
        p = tmp_path / "environment.yml"
        p.write_text(
            "name: legacy\n"
            "dependencies:\n"
            "  - python=3.9\n"
            "  - numpy=1.23\n"
            "  - pandas==1.3.5\n"
            "  - pip:\n"
            "    - scipy==1.7.3\n"
            "    - scikit-learn==1.0.2\n")
        assert env_repo.pip_requirements_from_environment_yml(p) == [
            "pandas==1.3.5", "scipy==1.7.3", "scikit-learn==1.0.2"]

    def test_dropped_conda_deps_are_counted(self, tmp_path):
        dest = make_repo(tmp_path, deps={"requirements_txt": False,
                                         "environment_yml": True,
                                         "setup_py": False})
        (dest / "environment.yml").write_text(
            "dependencies:\n  - python=3.9\n  - numpy=1.23\n  - pandas==1.3.5\n")
        state = env_repo.env_state(dest, manifest_of(dest))
        assert state["dropped_conda_deps"] == 2
        assert state["requirements"] == ["pandas==1.3.5"]


class TestEnvKey:
    def test_stable_and_16_hex(self):
        key = env_repo.env_key("sha1", "numpy==1.23.1\n")
        assert key == env_repo.env_key("sha1", "numpy==1.23.1\n")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_changes_with_sha_or_deps(self):
        base = env_repo.env_key("sha1", "numpy\n")
        assert base != env_repo.env_key("sha2", "numpy\n")
        assert base != env_repo.env_key("sha1", "pandas\n")

    def test_is_sha256_of_sha_and_deps(self):
        expected = hashlib.sha256(b"sha1\nnumpy\n").hexdigest()[:16]
        assert env_repo.env_key("sha1", "numpy\n") == expected


class TestBuildEnv:
    def test_uv_argv_order_and_flags(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        row = env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                                 root=tmp_path)
        assert len(fake.calls) == 2
        assert fake.calls[0][:6] == ["uv", "venv", "--no-project", "--clear",
                                     "--python", env_repo.DEFAULT_PYTHON]
        assert fake.calls[0][-1] == str(dest / ".venv")
        assert fake.calls[1][:3] == ["uv", "pip", "install"]
        assert fake.calls[1][3] == "--python"
        assert fake.calls[1][4] == str(dest / ".venv" / "bin" / "python")
        assert fake.calls[1][5] == "-r"
        assert row["mode"] == "repo-venv"
        assert row["status"] == "ready"
        assert row["venv"] == "data/repos/Auquan-Tutorials/.venv"
        assert row["python"] == "3.11.2"

    def test_no_venv_when_no_deps(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        row = env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                                 root=tmp_path)
        assert fake.calls == []
        assert row["mode"] == "main"
        assert row["venv"] is None
        assert row["python"] is None
        assert row["status"] == "ready"

    def test_temp_requirements_file_is_removed(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert list(dest.glob(".uv-requirements-*")) == []
        assert (dest / "requirements.txt").read_text() == "numpy==1.23.1\n"

    def test_requirements_file_is_filtered(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt":
                                "# c\nnumpy==1.23.1\n-r other.txt\n"})
        seen = {}

        def fake(argv, timeout=None):
            if argv[1] == "pip":
                seen["text"] = Path(argv[-1]).read_text()
            return ""

        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert seen["text"] == "numpy==1.23.1\n"


class TestReuse:
    def build(self, tmp_path, monkeypatch, dest, label="Auquan/Tutorials"):
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        return fake

    def test_same_key_is_reused_without_uv(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        fake = self.build(tmp_path, monkeypatch, dest)
        assert len(fake.calls) == 2
        fake.calls.clear()
        assert env_repo.main_with([], root=tmp_path) == 0
        assert fake.calls == []

    def test_changed_sha_rebuilds(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        self.build(tmp_path, monkeypatch, dest)
        manifest = manifest_of(dest)
        manifest["repo"]["sha"] = "def456"
        (dest / "manifest.json").write_text(json.dumps(manifest))
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        assert len(fake.calls) == 2

    def test_changed_deps_rebuild(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        self.build(tmp_path, monkeypatch, dest)
        (dest / "requirements.txt").write_text("numpy==1.23.2\n")
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        assert len(fake.calls) == 2

    def test_missing_venv_dir_rebuilds(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        self.build(tmp_path, monkeypatch, dest)
        import shutil
        shutil.rmtree(dest / ".venv")
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        assert len(fake.calls) == 2

    def test_rebuild_flag_forces(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        self.build(tmp_path, monkeypatch, dest)
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with(["--rebuild"], root=tmp_path) == 0
        assert len(fake.calls) == 2


class TestIsLegacyApiError:
    @pytest.mark.parametrize("text", [
        "AttributeError: 'DataFrame' object has no attribute 'append'",
        "AttributeError: 'DataFrame' object has no attribute 'ix'",
        "AttributeError: 'DataFrame' object has no attribute 'iteritems'",
        "AttributeError: 'DataFrame' object has no attribute 'Panel'",
        "AttributeError: module 'numpy' has no attribute 'float'",
        "AttributeError: module 'numpy' has no attribute 'bool'",
        "AttributeError: module 'numpy' has no attribute 'int'",
        "AttributeError: module 'numpy' has no attribute 'object'",
        "AttributeError: module 'numpy' has no attribute 'str'",
        "AttributeError: module 'numpy' has no attribute 'bool8'",
        "np.float is deprecated",
        "np.str is deprecated",
        "np.bool8 is deprecated",
    ])
    def test_matches_legacy_signatures(self, text):
        assert env_repo.is_legacy_api_error(text) is True

    @pytest.mark.parametrize("text", [
        "ModuleNotFoundError: No module named 'pandas'",
        "ModuleNotFoundError: No module named pandas",
        "ValueError: something else went wrong",
        "RuntimeError: boom",
        "",
    ])
    def test_does_not_match_other_errors(self, text):
        assert env_repo.is_legacy_api_error(text) is False


class TestPlanFallback:
    def row(self, deps_source="none"):
        return {"repo": "Auquan/Tutorials", "path": "data/repos/Auquan-Tutorials",
                "mode": "main", "deps_source": deps_source, "venv": None,
                "python": None, "sha": "abc", "deps_sha256": "d",
                "env_key": "k", "dropped_conda_deps": 0, "blocked": [],
                "status": "failed", "error": "boom", "built_at": "t"}

    def test_fires_on_legacy_error_without_declared_pins(self):
        out = env_repo.plan_fallback(
            self.row(), "AttributeError: 'DataFrame' object has no attribute 'append'")
        assert out is not None
        assert out["mode"] == "legacy-fallback"
        assert out["legacy_python"] == env_repo.LEGACY_PYTHON
        assert out["legacy_pins"] == env_repo.LEGACY_PINS
        assert "append" in out["fallback_reason"]

    def test_no_fallback_when_repo_declared_pins(self):
        assert env_repo.plan_fallback(
            self.row("requirements.txt"),
            "AttributeError: 'DataFrame' object has no attribute 'append'") is None

    def test_no_fallback_on_unrelated_error(self):
        assert env_repo.plan_fallback(self.row(), "ValueError: nope") is None

    def test_missing_module_is_not_an_api_break(self):
        assert env_repo.plan_fallback(
            self.row(), "ModuleNotFoundError: No module named 'pandas'") is None


class TestTalibRequired:
    def test_requirements_pin_is_blocked(self, tmp_path):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "TA-Lib==0.4.28\n"})
        blocked = env_repo.talib_required(dest, manifest_of(dest))
        assert len(blocked) == 1
        assert blocked[0]["path"] == "requirements.txt"
        assert "TA-Lib" in blocked[0]["reason"]
        assert "not built" in blocked[0]["reason"]

    def test_notebook_import_is_blocked(self, tmp_path):
        nb = {"cells": [{"cell_type": "code",
                         "source": ["import talib\n", "talib.MACD()"]}]}
        dest = make_repo(tmp_path, files={"nb.ipynb": json.dumps(nb)})
        manifest = manifest_of(dest)
        manifest["notebooks"] = [{"path": "nb.ipynb", "kernel": "python3",
                                   "cells": 1, "has_outputs": False}]
        blocked = env_repo.talib_required(dest, manifest)
        assert len(blocked) == 1
        assert blocked[0]["path"] == "nb.ipynb"
        assert blocked[0]["kind"] == "notebook"
        assert blocked[0]["reason"] == (
            "ta-lib needs a C library (TA-Lib) that is not installed; "
            "not built")

    def test_clean_repo_is_not_blocked(self, tmp_path):
        dest = make_repo(tmp_path, files={"run.py": "import numpy\n"})
        manifest = manifest_of(dest)
        manifest["scripts"] = [{"path": "run.py", "has_main": False}]
        assert env_repo.talib_required(dest, manifest) == []


class TestMainWith:
    def test_all_ready_exits_zero_and_writes_rows(self, tmp_path, monkeypatch,
                                                  capsys):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        assert env_repo.main_with([], root=tmp_path) == 0
        rows = env_repo.load(env_repo.ENVS)
        assert len(rows) == 1
        row = rows[0]
        assert row["repo"] == "Auquan/Tutorials"
        assert row["status"] == "ready"
        assert row["mode"] == "repo-venv"
        assert row["env_key"] == env_repo.env_key(
            "abc123", "numpy==1.23.1\n")
        assert row["deps_sha256"] == hashlib.sha256(
            b"numpy==1.23.1\n").hexdigest()
        assert row["built_at"].endswith("Z")
        assert "repo-venv=1" in capsys.readouterr().out

    def test_nothing_to_build_exits_zero(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        (tmp_path / "data" / "repos").mkdir(parents=True)
        assert env_repo.main_with([], root=tmp_path) == 0
        assert "nothing to build" in capsys.readouterr().out

    def test_malformed_spec_exits_one(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        assert env_repo.main_with(["not-a-spec"], root=tmp_path) == 1
        assert "bad spec" in capsys.readouterr().err

    def test_missing_manifest_exits_one(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        assert env_repo.main_with(["Some/Repo"], root=tmp_path) == 1
        assert "no manifest for Some/Repo" in capsys.readouterr().err

    def test_one_failure_does_not_stop_the_batch(self, tmp_path, monkeypatch,
                                                 capsys):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, owner="Good", name="One",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        make_repo(tmp_path, owner="Bad", name="Two",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "pandas==1.3.5\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv(fail_for="Bad-Two"))
        assert env_repo.main_with([], root=tmp_path) == 1
        rows = {r["repo"]: r for r in env_repo.load(env_repo.ENVS)}
        assert rows["Good/One"]["status"] == "ready"
        assert rows["Bad/Two"]["status"] == "failed"
        assert "uv exploded" in rows["Bad/Two"]["error"]
        assert "uv exploded" in capsys.readouterr().err

    def test_failed_repo_keeps_identity_fields(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, owner="Bad", name="Two",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "pandas==1.3.5\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv(fail_for="Bad-Two"))
        env_repo.main_with([], root=tmp_path)
        row = env_repo.load(env_repo.ENVS)[0]
        assert row["sha"] == "abc123"
        assert row["deps_source"] == "requirements.txt"
        assert row["env_key"] == env_repo.env_key("abc123", "pandas==1.3.5\n")
        assert row["status"] == "failed"
        assert row["error"]

    def test_repos_file_is_read(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        spec_file = tmp_path / "repos.txt"
        spec_file.write_text("# comment\nAuquan/Tutorials\n")
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        assert env_repo.main_with(["--repos", str(spec_file)],
                                   root=tmp_path) == 0
        rows = env_repo.load(env_repo.ENVS)
        assert [r["repo"] for r in rows] == ["Auquan/Tutorials"]

    def test_repos_are_not_duplicated_across_runs(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        env_repo.main_with([], root=tmp_path)
        env_repo.main_with([], root=tmp_path)
        assert len(env_repo.load(env_repo.ENVS)) == 1


class TestUvLockInvariant:
    def test_uv_lock_is_untouched_by_a_build(self, tmp_path, monkeypatch):
        # The acceptance criterion: no repo dependency ever lands in the
        # main uv.lock. The build is mocked, so this is a regression guard
        # on the mechanism (--no-project plus --python <venv>/bin/python).
        lock = PROJECT_ROOT / "uv.lock"
        before = hashlib.sha256(lock.read_bytes()).hexdigest()
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.main_with([], root=tmp_path)
        after = hashlib.sha256(lock.read_bytes()).hexdigest()
        assert after == before
        assert fake.calls[0][2] == "--no-project"
        assert fake.calls[1][4].endswith("/.venv/bin/python")

    def test_project_requirements_are_not_rewritten(self, tmp_path,
                                                    monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        env_repo.main_with([], root=tmp_path)
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "pandas==1.3.5" not in pyproject


class TestLegacyFallbackBuild:
    """T1: the legacy fallback env must use the legacy interpreter."""

    def repo(self, tmp_path):
        return make_repo(tmp_path)

    def test_legacy_mode_builds_venv_legacy_with_the_legacy_python(
            self, tmp_path, monkeypatch):
        dest = self.repo(tmp_path)
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        row = env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                                 mode="legacy-fallback", root=tmp_path)
        assert fake.calls[0][-3] == "--python"
        assert fake.calls[0][-2] == env_repo.LEGACY_PYTHON
        assert fake.calls[0][-1].endswith(".venv-legacy")
        assert fake.calls[1][4].endswith(".venv-legacy/bin/python")
        assert row["mode"] == "legacy-fallback"
        assert row["venv"].endswith(".venv-legacy")

    def test_legacy_mode_honours_an_explicit_python(self, tmp_path,
                                                    monkeypatch):
        dest = self.repo(tmp_path)
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           mode="legacy-fallback", python="3.10", root=tmp_path)
        assert fake.calls[0][-2] == "3.10"


class TestPlanFallbackRetryOnce:
    """T12: the fallback fires exactly once."""

    def row(self):
        return {"repo": "Auquan/Tutorials", "path": "data/repos/Auquan-Tutorials",
                "mode": "repo-venv", "deps_source": "none", "venv": None,
                "python": None, "sha": "abc", "deps_sha256": "d",
                "env_key": "k", "dropped_conda_deps": 0, "blocked": [],
                "status": "failed", "error": "boom", "built_at": "t"}

    def test_no_fallback_for_an_already_legacy_row(self):
        row = self.row()
        row["mode"] = "legacy-fallback"
        assert env_repo.plan_fallback(
            row, "AttributeError: 'DataFrame' object has no attribute 'append'"
        ) is None

    def test_no_fallback_when_a_fallback_reason_is_recorded(self):
        row = self.row()
        row["fallback_reason"] = "AttributeError: no attribute 'append'"
        assert env_repo.plan_fallback(
            row, "AttributeError: 'DataFrame' object has no attribute 'append'"
        ) is None

    def test_fallback_reason_is_truncated(self):
        error = ("AttributeError: 'DataFrame' object has no attribute 'append'"
                 + "x" * 400)
        out = env_repo.plan_fallback(self.row(), error)
        assert out is not None
        assert len(out["fallback_reason"]) == 300

    def test_python_override_is_honoured(self):
        out = env_repo.plan_fallback(
            self.row(), "AttributeError: 'DataFrame' object has no attribute 'append'",
            python="3.10")
        assert out is not None
        assert out["legacy_python"] == "3.10"


class TestTalibRegex:
    """T8: the ta-lib detector must not fire on ordinary identifiers."""

    @pytest.mark.parametrize("text", [
        "from data_lib import loader",
        "from datalib import x",
        "import beta_lib",
        "delta-lib==1.0",
    ])
    def test_benign_identifiers_are_not_ta_lib(self, tmp_path, text):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": text + "\n"})
        assert env_repo.talib_required(dest, manifest_of(dest)) == []

    @pytest.mark.parametrize("text", [
        "import talib",
        "TA-Lib==0.4.28",
        "import talib as tl",
    ])
    def test_real_ta_lib_references_still_match(self, tmp_path, text):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": text + "\n"})
        blocked = env_repo.talib_required(dest, manifest_of(dest))
        assert len(blocked) == 1

    def test_script_artifact_using_talib_is_blocked(self, tmp_path):
        dest = make_repo(tmp_path, files={"run.py": "import talib\n"})
        manifest = manifest_of(dest)
        manifest["scripts"] = [{"path": "run.py"}]
        blocked = env_repo.talib_required(dest, manifest)
        assert len(blocked) == 1
        assert blocked[0]["kind"] == "script"
        assert blocked[0]["path"] == "run.py"


class TestEnvironmentYmlParsing:
    """T18 / P6: conda environment.yml handling."""

    def test_environment_python_reads_the_conda_pin(self):
        assert env_repo.environment_python(
            "dependencies:\n  - python=3.9\n") == "3.9"
        assert env_repo.environment_python(
            "dependencies:\n  - python==3.10\n") == "3.10"
        assert env_repo.environment_python(
            "dependencies:\n  - numpy=1.23\n") is None
        assert env_repo.environment_python("dependencies:\n") is None

    def test_bare_names_are_pip_requirements(self, tmp_path):
        p = tmp_path / "environment.yml"
        p.write_text("dependencies:\n  - pandas\n  - pytz\n")
        assert env_repo.pip_requirements_from_environment_yml(p) == [
            "pandas", "pytz"]

    @pytest.mark.parametrize("spec", ["python", "pip", "setuptools",
                                      "python>=3.9", "pip>=20",
                                      "setuptools>=40"])
    def test_interpreter_and_tooling_names_are_excluded(self, tmp_path, spec):
        p = tmp_path / "environment.yml"
        p.write_text(f"dependencies:\n  - {spec}\n  - pandas\n")
        assert env_repo.pip_requirements_from_environment_yml(p) == ["pandas"]

    def test_conda_single_equals_is_still_dropped(self, tmp_path):
        p = tmp_path / "environment.yml"
        p.write_text("dependencies:\n  - numpy=1.23\n  - numpy>=1.23\n")
        reqs = env_repo.pip_requirements_from_environment_yml(p)
        assert reqs == ["numpy>=1.23"]

    def test_build_env_uses_the_conda_python_pin(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": False,
                                         "environment_yml": True,
                                         "setup_py": False},
                         files={"environment.yml":
                                "dependencies:\n  - python=3.9\n  - pandas\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert fake.calls[0][-3] == "--python"
        assert fake.calls[0][-2] == "3.9"


class TestRunUvErrors:
    """T2: the subprocess failure contract of the only uv call site."""

    def test_called_process_error_reports_stderr(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise subprocess.CalledProcessError(1, argv, output="",
                                                stderr="boom stderr")
        monkeypatch.setattr(env_repo.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError) as exc:
            env_repo.run_uv(["uv", "venv"])
        assert "boom stderr" in str(exc.value)

    def test_timeout_is_reported(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 5)
        monkeypatch.setattr(env_repo.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError) as exc:
            env_repo.run_uv(["uv", "venv"], timeout=5)
        assert "uv timed out after 5s" in str(exc.value)

    def test_missing_uv_is_reported(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise FileNotFoundError(2, "No such file or directory")
        monkeypatch.setattr(env_repo.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError) as exc:
            env_repo.run_uv(["uv", "venv"])
        assert "uv not found on PATH" in str(exc.value)


class TestMainWithEdgeCases:
    """T3 / T4 / T5 / T9 / T10 / T11 / T19."""

    def test_corrupt_manifest_becomes_a_failed_row(self, tmp_path,
                                                   monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, owner="Good", name="One",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        bad = make_repo(tmp_path, owner="Bad", name="Two")
        (bad / "manifest.json").write_text("{not json")
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        assert env_repo.main_with([], root=tmp_path) == 1
        rows = {r["repo"]: r for r in env_repo.load(env_repo.ENVS)}
        assert rows["Good/One"]["status"] == "ready"
        assert rows["Bad-Two"]["status"] == "failed"
        assert "corrupt manifest" in rows["Bad-Two"]["error"]

    def test_stale_row_of_a_fixed_manifest_is_dropped(self, tmp_path,
                                                       monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, owner="Bad", name="Two")
        (dest / "manifest.json").write_text("{not json")
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        env_repo.main_with([], root=tmp_path)
        assert len(env_repo.load(env_repo.ENVS)) == 1
        # the manifest is repaired: the fresh row must replace the stale one
        make_repo(tmp_path, owner="Bad", name="Two")
        env_repo.main_with([], root=tmp_path)
        rows = env_repo.load(env_repo.ENVS)
        assert len(rows) == 1
        assert rows[0]["repo"] == "Bad/Two"

    def test_declared_but_missing_dep_file_is_a_failed_row(self, tmp_path,
                                                           monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False})
        # the manifest declares requirements.txt, but the file is not there
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 1
        row = env_repo.load(env_repo.ENVS)[0]
        assert row["status"] == "failed"
        assert "requirements.txt declared by manifest but missing" in row["error"]
        assert fake.calls == []

    def test_missing_repos_file_exits_one(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        assert env_repo.main_with(["--repos", str(tmp_path / "nope.txt")],
                                   root=tmp_path) == 1
        assert "error: cannot read" in capsys.readouterr().err

    def test_deps_source_flip_rebuilds(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        # byte-identical dep text, different declared source
        manifest = manifest_of(dest)
        manifest["deps"] = {"requirements_txt": False, "environment_yml": True,
                            "setup_py": False}
        (dest / "manifest.json").write_text(json.dumps(manifest))
        (dest / "environment.yml").write_text("numpy==1.23.1\n")
        fake.calls.clear()
        assert env_repo.main_with([], root=tmp_path) == 0
        assert len(fake.calls) == 2

    def test_failed_repo_is_retried_on_the_next_run(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv(fail_for="Auquan"))
        assert env_repo.main_with([], root=tmp_path) == 1
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        assert len(fake.calls) == 2

    def test_timeout_is_forwarded_to_every_uv_call(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.main_with([], root=tmp_path)
        assert fake.timeouts == [env_repo.UV_TIMEOUT, env_repo.UV_TIMEOUT]

    def test_timeout_flag_reaches_every_uv_call(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.main_with(["--timeout", "5"], root=tmp_path)
        assert fake.timeouts == [5, 5]

    def test_temp_requirements_file_lives_in_the_repo_dir(self, tmp_path,
                                                           monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        seen: dict = {}

        def fake(argv, timeout=None):
            if argv[1] == "pip":
                seen["parent"] = Path(argv[-1]).parent
            return ""

        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert seen["parent"] == dest

    def test_blocked_is_recorded_on_the_row(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        dest = make_repo(tmp_path, files={"run.py": "import talib\n"})
        manifest = manifest_of(dest)
        manifest["scripts"] = [{"path": "run.py"}]
        (dest / "manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        assert env_repo.main_with([], root=tmp_path) == 0
        row = env_repo.load(env_repo.ENVS)[0]
        assert len(row["blocked"]) == 1
        assert row["blocked"][0]["kind"] == "script"

    def test_environment_yml_repo_through_main_with(self, tmp_path,
                                                     monkeypatch):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, deps={"requirements_txt": False,
                                  "environment_yml": True, "setup_py": False},
                  files={"environment.yml":
                         "name: legacy\n"
                         "dependencies:\n"
                         "  - python=3.9\n"
                         "  - numpy=1.23\n"
                         "  - pandas\n"
                         "  - pytz\n"
                         "  - pip:\n"
                         "    - scipy==1.7.3\n"})
        seen: dict = {}

        def fake(argv, timeout=None):
            if argv[1] == "pip":
                seen["text"] = Path(argv[-1]).read_text()
            seen.setdefault("calls", []).append(list(argv))
            return ""

        monkeypatch.setattr(env_repo, "run_uv", fake)
        assert env_repo.main_with([], root=tmp_path) == 0
        row = env_repo.load(env_repo.ENVS)[0]
        assert row["mode"] == "repo-venv"
        assert row["deps_source"] == "environment.yml"
        assert row["dropped_conda_deps"] >= 1
        lines = seen["text"].splitlines()
        assert "scipy==1.7.3" in lines
        assert "pandas" in lines
        assert "pytz" in lines
        assert not any(line.startswith("python") for line in lines)
        assert not any(line.startswith("pip") for line in lines)
        assert seen["calls"][0][-3] == "--python"
        assert seen["calls"][0][-2] == "3.9"

    def test_alternate_root_does_not_touch_the_real_envs(self, tmp_path,
                                                          monkeypatch):
        real = env_repo.ENVS
        before = real.read_text() if real.exists() else None
        make_repo(tmp_path, deps={"requirements_txt": True,
                                  "environment_yml": False, "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        assert env_repo.main_with([], root=tmp_path) == 0
        after = real.read_text() if real.exists() else None
        assert after == before
        assert len(env_repo.load(tmp_path / "data" / "repo_envs.jsonl")) == 1

    def test_two_failures_are_counted_once_each(self, tmp_path, monkeypatch,
                                                capsys):
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, owner="Bad", name="One",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        make_repo(tmp_path, owner="Bad", name="Two",
                  deps={"requirements_txt": True, "environment_yml": False,
                        "setup_py": False},
                  files={"requirements.txt": "numpy==1.23.1\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv(fail_for="Bad"))
        assert env_repo.main_with([], root=tmp_path) == 1
        assert "failed=2" in capsys.readouterr().out


class TestRowExtras:
    """T16 / T17 / P10 / P12."""

    def test_dropped_pip_flags_are_recorded(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt":
                                "-r other.txt\nnumpy==1.23.1\n"
                                "--index-url https://example.invalid/simple\n"})
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        row = env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                                 root=tmp_path)
        assert row["dropped_pip_flags"] == 2

    def test_failed_row_can_be_a_legacy_fallback(self):
        row = env_repo._failed_row("A/B", Path("/tmp/x"), None, "boom",
                                   Path("/tmp"), mode="legacy-fallback")
        assert row["mode"] == "legacy-fallback"
        assert row["status"] == "failed"

    def test_failed_row_defaults_to_repo_venv(self):
        row = env_repo._failed_row("A/B", Path("/tmp/x"), None, "boom",
                                   Path("/tmp"))
        assert row["mode"] == "repo-venv"

    def test_stale_temp_requirement_files_are_swept(self, tmp_path,
                                                     monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        stale = dest / ".uv-requirements-abc123.txt"
        stale.write_text("pandas==1.3.5\n")
        monkeypatch.setattr(env_repo, "run_uv", FakeUv())
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert not stale.exists()


class TestUvArgvFlags:
    """T14: `uv venv` must clear the target directory."""

    def test_venv_argv_carries_clear(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path, deps={"requirements_txt": True,
                                         "environment_yml": False,
                                         "setup_py": False},
                         files={"requirements.txt": "numpy==1.23.1\n"})
        fake = FakeUv()
        monkeypatch.setattr(env_repo, "run_uv", fake)
        env_repo.build_env(dest, manifest_of(dest), "Auquan/Tutorials",
                           root=tmp_path)
        assert "--clear" in fake.calls[0]
        assert fake.calls[0][:3] == ["uv", "venv", "--no-project"]
