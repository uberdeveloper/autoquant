"""Tests for triage_repo.py — repo artifact enumeration, prompts, and scoring."""
from __future__ import annotations

import json
import re

import pytest

import llm
import triage_repo


# Captured before any monkeypatch: the real breaker for abort tests.
REAL_CIRCUIT_BREAK = llm.circuit_break

ROW = {"repo": "Auquan/Tutorials", "kind": "notebook", "path": "nb.ipynb",
       "meta": {"kernel": "python3", "cells": 4, "has_outputs": True}}


FULL_REPLY = {
    "verdict": "RUN_WITH_OUR_DATA",
    "symbols": ["SPY"],
    "data_loading": ["yf.download"],
    "data_mode": "network",
    "dep_weight": "ordinary",
    "heavy_deps": [],
    "runtime_class": "seconds",
    "confidence": 0.9,
    "reason": "loads yf prices",
}


def nb_json() -> dict:
    """A notebook with list-sourced and string-sourced cells plus raw noise."""
    return {
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": [
            {"cell_type": "code", "source": ["a = 1\n", "b = 2\n"],
             "outputs": [], "execution_count": None},
            {"cell_type": "markdown", "source": "# hi"},
            {"cell_type": "raw", "source": "raw stuff"},
            {"cell_type": "code", "source": ["plot()"],
             "outputs": [{"output_type": "stream", "text": "x"}],
             "execution_count": 1},
        ],
    }


def make_repo(root, owner="Auquan", name="Tutorials"):
    """Build <root>/data/repos/<owner>-<name>/ with manifest + real artifacts."""
    dest = root / "data" / "repos" / f"{owner}-{name}"
    dest.mkdir(parents=True)
    manifest = {
        "repo": {"owner": owner, "name": name,
                 "url": f"https://github.com/{owner}/{name}.git",
                 "sha": "abc123", "cloned_at": "2026-10-21T00:00:00+00:00"},
        "notebooks": [{"path": "nb.ipynb", "kernel": "python3",
                       "cells": 4, "has_outputs": True}],
        "scripts": [{"path": "run.py", "has_main": True}],
        "specs": [{"path": "strategy.yaml"}],
        "data": [{"path": "prices.csv", "bytes": 8}],
        "deps": {"requirements_txt": False, "environment_yml": False,
                 "setup_py": False},
        "counts": {"notebooks": 1, "scripts": 1, "specs": 1, "data": 1},
    }
    (dest / "nb.ipynb").write_text(json.dumps(nb_json()))
    (dest / "run.py").write_text(
        "def go():\n    return 1\n\nif __name__ == '__main__':\n    go()\n")
    (dest / "strategy.yaml").write_text("strategy:\n  name: x\n")
    (dest / "prices.csv").write_text("a,b\n1,2\n")
    (dest / "manifest.json").write_text(json.dumps(manifest))
    return dest


def stage(tmp_path, monkeypatch):
    """Redirect module constants to tmp_path and stub the LLM CLI (offline)."""
    dest = make_repo(tmp_path)
    monkeypatch.setattr(triage_repo, "ROOT", tmp_path)
    monkeypatch.setattr(triage_repo, "TRIAGE", tmp_path / "data" / "repo_triage.jsonl")
    prompt = tmp_path / "triage_repo_prompt.md"
    prompt.write_text("RUBRIC")
    monkeypatch.setattr(triage_repo, "PROMPT", prompt)
    monkeypatch.setattr(triage_repo.llm, "preflight", lambda: None)
    monkeypatch.setattr(triage_repo.llm, "circuit_break",
                        lambda results, k=3: None)
    return dest


def manifest_of(dest):
    return json.loads((dest / "manifest.json").read_text())


class TestConstants:
    def test_default_paths(self):
        assert triage_repo.REPOS == triage_repo.ROOT / "data" / "repos"
        assert triage_repo.TRIAGE == triage_repo.ROOT / "data" / "repo_triage.jsonl"
        assert triage_repo.PROMPT == triage_repo.ROOT / "triage_repo_prompt.md"
        assert triage_repo.MAX_ARTIFACT_CHARS == 40_000
        assert triage_repo.MAX_DEPS_CHARS == 4_000
        assert triage_repo.CLI_TIMEOUT == 300


class TestNotebookSource:
    def test_joins_code_and_markdown_and_drops_raw(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text(json.dumps(nb_json()))
        text = triage_repo.notebook_source(p)
        # sources are joined verbatim with "\n\n" (the code cell's own
        # trailing newline is part of its source)
        assert text == "a = 1\nb = 2\n\n\n# hi\n\nplot()"
        assert "raw stuff" not in text

    def test_string_and_list_sources_both_work(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        obj = {"cells": [
            {"cell_type": "code", "source": "x = 1"},
            {"cell_type": "markdown", "source": ["line one\n", "line two"]},
        ]}
        p.write_text(json.dumps(obj))
        assert triage_repo.notebook_source(p) == "x = 1\n\nline one\nline two"

    def test_corrupt_notebook_raises_json_error(self, tmp_path):
        p = tmp_path / "nb.ipynb"
        p.write_text("{not json")
        with pytest.raises(json.JSONDecodeError):
            triage_repo.notebook_source(p)


class TestArtifactRows:
    def test_maps_kinds_meta_and_drops_data(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        rows = triage_repo.artifact_rows(manifest, "Auquan/Tutorials")
        assert [(r["kind"], r["path"]) for r in rows] == [
            ("notebook", "nb.ipynb"), ("script", "run.py"),
            ("spec", "strategy.yaml")]
        assert rows[0]["meta"] == {"kernel": "python3", "cells": 4,
                                   "has_outputs": True}
        assert rows[1]["meta"] == {"has_main": True}
        assert rows[2]["meta"] == {}
        assert all(r["repo"] == "Auquan/Tutorials" for r in rows)
        assert not any(r["kind"] == "data" for r in rows)


class TestRepoManifests:
    def test_enumerates_sorted_manifest_dirs(self, tmp_path):
        make_repo(tmp_path, owner="b", name="two")
        make_repo(tmp_path, owner="a", name="one")
        (tmp_path / "data" / "repos" / "no-manifest").mkdir()
        out = triage_repo.repo_manifests(tmp_path, [])
        assert [label for _dest, _m, label in out] == ["a/one", "b/two"]

    def test_label_comes_from_manifest_not_dirname(self, tmp_path):
        dest = make_repo(tmp_path, owner="a", name="b-c")
        manifest = manifest_of(dest)
        manifest["repo"].update({"owner": "real", "name": "owner"})
        (dest / "manifest.json").write_text(json.dumps(manifest))
        out = triage_repo.repo_manifests(tmp_path, [])
        assert [(str(d), label) for d, _m, label in out] == \
            [(str(dest), "real/owner")]

    def test_specs_deduped_and_parsed(self, tmp_path):
        make_repo(tmp_path, owner="a", name="b-c")
        out = triage_repo.repo_manifests(tmp_path, ["a/b-c", "a/b-c"])
        assert len(out) == 1
        assert out[0][2] == "a/b-c"

    def test_bad_spec_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="bad spec"):
            triage_repo.repo_manifests(tmp_path, ["not-a-spec"])

    def test_missing_manifest_raises_valueerror(self, tmp_path):
        with pytest.raises(ValueError, match="no manifest for Some/Repo"):
            triage_repo.repo_manifests(tmp_path, ["Some/Repo"])

    def test_corrupt_manifest_raises_json_error(self, tmp_path):
        dest = make_repo(tmp_path)
        (dest / "manifest.json").write_text("{broken")
        with pytest.raises(json.JSONDecodeError):
            triage_repo.repo_manifests(tmp_path, [])

    def test_manifest_without_repo_record_is_skipped(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        manifest.pop("repo")
        (dest / "manifest.json").write_text(json.dumps(manifest))
        assert triage_repo.repo_manifests(tmp_path, []) == []


class TestArtifactText:
    def test_dispatches_on_kind(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        rows = {r["kind"]: r for r in
                triage_repo.artifact_rows(manifest, "o/n")}
        assert "a = 1" in triage_repo.artifact_text(dest, rows["notebook"])
        assert "if __name__" in triage_repo.artifact_text(dest, rows["script"])
        assert "strategy:" in triage_repo.artifact_text(dest, rows["spec"])

    def test_missing_file_raises_oserror(self, tmp_path):
        dest = make_repo(tmp_path)
        row = {"repo": "o/n", "kind": "script", "path": "gone.py", "meta": {}}
        with pytest.raises(OSError):
            triage_repo.artifact_text(dest, row)


class TestRepoDeps:
    def test_reads_claimed_dep_files(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        manifest["deps"]["requirements_txt"] = True
        (dest / "requirements.txt").write_text("numpy==1.23.1\n")
        deps = triage_repo.repo_deps(dest, manifest)
        assert deps["requirements_txt"] == "numpy==1.23.1\n"
        assert deps["environment_yml"] == ""

    def test_unclaimed_deps_stay_empty(self, tmp_path):
        dest = make_repo(tmp_path)
        deps = triage_repo.repo_deps(dest, manifest_of(dest))
        assert deps == {"requirements_txt": "", "environment_yml": ""}

    def test_manifest_flag_is_contract_even_when_file_on_disk(self, tmp_path):
        # harvest_repo.inventory() decides presence; the filesystem can
        # disagree, and the manifest wins either way.
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        assert manifest["deps"]["requirements_txt"] is False
        (dest / "requirements.txt").write_text("numpy==1.23.1\n")
        (dest / "environment.yml").write_text("name: x\n")
        deps = triage_repo.repo_deps(dest, manifest)
        assert deps == {"requirements_txt": "", "environment_yml": ""}

    def test_long_dep_text_truncated(self, tmp_path):
        dest = make_repo(tmp_path)
        manifest = manifest_of(dest)
        manifest["deps"]["requirements_txt"] = True
        (dest / "requirements.txt").write_text("x" * (triage_repo.MAX_DEPS_CHARS + 99))
        deps = triage_repo.repo_deps(dest, manifest)
        assert deps["requirements_txt"].endswith("[...truncated for length...]")
        assert len(deps["requirements_txt"]) < triage_repo.MAX_DEPS_CHARS + 60


class TestParseJson:
    def test_bare_json(self):
        assert triage_repo.parse_json('{"verdict": "SKIP"}') == {"verdict": "SKIP"}

    def test_fenced_json(self):
        text = "```json\n{\"verdict\": \"SKIP\"}\n```"
        assert triage_repo.parse_json(text) == {"verdict": "SKIP"}

    def test_prose_wrapped_object(self):
        text = 'Here you go: {"verdict": "SKIP", "reason": "r"} -- done'
        assert triage_repo.parse_json(text) == {"verdict": "SKIP", "reason": "r"}

    def test_no_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="no JSON object"):
            triage_repo.parse_json("there is nothing here")

    def test_array_reply_is_rejected(self):
        with pytest.raises(ValueError, match="object"):
            triage_repo.parse_json('[{"verdict": "SKIP"}]')


class TestBuildPrompt:
    def test_starts_with_rubric_and_carries_payload(self):
        deps = {"requirements_txt": "numpy==1.23.1", "environment_yml": ""}
        prompt = triage_repo.build_prompt("RUBRIC", dict(ROW), "cell code", deps)
        assert prompt.startswith("RUBRIC\n\n---\n")
        assert '"repo": "Auquan/Tutorials"' in prompt
        assert '"kind": "notebook"' in prompt
        assert '"path": "nb.ipynb"' in prompt
        assert '"kernel": "python3"' in prompt
        assert '"repo_deps"' in prompt
        assert "numpy==1.23.1" in prompt
        assert '"text": "cell code"' in prompt
        assert "no markdown fences" in prompt

    def test_long_text_truncated_with_marker(self):
        long = "x" * (triage_repo.MAX_ARTIFACT_CHARS + 5_000)
        prompt = triage_repo.build_prompt("R", dict(ROW), long,
                                          {"requirements_txt": "",
                                           "environment_yml": ""})
        assert "[...truncated for length...]" in prompt
        assert len(prompt) < (triage_repo.MAX_ARTIFACT_CHARS
                              + triage_repo.MAX_DEPS_CHARS + 500)


class TestValidate:
    def test_accepts_full_reply(self):
        out = triage_repo.validate(dict(FULL_REPLY))
        assert out == FULL_REPLY
        assert out["confidence"] == 0.9

    def test_none_lists_become_empty_and_blanks_dropped(self):
        out = triage_repo.validate(dict(FULL_REPLY, symbols=None,
                                        heavy_deps=["", "torch"]))
        assert out["symbols"] == []
        assert out["heavy_deps"] == ["torch"]

    def test_rejects_unknown_verdict(self):
        with pytest.raises(ValueError, match="^invalid verdict"):
            triage_repo.validate(dict(FULL_REPLY, verdict="MAYBE"))

    def test_rejects_missing_verdict(self):
        reply = dict(FULL_REPLY)
        reply.pop("verdict")
        with pytest.raises(ValueError, match="^invalid verdict"):
            triage_repo.validate(reply)

    @pytest.mark.parametrize("field", ["data_mode", "dep_weight", "runtime_class"])
    def test_rejects_bad_enum_fields(self, field):
        with pytest.raises(ValueError, match=f"^invalid {field}"):
            triage_repo.validate(dict(FULL_REPLY, **{field: "wild"}))

    @pytest.mark.parametrize("bad", [True, "0.9", 2.0, None])
    def test_rejects_bad_confidence(self, bad):
        with pytest.raises(ValueError, match="^invalid confidence"):
            triage_repo.validate(dict(FULL_REPLY, confidence=bad))

    def test_rejects_symbols_as_string(self):
        with pytest.raises(ValueError, match="^invalid symbols"):
            triage_repo.validate(dict(FULL_REPLY, symbols="SPY"))

    def test_rejects_non_string_list_entries(self):
        with pytest.raises(ValueError, match="^invalid data_loading"):
            triage_repo.validate(dict(FULL_REPLY, data_loading=[1]))

    @pytest.mark.parametrize("reason", ["", "   ", None])
    def test_rejects_empty_reason(self, reason):
        with pytest.raises(ValueError, match="^invalid reason"):
            triage_repo.validate(dict(FULL_REPLY, reason=reason))


class TestScoreOne:
    def row(self, dest, kind="notebook"):
        manifest = manifest_of(dest)
        rows = triage_repo.artifact_rows(manifest, "Auquan/Tutorials")
        return next(r for r in rows if r["kind"] == kind)

    def test_success_returns_normalized_row(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)
        row = self.row(dest)
        seen = {}

        def fake_complete(prompt, *, model=None, extra=None, timeout=300):
            seen.update(model=model, extra=list(extra or []), timeout=timeout)
            return json.dumps(FULL_REPLY)

        monkeypatch.setattr(triage_repo.llm, "complete", fake_complete)
        out = triage_repo.score_one(dest, row, "RUBRIC", "sonnet", ["--x"])
        assert out["repo"] == "Auquan/Tutorials"
        assert out["kind"] == "notebook"
        assert out["path"] == "nb.ipynb"
        assert out["verdict"] == "RUN_WITH_OUR_DATA"
        assert out["confidence"] == 0.9
        assert out["model"] == "sonnet"
        assert out["reason"] == "loads yf prices"
        assert out["scored_at"].endswith("Z")
        assert "error" not in out
        assert seen == {"model": "sonnet", "extra": ["--x"],
                        "timeout": triage_repo.CLI_TIMEOUT}

    def test_model_defaults_to_default_label(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)
        monkeypatch.setattr(triage_repo.llm, "complete",
                            lambda prompt, **k: json.dumps(FULL_REPLY))
        out = triage_repo.score_one(dest, self.row(dest), "RUBRIC", None)
        assert out["model"] == "default"

    def test_unparseable_reply_is_stage_error(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)
        monkeypatch.setattr(triage_repo.llm, "complete",
                            lambda prompt, **k: "not json at all")
        out = triage_repo.score_one(dest, self.row(dest), "RUBRIC", None)
        assert out["error"].startswith("unparseable reply:")
        assert set(out) == {"repo", "kind", "path", "meta", "error"}
        assert llm.is_cli_error(out["error"]) is False

    def test_invalid_reply_is_stage_error(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)
        bad = dict(FULL_REPLY, verdict="MAYBE")
        monkeypatch.setattr(triage_repo.llm, "complete",
                            lambda prompt, **k: json.dumps(bad))
        out = triage_repo.score_one(dest, self.row(dest), "RUBRIC", None)
        assert out["error"].startswith("invalid verdict:")
        assert llm.is_cli_error(out["error"]) is False

    def test_llm_error_is_verbatim_cli_error(self, tmp_path, monkeypatch):
        dest = make_repo(tmp_path)

        def boom(prompt, **k):
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(triage_repo.llm, "complete", boom)
        out = triage_repo.score_one(dest, self.row(dest), "RUBRIC", None)
        assert out == {"repo": "Auquan/Tutorials", "kind": "notebook",
                       "path": "nb.ipynb",
                       "meta": {"kernel": "python3", "cells": 4,
                                "has_outputs": True},
                       "error": "`claude` CLI not found on PATH"}
        assert llm.is_cli_error(out["error"]) is True

    def test_missing_artifact_is_unreadable(self, tmp_path):
        dest = make_repo(tmp_path)
        row = {"repo": "Auquan/Tutorials", "kind": "notebook",
               "path": "gone.ipynb", "meta": {}}
        out = triage_repo.score_one(dest, row, "RUBRIC", None)
        assert out["error"].startswith("unreadable artifact:")
        assert llm.is_cli_error(out["error"]) is False

    def test_corrupt_notebook_is_unreadable(self, tmp_path):
        dest = make_repo(tmp_path)
        (dest / "nb.ipynb").write_text("{oops")
        out = triage_repo.score_one(dest, self.row(dest), "RUBRIC", None)
        assert out["error"].startswith("unreadable artifact:")


class TestMainWith:
    def test_resume_skips_scored_artifacts(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        monkeypatch.setattr(triage_repo.llm, "complete",
                            lambda prompt, **k: json.dumps(FULL_REPLY))
        assert triage_repo.main_with(["--jobs", "2"], root=tmp_path) == 0
        first = triage_repo.load(triage_repo.TRIAGE)
        assert len(first) == 3

        capsys.readouterr()
        assert triage_repo.main_with([], root=tmp_path) == 0
        out = capsys.readouterr().out
        assert "nothing to triage" in out
        assert triage_repo.load(triage_repo.TRIAGE) == first

    def test_rescore_replaces_rows_without_duplicates(self, tmp_path,
                                                      monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        reply = {"payload": dict(FULL_REPLY, verdict="SKIP")}

        def complete_a(prompt, **k):
            return json.dumps(reply["payload"])

        monkeypatch.setattr(triage_repo.llm, "complete", complete_a)
        assert triage_repo.main_with([], root=tmp_path) == 0
        assert {r["verdict"] for r in triage_repo.load(triage_repo.TRIAGE)} == {"SKIP"}

        reply["payload"] = dict(FULL_REPLY, verdict="RUN_WITH_OUR_DATA")
        capsys.readouterr()
        assert triage_repo.main_with(["--rescore"], root=tmp_path) == 0
        rows = triage_repo.load(triage_repo.TRIAGE)
        keys = [(r["repo"], r["kind"], r["path"]) for r in rows]
        assert len(keys) == len(set(keys)) == 3
        assert {r["verdict"] for r in rows} == {"RUN_WITH_OUR_DATA"}

    def test_limit_caps_the_batch(self, tmp_path, monkeypatch):
        stage(tmp_path, monkeypatch)
        calls = []

        def complete(prompt, **k):
            calls.append(prompt)
            return json.dumps(FULL_REPLY)

        monkeypatch.setattr(triage_repo.llm, "complete", complete)
        assert triage_repo.main_with(["--limit", "2"], root=tmp_path) == 0
        assert len(calls) == 2
        assert len(triage_repo.load(triage_repo.TRIAGE)) == 2

    def test_circuit_breaker_aborts_and_keeps_completed(self, tmp_path,
                                                         monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        monkeypatch.setattr(triage_repo.llm, "circuit_break", REAL_CIRCUIT_BREAK)

        def complete(prompt, **k):
            raise llm.LLMError("opencode exited 1: auth expired")

        monkeypatch.setattr(triage_repo.llm, "complete", complete)
        assert triage_repo.main_with(["--jobs", "1"], root=tmp_path) == 1
        out = capsys.readouterr().out
        assert "aborting" in out
        saved = triage_repo.load(triage_repo.TRIAGE)
        assert len(saved) == 3  # CIRCUIT_K completed rows, nothing more
        assert all(e["error"].startswith("opencode exited 1:")
                   for e in saved)

    def test_bad_spec_exits_1_on_stderr(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)
        assert triage_repo.main_with(["not-a-spec"], root=tmp_path) == 1
        assert "bad spec" in capsys.readouterr().err

    def test_missing_manifest_exits_1_on_stderr(self, tmp_path, monkeypatch,
                                                capsys):
        stage(tmp_path, monkeypatch)
        assert triage_repo.main_with(["Some/Repo"], root=tmp_path) == 1
        assert "no manifest for Some/Repo -- run harvest_repo.py first" \
            in capsys.readouterr().err

    def test_corrupt_manifest_exits_1_on_stderr(self, tmp_path, monkeypatch,
                                                capsys):
        dest = stage(tmp_path, monkeypatch)
        (dest / "manifest.json").write_text("{broken")
        assert triage_repo.main_with([], root=tmp_path) == 1
        assert "error:" in capsys.readouterr().err

    def test_missing_prompt_exits_1_on_stderr(self, tmp_path, monkeypatch,
                                              capsys):
        stage(tmp_path, monkeypatch)
        monkeypatch.setattr(triage_repo, "PROMPT", tmp_path / "nope.md")
        assert triage_repo.main_with([], root=tmp_path) == 1
        assert "cannot read prompt" in capsys.readouterr().err

    def test_preflight_failure_exits_with_llm_error(self, monkeypatch):
        def boom():
            raise llm.LLMError("`claude` CLI not found on PATH")

        monkeypatch.setattr(triage_repo.llm, "preflight", boom)
        with pytest.raises(SystemExit, match="not found on PATH"):
            triage_repo.main_with([])

    def test_resume_key_is_repo_scoped_across_repos(self, tmp_path,
                                                     monkeypatch, capsys):
        # Two repos with artifacts at IDENTICAL relative paths (the pilot
        # case: three repos all ship a README.md spec). Scoring repo A must
        # not mark repo B's same-path artifacts as already seen.
        stage(tmp_path, monkeypatch)
        make_repo(tmp_path, owner="Last", name="Ancient")
        calls = []

        def complete(prompt, **k):
            calls.append(prompt)
            return json.dumps(FULL_REPLY)

        monkeypatch.setattr(triage_repo.llm, "complete", complete)
        assert triage_repo.main_with(["Auquan/Tutorials"], root=tmp_path) == 0
        assert len(calls) == 3

        capsys.readouterr()
        calls.clear()
        assert triage_repo.main_with(["Last/Ancient"], root=tmp_path) == 0
        assert len(calls) == 3  # same paths in a new repo must be re-scored
        rows = triage_repo.load(triage_repo.TRIAGE)
        assert len(rows) == 6
        assert {r["repo"] for r in rows} == {"Auquan/Tutorials", "Last/Ancient"}
        assert len({(r["repo"], r["kind"], r["path"]) for r in rows}) == 6

    def test_error_rows_stay_scored_until_rescore(self, tmp_path, monkeypatch,
                                                  capsys):
        # A permanently-failing artifact must not burn LLM calls on every
        # run: its error row counts as scored until --rescore retries it.
        stage(tmp_path, monkeypatch)
        replies = {"script": "not json at all"}
        calls = []

        def complete(prompt, **k):
            calls.append(prompt)
            kind = "script" if '"kind": "script"' in prompt else (
                "spec" if '"kind": "spec"' in prompt else "notebook")
            if kind in replies:
                return replies[kind]
            return json.dumps(FULL_REPLY)

        monkeypatch.setattr(triage_repo.llm, "complete", complete)
        assert triage_repo.main_with([], root=tmp_path) == 0
        rows = triage_repo.load(triage_repo.TRIAGE)
        errors = [r for r in rows if "error" in r]
        assert len(rows) == 3 and len(errors) == 1
        assert errors[0]["kind"] == "script"
        assert errors[0]["error"].startswith("unparseable reply:")

        capsys.readouterr()
        calls.clear()
        assert triage_repo.main_with([], root=tmp_path) == 0
        out = capsys.readouterr().out
        assert "nothing to triage" in out  # error row was NOT retried
        assert calls == []
        assert triage_repo.load(triage_repo.TRIAGE) == rows

        replies.clear()
        capsys.readouterr()
        assert triage_repo.main_with(["--rescore"], root=tmp_path) == 0
        rows = triage_repo.load(triage_repo.TRIAGE)
        script = next(r for r in rows if r["kind"] == "script")
        assert "error" not in script  # retried and now scored
        assert script["verdict"] == "RUN_WITH_OUR_DATA"
        assert len(rows) == 3

    def test_verdict_summary_counts_whole_file_including_errors(self, tmp_path,
                                                               monkeypatch,
                                                               capsys):
        # The distribution sanity check only works if the summary counts
        # every row in repo_triage.jsonl -- including rows from earlier
        # runs and the error= bucket -- not just this run's results.
        stage(tmp_path, monkeypatch)
        triage_repo.save(triage_repo.TRIAGE, [
            {"repo": "Other/Repo", "kind": "script", "path": "old.py",
             "meta": {}, "verdict": "SKIP", "scored_at": "2026-01-01T00:00:00Z"},
            {"repo": "Other/Repo", "kind": "notebook", "path": "old.ipynb",
             "meta": {}, "verdict": "NEEDS_HEAVY_ENV",
             "scored_at": "2026-01-01T00:00:00Z"},
            {"repo": "Other/Repo", "kind": "spec", "path": "bad.yaml",
             "meta": {}, "error": "unparseable reply: no JSON object"},
        ])
        monkeypatch.setattr(triage_repo.llm, "complete",
                            lambda prompt, **k: json.dumps(FULL_REPLY))
        assert triage_repo.main_with([], root=tmp_path) == 0
        out = capsys.readouterr().out
        assert ("verdicts: NEEDS_HEAVY_ENV=1, RUN_WITH_OUR_DATA=3, "
                "SKIP=1, error=1") in out

    def test_verdict_distribution_summary(self, tmp_path, monkeypatch, capsys):
        stage(tmp_path, monkeypatch)

        def complete(prompt, **k):
            verdict = "SKIP" if '"kind": "script"' in prompt else "RUN_WITH_OUR_DATA"
            return json.dumps(dict(FULL_REPLY, verdict=verdict))

        monkeypatch.setattr(triage_repo.llm, "complete", complete)
        assert triage_repo.main_with([], root=tmp_path) == 0
        out = capsys.readouterr().out
        assert "verdicts: RUN_WITH_OUR_DATA=2, SKIP=1" in out
        assert "scores -> data/repo_triage.jsonl" in out
        assert re.search(
            r"\[\d/3\] RUN_WITH_OUR_DATA\s+notebook Auquan/Tutorials:nb\.ipynb",
            out)
        assert re.search(r"\[\d/3\] SKIP\s+script Auquan/Tutorials:run\.py", out)
