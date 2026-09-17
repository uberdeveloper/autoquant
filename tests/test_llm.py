"""Tests for llm.py — provider-agnostic single-shot CLI adapter."""
from __future__ import annotations

import subprocess

import pytest

import llm


def clear_env(monkeypatch):
    monkeypatch.delenv("AUTOQUANT_LLM_CMD", raising=False)
    monkeypatch.delenv("AUTOQUANT_LLM_EXTRA_ARGS", raising=False)


class TestBuildArgv:
    def test_default_cmd(self, monkeypatch):
        clear_env(monkeypatch)
        assert llm.build_argv() == ["opencode", "run"]

    def test_model_appended_when_no_placeholder(self, monkeypatch):
        clear_env(monkeypatch)
        assert llm.build_argv("sonnet") == ["opencode", "run", "--model", "sonnet"]

    def test_no_model_no_flag(self, monkeypatch):
        clear_env(monkeypatch)
        assert "--model" not in llm.build_argv()

    def test_env_swaps_whole_command(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "codex exec")
        assert llm.build_argv() == ["codex", "exec"]

    def test_model_placeholder_substituted(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "opencode run {model}")
        assert llm.build_argv("sonnet") == ["opencode", "run", "sonnet"]

    def test_model_placeholder_empty_when_model_is_none(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "opencode run {model}")
        # shlex drops the empty token, so an absent model contributes no arg
        assert llm.build_argv(None) == ["opencode", "run"]

    def test_env_extra_args_appended_last(self, monkeypatch):
        clear_env(monkeypatch)
        monkeypatch.setenv("AUTOQUANT_LLM_EXTRA_ARGS", "--max-turns 1")
        assert llm.build_argv(extra=["--verbose"]) == \
            ["opencode", "run", "--verbose", "--max-turns", "1"]


class TestPreflight:
    def test_missing_cli_raises(self, monkeypatch):
        monkeypatch.setenv("AUTOQUANT_LLM_CMD", "definitely-not-a-real-cli-xyz -p")
        with pytest.raises(llm.LLMError, match="not found on PATH"):
            llm.preflight()

    def test_present_cli_passes(self, monkeypatch):
        monkeypatch.setattr(llm.shutil, "which", lambda name: "/usr/bin/opencode")
        assert llm.preflight() is None


class TestComplete:
    def stub_run(self, monkeypatch, *, returncode=0, stdout="ok", stderr="",
                 raise_exc=None):
        recorded = {}

        def fake_run(argv, input, capture_output, text, timeout):
            recorded["argv"] = argv
            recorded["input"] = input
            recorded["timeout"] = timeout
            if raise_exc is not None:
                raise raise_exc

            class Proc:
                pass

            Proc.returncode = returncode
            Proc.stdout = stdout
            Proc.stderr = stderr
            return Proc()

        monkeypatch.setattr(llm.subprocess, "run", fake_run)
        return recorded

    def test_returns_stdout_and_forwards_prompt_and_timeout(self, monkeypatch):
        clear_env(monkeypatch)
        rec = self.stub_run(monkeypatch, stdout="reply text")
        assert llm.complete("the prompt", timeout=300) == "reply text"
        assert rec["argv"] == ["opencode", "run"]
        assert rec["input"] == "the prompt"
        assert rec["timeout"] == 300

    def test_model_and_extra_reach_argv(self, monkeypatch):
        clear_env(monkeypatch)
        rec = self.stub_run(monkeypatch)
        llm.complete("p", model="sonnet", extra=["--output-format", "json"])
        assert rec["argv"] == ["opencode", "run", "--model", "sonnet",
                               "--output-format", "json"]

    def test_nonzero_exit_raises_llmerror_with_stderr(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, returncode=1, stderr="auth expired")
        with pytest.raises(llm.LLMError, match=r"opencode exited 1: auth expired"):
            llm.complete("p")

    def test_timeout_raises_llmerror(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch,
                      raise_exc=subprocess.TimeoutExpired(cmd="opencode", timeout=300))
        with pytest.raises(llm.LLMError, match="timed out after 300s"):
            llm.complete("p", timeout=300)

    def test_missing_binary_raises_llmerror(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, raise_exc=FileNotFoundError(2, "No such file"))
        with pytest.raises(llm.LLMError, match="not found on PATH"):
            llm.complete("p")

    def test_stderr_truncated_to_200_chars(self, monkeypatch):
        clear_env(monkeypatch)
        self.stub_run(monkeypatch, returncode=1, stderr="x" * 500)
        with pytest.raises(llm.LLMError) as excinfo:
            llm.complete("p")
        assert len(str(excinfo.value)) < 250


class TestCircuitBreaker:
    def test_cli_level_errors_classified(self):
        assert llm.is_cli_error("opencode exited 1: auth expired")
        assert llm.is_cli_error("`claude` CLI not found on PATH")
        assert llm.is_cli_error("opencode CLI timed out after 300s")

    def test_stage_level_errors_not_classified(self):
        assert not llm.is_cli_error("unparseable YAML: while scanning")
        assert not llm.is_cli_error("invalid spec: missing costs")
        assert not llm.is_cli_error("unreadable spec: boom")
        assert not llm.is_cli_error("empty reply")
        assert not llm.is_cli_error("meta.slug does not match spec filename")

    def test_breaker_quiet_below_k(self):
        results = [{"error": "opencode exited 1: auth"}]
        assert llm.circuit_break(results) is None

    def test_breaker_trips_on_k_consecutive_cli_errors(self):
        results = [{"error": "opencode exited 1: auth"} for _ in range(3)]
        msg = llm.circuit_break(results)
        assert msg is not None and "aborting" in msg and "3" in msg

    def test_breaker_spared_by_a_stage_level_error(self):
        results = [{"error": "opencode exited 1: auth"},
                   {"error": "unparseable YAML: x"},
                   {"error": "opencode exited 1: auth"}]
        assert llm.circuit_break(results) is None

    def test_breaker_ignores_successes(self):
        results = [{"score": 4}, {"score": 3}, {"error": "opencode exited 1: a"}]
        assert llm.circuit_break(results) is None

    def test_breaker_k_is_configurable(self):
        results = [{"error": "opencode exited 1: a"} for _ in range(5)]
        assert llm.circuit_break(results, k=5) is not None
        assert llm.circuit_break(results[:4], k=5) is None
