"""Single-shot text completion via a configurable CLI.

The three LLM stages (triage.py, extract.py, codegen.py) call complete()
instead of assembling their own LLM CLI invocation. The provider is
swapped through env vars, no code edits:

    AUTOQUANT_LLM_CMD="opencode run"           # default
    AUTOQUANT_LLM_CMD="claude -p"              # any text-in/text-out CLI
    AUTOQUANT_LLM_CMD="crush run -q {model}"   # {model} placeholder optional

Extra arguments reach the CLI two ways: globally via AUTOQUANT_LLM_EXTRA_ARGS
(shlex-split), or per invocation via the `extra` parameter (wired to each
stage's repeatable --llm-arg flag). Without a {model} placeholder, --model
<model> is appended when a model is given.

This module is also the single place that handles a missing CLI: preflight()
checks PATH before a batch starts, and complete() turns FileNotFoundError
into LLMError so a mid-run failure becomes a per-item error, never a crash.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess

DEFAULT_CMD = "opencode run"
CLI_TIMEOUT = 600


class LLMError(Exception):
    """A completion failed: CLI missing, timed out, or nonzero exit."""


def build_argv(model: str | None = None,
               extra: list[str] | None = None) -> list[str]:
    """Assemble the CLI argv from AUTOQUANT_LLM_CMD (default: `opencode run`)."""
    template = os.environ.get("AUTOQUANT_LLM_CMD", DEFAULT_CMD)
    if "{model}" in template:
        argv = shlex.split(template.format(model=model or ""))
    else:
        argv = shlex.split(template)
        if model:
            argv += ["--model", model]
    argv += list(extra or [])
    argv += shlex.split(os.environ.get("AUTOQUANT_LLM_EXTRA_ARGS", ""))
    return argv


def preflight() -> None:
    """Raise LLMError early if the configured CLI is not on PATH."""
    template = os.environ.get("AUTOQUANT_LLM_CMD", DEFAULT_CMD)
    argv = shlex.split(template)
    if shutil.which(argv[0]) is None:
        raise LLMError(f"`{argv[0]}` CLI not found on PATH "
                       f"(AUTOQUANT_LLM_CMD={template!r})")


def complete(prompt: str, model: str | None = None,
             extra: list[str] | None = None,
             timeout: int = CLI_TIMEOUT) -> str:
    """One single-shot CLI completion: prompt in, text out.

    Raises LLMError on a missing binary, a timeout, or a nonzero exit, so
    callers only need one except clause to produce their error artifact.
    """
    argv = build_argv(model, extra)
    try:
        proc = subprocess.run(argv, input=prompt, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise LLMError(f"{argv[0]} CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise LLMError(f"`{argv[0]}` CLI not found on PATH")
    if proc.returncode != 0:
        raise LLMError(f"{argv[0]} exited {proc.returncode}: {proc.stderr[:200]}")
    return proc.stdout


CIRCUIT_K = 3  # consecutive CLI-level failures before a batch aborts

_STAGE_LEVEL_PREFIXES = ("unparseable", "invalid spec:", "unreadable spec:",
                         "empty reply", "meta.slug does not match")


def is_cli_error(error: str) -> bool:
    """True if an error-row string came from the LLM CLI plumbing
    (missing binary, nonzero exit, timeout) rather than the stage's own
    reply parsing/validation."""
    if error.startswith(_STAGE_LEVEL_PREFIXES):
        return False
    return ("CLI not found on PATH" in error
            or "CLI timed out after" in error
            or bool(re.match(r"^\S+ exited \d+:", error)))


def circuit_break(results: list[dict], k: int = CIRCUIT_K) -> str | None:
    """Abort message if the first k completed results are ALL CLI-level
    failures -- the provider is down or unauthenticated, and continuing
    would only write N identical error artifacts. None means keep going."""
    head = results[:k]
    if len(head) < k:
        return None
    if all("error" in r and is_cli_error(r["error"]) for r in head):
        first = head[0]["error"]
        return (f"aborting: first {k} results all failed at the CLI level "
                f"(first error: {first[:160]}) -- fix the LLM CLI/auth "
                f"and re-run; completed work is kept")
    return None
