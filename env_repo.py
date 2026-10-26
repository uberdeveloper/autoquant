"""Stage [3c] ENV_REPO -- build one isolated execution environment per repo.

Third-party repos carry conflicting dependencies (Auquan is 2018-era pandas,
the rotman RL repo pins torch 1.11 + numpy 1.23 against our pandas>=2.0 /
numpy>=1.26).  This stage gives every harvested repo its own virtualenv so
that **no repo dependency ever reaches the main project env or uv.lock**:
`uv venv --no-project --clear` keeps the project out of the way and wipes
any leftover venv, and
`uv pip install --python <venv>/bin/python` installs into the venv, not into
the project.

CLI usage:
    python3 env_repo.py                          # every harvested repo
    python3 env_repo.py Auquan/Tutorials         # one repo (repeatable)
    python3 env_repo.py --repos list.txt         # one spec per line
    python3 env_repo.py --rebuild                # force rebuild even if the key matches
    python3 env_repo.py --legacy-python 3.9      # reserved for #27's fallback builds

Output: data/repo_envs.jsonl, one row per repo.  Resumable: a repo whose
recorded env key still matches (and whose venv is still on disk) is reported
as reused, not rebuilt.

Scope: this stage builds environments, it does not execute anything.  The
legacy fallback is a *plan*, not an action: when #27's `run_external.py`
catches a legacy pandas/numpy API break it calls
`plan_fallback(row, error_text)`; when that returns a dict, #27 builds the
fallback env itself, reading the values out of the plan (they may be
overrides, so never assume the module constants):

    build_env(dest, manifest, label, mode="legacy-fallback",
              python=plan["legacy_python"], requirements=plan["legacy_pins"],
              root=root)

`plan_fallback` returns the keys `legacy_python` / `legacy_pins` -- it does
not return `LEGACY_PYTHON` / `LEGACY_PINS`.

MUST-HONOUR SIGNAL: a non-empty `blocked` list on the row is the SKIP signal.
#27 MUST downgrade those artifacts to SKIP with the recorded reason; a
non-empty `blocked` list is never ignorable.

ROW CONTRACT for #27 (read before consuming a row):

* A row built with an explicit `requirements=` argument (the legacy
  fallback) keys its `env_key` and `deps_sha256` over the JOINED PINS, not
  over any file's text.  Such a row is ephemeral: a later `main_with` run
  for the same repo rebuilds the row from the repo's own dep file and
  REPLACES it.
* `mode` is the discriminator for a row whose `python` is null: "main" means
  the repo runs in the main project env (no venv, no pins), while
  "repo-venv" and "legacy-fallback" are rows with a real interpreter.
* A row whose `status` is "failed" has an EMPTY `blocked` list, so
  consumers must NEVER read `blocked` off a failed row -- the SKIP signal
  exists only on "ready" rows.

This stage never writes to `data/repo_triage.jsonl` -- `triage_repo.py` owns
that file (one writer per file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import harvest_repo
import yaml

ROOT = Path(__file__).resolve().parent
REPOS = ROOT / "data" / "repos"
ENVS = ROOT / "data" / "repo_envs.jsonl"

VENV_DIRNAME = ".venv"
LEGACY_VENV_DIRNAME = ".venv-legacy"

# The legacy fallback interpreter and the pins that are known to import
# cleanly on it (pandas 1.x era).
LEGACY_PYTHON = "3.9"
LEGACY_PINS = [
    "pandas==1.3.5",
    "numpy==1.22.4",
    "scipy==1.7.3",
    "scikit-learn==1.0.2",
    "matplotlib==3.5.3",
    "statsmodels==0.13.2",
    "python-dateutil==2.9.0.post0",
    "pytz",
    "pyyaml",
    "requests",
    "beautifulsoup4",
    "pandas-datareader",
]

# Repo venvs use a fixed interpreter, not the ambient one, so a repo env is
# the same env on every machine.  3.10 (not 3.11) because torch==1.11.0
# resolves on CPython 3.10 but has no cp311 wheels, and a repo venv's
# purpose is to satisfy the repo's own pins.
DEFAULT_PYTHON = "3.10"

DEFAULT_TIMEOUT = 1800
UV_TIMEOUT = DEFAULT_TIMEOUT  # uv resolve/install budget (torch is slow)

TALIB_REASON = ("ta-lib needs a C library (TA-Lib) that is not installed; "
                "not built")

_TALIB_RE = re.compile(r"(?<![A-Za-z0-9_.\-])ta[-_]?lib(?![A-Za-z0-9_])|import\s+talib|from\s+talib",
                        re.IGNORECASE)

# Signatures of a pandas/numpy *API break* (as opposed to a missing dep).
# "No module named ..." is deliberately absent: that is a missing dependency,
# not an API break, and falling back would not fix it.
LEGACY_API_SIGNATURES = (
    r"object has no attribute ['\"]append['\"]",
    r"object has no attribute ['\"]ix['\"]",
    r"object has no attribute ['\"]iteritems['\"]",
    r"(?:has no attribute|import name|cannot import name)\s+['\"]Panel['\"]",
    r"module ['\"]numpy['\"] has no attribute ['\"](?:bool8?|float|int|object|str)['\"]",
    r"\bnp\.(?:bool8?|float|int|object|str)\b",
)


def load(path: Path) -> list[dict]:
    """Read a jsonl file into a list of dicts (missing file -> empty list)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    """Write rows to a jsonl file atomically (tmp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def _rel(path: Path, root: Path) -> str:
    """Path relative to the project root, so rows stay machine-independent."""
    path = Path(path)
    try:
        return str(path.relative_to(Path(root)))
    except ValueError:
        return str(path)


def deps_source(dest: Path, manifest: dict) -> str:
    """The install source for a repo: requirements.txt, environment.yml, none.

    The manifest's `deps` flags are the contract (exactly as in #24's
    `repo_deps`): a `setup.py` alone is not a pip-installable source, so such
    a repo yields "none" and runs in the main env.
    """
    flags = manifest.get("deps") if isinstance(manifest, dict) else None
    if not isinstance(flags, dict):
        flags = {}
    if flags.get("requirements_txt"):
        return "requirements.txt"
    if flags.get("environment_yml"):
        return "environment.yml"
    return "none"


def read_requirements_detailed(path: Path) -> tuple[list[str], int]:
    """(installable requirement lines, dropped pip flags) of a requirements.txt.

    Comments, blank lines and pip flags (-r/--requirement, -e/--editable,
    --index-url, ...) are dropped: they are not installable pins and
    `uv pip install -r` would fail on the unsupported ones.  The number of
    dropped flags is returned so the row can record what was left out.
    """
    out: list[str] = []
    dropped = 0
    for line in Path(path).read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("-"):  # a pip flag, not a requirement
            dropped += 1
            continue
        s = re.sub(r"\s+#.*$", "", s).strip()
        if s:
            out.append(s)
    return out, dropped


def read_requirements(path: Path) -> list[str]:
    """Installable requirement lines of a requirements.txt."""
    return read_requirements_detailed(path)[0]


# Names that are interpreters or installers, not distributable packages:
# asking pip for a distribution called "python" fails the whole build.
_PIP_EXCLUDED_NAMES = frozenset({"python", "pip", "setuptools"})

_PLAIN_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+")


def _spec_name(spec: str) -> str:
    """The distribution name of a requirement spec (lower-cased)."""
    m = _NAME_RE.match(spec.strip())
    return m.group(0).lower() if m else ""


def _is_pip_spec(spec: str) -> bool:
    """True for a conda environment.yml entry that pip can install.

    Conda matchspecs ("numpy=1.23", "python=3.9") are not pip requirements;
    conda-only entries are dropped and counted instead.  A bare name
    ("pandas", "pytz") is a perfectly installable pip requirement and is
    kept.  Interpreter/installer names ("python", "pip", "setuptools", and
    any version spec of them) are never pip requirements.
    """
    s = spec.strip()
    if not s or s.startswith("-"):
        return False
    if _spec_name(s) in _PIP_EXCLUDED_NAMES:
        return False
    if "==" in s:
        return True
    if any(op in s for op in ("<", ">", "~=", "!=")):
        return True
    if "=" in s:  # a conda pin ("numpy=1.23"), not a pip requirement
        return False
    return bool(_PLAIN_NAME_RE.fullmatch(s))


def _environment_yml_requirements(path: Path) -> tuple[list[str], int]:
    """(pip requirements, dropped conda entries) of a conda environment.yml.

    A file that is not valid YAML yields ([], 0) instead of raising: one
    corrupt file must not abort a whole batch (the caller decides what to do
    with it -- `env_state` turns a corrupt file into a failed row).
    """
    try:
        obj = yaml.safe_load(Path(path).read_text(errors="replace"))
    except yaml.YAMLError:
        return [], 0
    deps = obj.get("dependencies") if isinstance(obj, dict) else None
    out: list[str] = []
    dropped = 0
    if isinstance(deps, list):
        for item in deps:
            if isinstance(item, str):
                if _is_pip_spec(item):
                    out.append(item)
                else:
                    dropped += 1
            elif isinstance(item, dict):
                pip = item.get("pip")
                if isinstance(pip, list):
                    for s in pip:
                        if not isinstance(s, str):
                            continue
                        if _spec_name(s) in _PIP_EXCLUDED_NAMES:
                            dropped += 1
                            continue
                        out.append(s)
    return out, dropped


_PYTHON_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def _normalise_python_spec(spec: str) -> str | None:
    """A plain dotted python version, or None when the spec is not one.

    `uv venv --python` only accepts a plain version ("3.9", "3.9.7"); a conda
    matchspec ("3.9.*") or a conda build string ("3.9.7=h1234") makes uv
    hard-fail the whole build.

    The boundary: a trailing ".*" wildcard is stripped and the rest must be
    a plain dotted version ("3.9.*" -> "3.9", "3.*" -> "3", "3" -> "3").
    Anything else -- a wildcard that is not a trailing ".*", a build string,
    garbage -- returns None, so the caller falls back to DEFAULT_PYTHON.
    """
    s = spec.strip()
    if s.endswith(".*"):
        s = s[:-2]
    return s if _PYTHON_VERSION_RE.match(s) else None


def environment_python(yml_text: str) -> str | None:
    """The conda `python=` pin of an environment.yml as a plain version.

    The pin is the repo's own declaration of the interpreter it expects;
    when it is present it is the interpreter the env is built with.  It is
    normalised to a plain dotted version (see `_normalise_python_spec`), so
    it can be handed to `uv venv --python` unchanged; a pin that is not a
    usable version yields None, and the caller falls back to DEFAULT_PYTHON.
    """
    try:
        obj = yaml.safe_load(yml_text)
    except yaml.YAMLError:
        return None
    deps = obj.get("dependencies") if isinstance(obj, dict) else None
    if not isinstance(deps, list):
        return None
    for item in deps:
        if not isinstance(item, str):
            continue
        m = re.match(r"^\s*python\s*={1,2}\s*(\S+)\s*$", item)
        if m:
            return _normalise_python_spec(m.group(1))
    return None


def pip_requirements_from_environment_yml(path: Path) -> list[str]:
    """The pip-installable requirements of a conda environment.yml."""
    return _environment_yml_requirements(path)[0]


def env_key(sha: str, deps_text: str) -> str:
    """Reuse key of a repo env: 16 hex chars of sha256(sha + deps text)."""
    return hashlib.sha256(f"{sha}\n{deps_text}".encode()).hexdigest()[:16]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _manifest_sha(manifest: dict) -> str:
    rec = manifest.get("repo") if isinstance(manifest, dict) else None
    sha = rec.get("sha") if isinstance(rec, dict) else None
    return sha if isinstance(sha, str) else ""


def _source_text(path: Path) -> str:
    """Text of a script or of a notebook's cells."""
    path = Path(path)
    if path.suffix.lower() != ".ipynb":
        return path.read_text(errors="replace")
    obj = json.loads(path.read_text(errors="replace"))
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []
    for cell in obj.get("cells", []):
        if not isinstance(cell, dict):
            continue
        src = cell.get("source")
        if isinstance(src, list):
            src = "".join(src)
        if isinstance(src, str):
            parts.append(src)
    return "\n".join(parts)


def talib_required(dest: Path, manifest: dict) -> list[dict]:
    """Artifacts that need ta-lib, which needs a C library we do not have.

    The dep file text and every notebook/script recorded in the manifest are
    scanned for ta-lib references.  The hits become `blocked` entries; #27
    downgrades those artifacts to SKIP with the recorded reason.
    """
    dest = Path(dest)
    out: list[dict] = []
    source = deps_source(dest, manifest)
    if source != "none":
        if _TALIB_RE.search(_read_text(dest / source)):
            out.append({"kind": "deps", "path": source, "reason": TALIB_REASON})
    entries = []
    if isinstance(manifest, dict):
        entries.extend(manifest.get("notebooks", []) or [])
        entries.extend(manifest.get("scripts", []) or [])
    for entry in entries:
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not rel:
            continue
        try:
            text = _source_text(dest / rel)
        except (OSError, json.JSONDecodeError):
            continue
        if _TALIB_RE.search(text):
            kind = "notebook" if rel.endswith(".ipynb") else "script"
            out.append({"kind": kind, "path": rel, "reason": TALIB_REASON})
    return out


def is_legacy_api_error(text: str) -> bool:
    """True when `text` looks like a pandas/numpy API break.

    A missing module is *not* an API break, so "No module named ..." never
    matches.
    """
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in LEGACY_API_SIGNATURES)


def plan_fallback(row: dict, error_text: str,
                  python: str | None = None) -> dict | None:
    """Plan the legacy fallback for a failed repo, or None when it does not apply.

    The fallback only makes sense when the repo declared no pins of its own
    (a newer library version is then the likely culprit) and the failure is a
    pandas/numpy API break.  Exactly one retry is allowed; the fallback env
    is never retried into a third env.
    """
    if not isinstance(row, dict):
        return None
    if row.get("mode") == "legacy-fallback" or row.get("fallback_reason"):
        return None  # exactly one retry: never a fallback of a fallback
    if row.get("deps_source") != "none":
        return None
    if not is_legacy_api_error(error_text):
        return None
    return {"mode": "legacy-fallback",
            "fallback_reason": (error_text or "").strip()[:300],
            "legacy_python": python or LEGACY_PYTHON,
            "legacy_pins": list(LEGACY_PINS)}


def run_uv(argv: list[str], timeout: int = UV_TIMEOUT, *,
           cwd: Path | None = None) -> str:
    """Run a uv command, returning stripped stdout.

    This is the module's ONLY subprocess call site (mirroring
    `harvest_repo.run_git`), so tests can monkeypatch it and never touch the
    network.  Failures and timeouts become RuntimeError with the message uv
    printed.
    """
    try:
        r = subprocess.run(argv, capture_output=True, text=True, check=True,
                           timeout=timeout, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError((exc.stderr or exc.stdout or
                            f"command failed: {' '.join(argv)}").strip()) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"uv timed out after {timeout}s: "
                           f"{' '.join(argv)}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"uv not found on PATH: {' '.join(argv)}") from exc
    return (r.stdout or "").strip()


def _environment_yml_parses(text: str) -> bool:
    """True when `text` is valid YAML (a corrupt file is not usable)."""
    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def env_state(dest: Path, manifest: dict) -> dict:
    """Everything the reuse key and the row are made of, for one repo.

    `deps_source` is "none" for a repo without pins; then the requirements
    list is empty and the dep text is empty too (the key still changes with
    the repo's SHA, so a new commit re-checks the repo).

    A declared dep file that is not on disk is an error, not an empty
    install: `uv pip install -r <empty file>` is a silent no-op, so the env
    would be reused forever without ever getting its packages.
    """
    dest = Path(dest)
    source = deps_source(dest, manifest)
    if source != "none" and not (dest / source).exists():
        raise ValueError(f"{source} declared by manifest but missing")
    deps_text = ""
    requirements: list[str] = []
    dropped = 0
    dropped_flags = 0
    conda_python = None
    if source == "requirements.txt":
        deps_text = _read_text(dest / source)
        requirements, dropped_flags = (
            read_requirements_detailed(dest / source) if deps_text else ([], 0))
    elif source == "environment.yml":
        deps_text = _read_text(dest / source)
        if deps_text:
            if not _environment_yml_parses(deps_text):
                raise ValueError("environment.yml is not valid YAML")
            requirements, dropped = _environment_yml_requirements(dest / source)
            conda_python = environment_python(deps_text)
    sha = _manifest_sha(manifest)
    return {"deps_source": source, "deps_text": deps_text,
            "requirements": requirements, "dropped_conda_deps": dropped,
            "dropped_pip_flags": dropped_flags, "conda_python": conda_python,
            "sha": sha, "deps_sha256": _sha256(deps_text),
            "env_key": env_key(sha, deps_text)}


def _venv_python_version(venv: Path) -> str | None:
    """The interpreter version of a venv, read from its pyvenv.cfg."""
    for line in _read_text(Path(venv) / "pyvenv.cfg").splitlines():
        if line.startswith("version_info"):
            return line.split("=", 1)[1].strip() or None
    return None


def build_env(dest: Path, manifest: dict, label: str, *,
              state: dict | None = None,
              requirements: list[str] | None = None,
              python: str | None = None,
              timeout: int = UV_TIMEOUT,
              mode: str = "repo-venv",
              root: Path | None = None) -> dict:
    """Build the repo's virtualenv and return its row.

    A repo without declared pins gets no venv at all: it runs in the main
    env (`mode` "main", `venv` null).  Otherwise two uv calls are made, in
    this order:

        uv venv --no-project --clear --python <py> <dest>/.venv
        uv pip install --python <venv>/bin/python -r <tmpfile>

    `--no-project` keeps uv from discovering (and enforcing) our project's
    requires-python, and `--python <venv>/bin/python` installs into the venv
    instead of the project env -- that pair is what keeps repo dependencies
    out of `uv.lock`.  `--clear` wipes a leftover venv directory, so a
    rebuild cannot inherit content from a previous build.  The filtered
    requirement list goes into a temp file inside the repo's own directory
    (so relative paths resolve) and is removed afterwards;
    `requirements.txt` is never modified.

    The interpreter is: for a legacy fallback, the explicit `python`
    argument (the plan's `legacy_python`) or LEGACY_PYTHON; for a repo with
    an `environment.yml`, the conda `python=` pin that file declares; for
    every other repo, the explicit `python` argument or DEFAULT_PYTHON.
    """
    dest = Path(dest).resolve()
    root = Path(root) if root is not None else ROOT
    st = state if state is not None else env_state(dest, manifest)
    source = st["deps_source"]
    if requirements is None:
        reqs = list(st["requirements"])
        deps_text = st["deps_text"]
    else:
        reqs = list(requirements)
        deps_text = "\n".join(reqs)
    sha = st["sha"]
    row = {
        "repo": label,
        "path": _rel(dest, root),
        "mode": mode,
        "deps_source": source,
        "venv": None,
        "python": None,
        "sha": sha,
        "deps_sha256": _sha256(deps_text),
        "env_key": env_key(sha, deps_text),
        "dropped_conda_deps": st.get("dropped_conda_deps", 0),
        "dropped_pip_flags": st.get("dropped_pip_flags", 0),
        "blocked": talib_required(dest, manifest),
        "status": "ready",
        "error": None,
        "built_at": _now(),
    }
    if mode == "repo-venv" and source == "none":
        row["mode"] = "main"
        return row

    # A SIGKILL during a previous build can leave its temp requirement file
    # behind; sweep it so the repo directory stays clean.
    for stale in dest.glob(".uv-requirements-*.txt"):
        stale.unlink(missing_ok=True)

    venv = dest / (LEGACY_VENV_DIRNAME if mode == "legacy-fallback"
                   else VENV_DIRNAME)
    conda_python = st.get("conda_python") if source == "environment.yml" else None
    if mode == "legacy-fallback":
        python_spec = python or LEGACY_PYTHON
    elif conda_python:
        python_spec = conda_python
    else:
        python_spec = python or DEFAULT_PYTHON
    run_uv(["uv", "venv", "--no-project", "--clear", "--python", python_spec,
            str(venv)], timeout=timeout, cwd=dest)
    req_file: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".uv-requirements-",
                                         suffix=".txt", dir=str(dest))
        req_file = Path(tmp_name)
        with os.fdopen(fd, "w") as fh:
            fh.write("".join(line + "\n" for line in reqs))
        run_uv(["uv", "pip", "install", "--python",
                str(venv / "bin" / "python"), "-r", str(req_file)],
               timeout=timeout, cwd=dest)
    finally:
        if req_file is not None:
            req_file.unlink(missing_ok=True)
    row["venv"] = _rel(venv, root)
    row["python"] = _venv_python_version(venv) or python_spec
    return row


def _failed_row(label: str, dest: Path, state: dict | None, error: str,
                root: Path, mode: str = "repo-venv") -> dict:
    st = state or {}
    return {"repo": label, "path": _rel(dest, root), "mode": mode,
            "deps_source": st.get("deps_source", "none"), "venv": None,
            "python": None, "sha": st.get("sha", ""),
            "deps_sha256": st.get("deps_sha256", ""),
            "env_key": st.get("env_key", ""),
            "dropped_conda_deps": st.get("dropped_conda_deps", 0),
            "dropped_pip_flags": st.get("dropped_pip_flags", 0),
            "blocked": [], "status": "failed", "error": error,
            "built_at": _now()}


def _walk(root: Path, specs: list[str], strict: bool) -> list[tuple]:
    """Enumerate repos as (dest, manifest|None, label, error|None).

    With specs: exactly those repos, each parsed with
    `harvest_repo.parse_repo_spec` (ValueError on a malformed spec) and
    required to have a manifest.  Without specs: every child directory of
    <root>/data/repos/ holding a manifest.json, sorted.  The owner/name label
    always comes from the manifest's repo record, never from the directory
    name.  In strict mode a corrupt manifest raises json.JSONDecodeError;
    otherwise it is contained per repo as an error entry.
    """
    out: list[tuple] = []

    def add(dest: Path, spec: str | None) -> None:
        manifest_path = dest / "manifest.json"
        if not manifest_path.exists():
            if spec is not None:
                raise ValueError(f"no manifest for {spec} -- "
                                 "run harvest_repo.py first")
            return
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            if strict:
                raise
            out.append((dest, None, spec or dest.name,
                        f"corrupt manifest: {exc}"))
            return
        rec = manifest.get("repo") if isinstance(manifest, dict) else None
        if not isinstance(rec, dict):
            return
        owner, name = rec.get("owner"), rec.get("name")
        if not owner or not name:
            return
        out.append((dest, manifest, f"{owner}/{name}", None))

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


def repo_manifests(root: Path, specs: list[str]) -> list[tuple[Path, dict, str]]:
    """Locate (dest, manifest, label) for every repo to build an env for.

    Raises ValueError for a malformed spec or a missing manifest, and
    json.JSONDecodeError for a corrupt manifest.
    """
    return [(dest, manifest, label)
            for dest, manifest, label, _err in _walk(root, specs, strict=True)]


def _reusable(prev: dict | None, state: dict, root: Path) -> bool:
    """True when the recorded env still matches what the repo needs now.

    The key must match, the dep source must match, and the venv must still
    be on disk (a deleted .venv self-heals, mirroring #23's manifest rule).
    """
    if not prev or prev.get("status") == "failed":
        return False
    if prev.get("env_key") != state["env_key"]:
        return False
    if prev.get("deps_source") != state["deps_source"]:
        return False
    venv = prev.get("venv")
    if venv and not (Path(root) / venv).exists():
        return False
    return True


def main_with(argv: list[str] | None = None, root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="build per-repo execution environments (stage [3c])")
    ap.add_argument("specs", nargs="*", metavar="owner/repo",
                    help="harvested repo(s) (default: every manifest)")
    ap.add_argument("--repos", dest="repos_file", default=None, metavar="FILE",
                    help="file with one owner/repo spec per line")
    ap.add_argument("--root", dest="root_override", default=None,
                    help="project root holding data/repos")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild even when the recorded env key matches")
    ap.add_argument("--legacy-python", dest="legacy_python",
                    default=LEGACY_PYTHON,
                    help=f"reserved for #27's legacy fallback builds; this "
                         f"stage never builds one (default: {LEGACY_PYTHON})")
    ap.add_argument("--jobs", type=int, default=1,
                    help="accepted for CLI symmetry; this stage builds envs "
                         "sequentially and never uses it")
    ap.add_argument("--timeout", type=int, default=UV_TIMEOUT,
                    help=f"seconds per uv call (default: {UV_TIMEOUT})")
    args = ap.parse_args(argv)

    if args.root_override is not None:
        root = Path(args.root_override)
    root = Path(root) if root is not None else ROOT
    # The output file follows the root: rows built under an alternate root
    # must not land in the project's own data/repo_envs.jsonl.
    envs_path = (ENVS if Path(root) == ROOT
                 else Path(root) / "data" / "repo_envs.jsonl")

    specs: list[str] = list(args.specs)
    if args.repos_file:
        try:
            specs.extend(harvest_repo.repos_from_file(Path(args.repos_file)))
        except OSError as exc:
            print(f"error: cannot read {args.repos_file}: {exc}", file=sys.stderr)
            return 1

    try:
        entries = _walk(root, specs, strict=False)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("nothing to build -- run harvest_repo.py, or pass repo specs")
        return 0

    previous = {r.get("repo"): r for r in load(envs_path)}
    rows: list[dict] = []
    failed = 0
    for i, (dest, manifest, label, error) in enumerate(entries, 1):
        verb = "FAILED"
        state: dict | None = None
        if error is not None:
            row = _failed_row(label, dest, None, error, root)
            failed += 1
            print(f"[{i}/{len(entries)}] {'FAILED':<7} {label}: {error}",
                  file=sys.stderr)
        else:
            try:
                state = env_state(dest, manifest)
                prev = previous.get(label)
                if not args.rebuild and _reusable(prev, state, root):
                    row = dict(prev)
                    verb = "reused"
                else:
                    row = build_env(dest, manifest, label, state=state,
                                    timeout=args.timeout, root=root)
                    verb = "built"
            except (RuntimeError, ValueError, OSError) as exc:
                # A repo whose state could not even be read (a declared dep
                # file that is missing, a corrupt environment.yml) must still
                # record the DECLARED dep source, not "none".
                failed_state = state or {
                    "deps_source": deps_source(dest, manifest),
                    "sha": _manifest_sha(manifest)}
                row = _failed_row(label, dest, failed_state, str(exc), root)
                failed += 1
                verb = "FAILED"
                print(f"[{i}/{len(entries)}] FAILED  {label}: {exc}",
                      file=sys.stderr)
        rows.append(row)
        if row.get("status") != "failed":
            print(f"[{i}/{len(entries)}] {verb:<7} {label} ({row.get('mode')})")

    # Replace the prior rows of the repos we just handled, and drop the rows
    # of the same paths (a repo whose manifest was corrupt was labelled by
    # its directory name, so both rows would otherwise linger).
    fresh = {r.get("repo") for r in rows}
    fresh_paths = {r.get("path") for r in rows}
    prior = [r for r in load(envs_path)
             if r.get("repo") not in fresh and r.get("path") not in fresh_paths]
    save(envs_path, prior + rows)

    counts: dict[str, int] = {}
    for r in load(envs_path):
        key = r.get("mode", "unknown")
        counts[key] = counts.get(key, 0) + 1
    print("\nenvs: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"failed={failed}")
    try:
        shown = envs_path.relative_to(root)
    except ValueError:
        shown = envs_path
    print(f"envs -> {shown}")
    return 1 if failed else 0


def main() -> int:
    return main_with()


if __name__ == "__main__":
    raise SystemExit(main())
