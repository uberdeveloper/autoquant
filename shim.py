"""Stage [3d] SHIM -- the offline data-shim seam for external execution (#26).

External strategy repos call ``yfinance`` / ``pandas_datareader`` and would
otherwise download prices live: rate limits, flakiness, non-deterministic
runs, and a backtest that does not use the data the project actually trusts.
This stage makes them run against **our** price cache instead.

How it works
------------

``external_shim/`` holds stub ``yfinance`` and ``pandas_datareader``
packages.  They are not installed -- they are injected by putting that
directory on ``PYTHONPATH``, so ``import yfinance`` inside the external
process resolves to the stub.  The stubs serve the project's own catalog
cache through ``catalog.load`` (the same normalization, fallback and caching
the project already trusts) and never phone out.

``shim_env()`` builds the environment mapping a launcher should use:

    env = shim_env(catalog_root=<cache dir>)   # -> {"PYTHONPATH": ..., ...}
    subprocess.run(argv, env={**os.environ, **env})

``PYTHONPATH`` is ``<shim root>[:extra path][:<existing PYTHONPATH>]`` -- the
shim first, so the stub wins over site-packages.

Guarantees
----------

* **Offline by default.**  ``AUTOQUANT_SHIM_OFFLINE`` (default ``1``) makes a
  cache miss raise ``DataNotAvailable`` (catchable, names the symbol) instead
  of reaching a provider.  Set it to ``0`` only for local debugging.
* **Injectable cache root.**  ``AUTOQUANT_CATALOG_DIR`` (default
  ``<project root>/data/catalog``) points the shim at any cache directory --
  a run-copy, a warmed cache, or a test fixture.
* **Isolation.**  This stage never writes into ``data/repos/`` and never
  mutates an original clone.  Running external code in a *copy* of the repo
  is #27's job (``run_external.py``); #27 calls ``shim_env()`` only for
  artifacts whose triage verdict is ``RUN_WITH_OUR_DATA`` -- repos that ship
  their own complete data (``RUN_BUNDLED_DATA``) run in a copy with **no**
  shim injected.
* **Hard offline (optional).**  ``unshare -n`` gives a kernel-level network
  namespace with no interfaces.  It is feature-flagged
  (``AUTOQUANT_SHIM_HARD_OFFLINE=1``) and only usable where the kernel
  allows it (it is *not* available inside restricted sandboxes); the
  deterministic guarantee is the shim's own offline behaviour.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHIM_DIR = ROOT / "external_shim"
DEFAULT_CATALOG_DIR = ROOT / "data" / "catalog"

CATALOG_DIR_ENV = "AUTOQUANT_CATALOG_DIR"
OFFLINE_ENV = "AUTOQUANT_SHIM_OFFLINE"
HARD_OFFLINE_ENV = "AUTOQUANT_SHIM_HARD_OFFLINE"

UNSHARE = ("unshare", "-n")


def _load_errors() -> type[RuntimeError]:
    """The shim's error type, however this module was imported."""
    try:
        from external_shim._core import DataNotAvailable
        return DataNotAvailable
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "_autoquant_shim_errors", SHIM_DIR / "_core.py")
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DataNotAvailable


DataNotAvailable = _load_errors()

__all__ = ["shim_root", "shim_env", "hard_offline_prefix", "unshare_available",
           "DataNotAvailable", "CATALOG_DIR_ENV", "OFFLINE_ENV",
           "HARD_OFFLINE_ENV"]


def shim_root() -> Path:
    """The ``external_shim/`` directory to put on ``PYTHONPATH``."""
    return SHIM_DIR


def shim_env(catalog_root=None, extra_path=None, existing=None) -> dict:
    """The environment mapping a launcher should apply for a shimmed run.

    ``catalog_root`` -- the cache directory the shim serves from
    (default: ``<project root>/data/catalog``).
    ``extra_path``   -- an extra ``PYTHONPATH`` entry (e.g. the external
    repo's own dependency directory), placed after the shim.
    ``existing``     -- the ``PYTHONPATH`` to append to (default: the current
    process's); pass ``""`` to build the value from scratch.

    The returned mapping is meant to be merged into ``os.environ`` (or
    passed as the ``env=`` argument of a subprocess).
    """
    parts = [str(SHIM_DIR)]
    if extra_path:
        parts.append(str(extra_path))
    base = existing if existing is not None else os.environ.get("PYTHONPATH", "")
    if base:
        parts.extend(p for p in str(base).split(os.pathsep) if p)
    root = Path(catalog_root) if catalog_root else DEFAULT_CATALOG_DIR
    return {
        "PYTHONPATH": os.pathsep.join(parts),
        CATALOG_DIR_ENV: str(root),
        OFFLINE_ENV: os.environ.get(OFFLINE_ENV, "1"),
    }


def unshare_available() -> bool:
    """True when ``unshare -n`` actually works here.

    Probed, not assumed: inside a restricted sandbox the call fails with
    "Operation not permitted", and the hard-offline path must then stay off.
    """
    exe = shutil.which(UNSHARE[0])
    if not exe:
        return False
    try:
        result = subprocess.run([exe, "-n", "true"], capture_output=True,
                                timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def hard_offline_prefix() -> list[str] | None:
    """The command prefix that removes the network, or None when unusable.

    Feature-flagged: it only applies when ``AUTOQUANT_SHIM_HARD_OFFLINE=1``
    *and* the kernel allows an unprivileged network namespace.  The
    deterministic offline guarantee is the shim's own behaviour
    (``AUTOQUANT_SHIM_OFFLINE``), not this wrapper.
    """
    flag = os.environ.get(HARD_OFFLINE_ENV, "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    if not unshare_available():
        return None
    return list(UNSHARE)


def main() -> int:
    """Print the shim environment -- a smoke check, never touches the network."""
    env = shim_env()
    for key in ("PYTHONPATH", CATALOG_DIR_ENV, OFFLINE_ENV):
        print(f"{key}={env[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
