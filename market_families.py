"""Market family registry and peer resolution for cross-market robustness testing."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_FAMILIES_FILE = ROOT / "market_families.yaml"


def load_families(path: Path | str | None = None) -> dict[str, Any]:
    """Load market families from a YAML rulebook file."""
    p = Path(path) if path else DEFAULT_FAMILIES_FILE
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text())
    return data.get("families", {}) if isinstance(data, dict) else {}


def _normalize_ticker(ticker: str) -> str:
    """Normalize a ticker or catalog id for matching."""
    t = ticker.strip()
    if not ":" in t:
        return f"yahoo:{t}"
    return t


def find_family_for_ticker(ticker: str, path: Path | str | None = None) -> str | None:
    """Find the family name that contains the given ticker or catalog ID."""
    families = load_families(path)
    norm = _normalize_ticker(ticker)
    for name, fam in families.items():
        if fam.get("benchmark") == norm or _normalize_ticker(fam.get("benchmark", "")) == norm:
            return name
        for peer in fam.get("peers", []):
            pid = peer.get("id") if isinstance(peer, dict) else str(peer)
            if pid == norm or _normalize_ticker(pid) == norm:
                return name
    return None


def get_peers_for_ticker(ticker: str, path: Path | str | None = None) -> list[dict[str, str]]:
    """Return the list of peers for the family matching the ticker."""
    fam_name = find_family_for_ticker(ticker, path)
    if not fam_name:
        return []
    families = load_families(path)
    return families.get(fam_name, {}).get("peers", [])
