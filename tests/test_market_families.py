import pytest
from pathlib import Path

def test_load_market_families():
    import market_families

    families = market_families.load_families()
    assert "equity_large_cap" in families
    assert families["equity_large_cap"]["benchmark"] == "yahoo:SPY"
    assert len(families["equity_large_cap"]["peers"]) >= 3


def test_resolve_family_for_ticker():
    import market_families

    # Bare ticker or catalog ID resolution
    assert market_families.find_family_for_ticker("SPY") == "equity_large_cap"
    assert market_families.find_family_for_ticker("yahoo:SPY") == "equity_large_cap"
    assert market_families.find_family_for_ticker("yahoo:^NSEI") == "equity_large_cap"
    assert market_families.find_family_for_ticker("unknown:NONEXISTENT") is None


def test_get_peers_for_ticker():
    import market_families

    peers = market_families.get_peers_for_ticker("SPY")
    peer_ids = [p["id"] for p in peers]
    assert "yahoo:SPY" in peer_ids
    assert "yahoo:^NSEI" in peer_ids
