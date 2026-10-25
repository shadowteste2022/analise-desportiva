from datetime import datetime, timezone

import pandas as pd
import pytest

from app import analysis, markets as mk, sports
from app.config import settings
from app.data import oddspapi


def outcome(price, active=True):
    return {"players": {"0": {"active": active, "price": price, "changedAt": "2026-10-25T11:30:00Z"}}}


def raw_fixture(fid, p1, p2, start, book="betano.pt", prices=(2.1, 3.4, 3.5), suspended=False, active_market=True):
    return {"fixtureId": fid, "participant1Id": p1, "participant2Id": p2, "tournamentId": 239, "startTime": start,
            "updatedAt": "2026-10-24T20:00:00Z", "bookmakerOdds": {book: {
                "bookmakerIsActive": True, "suspended": suspended, "markets": {
                    "101": {"marketActive": active_market, "outcomes": {"101": outcome(prices[0]), "102": outcome(prices[1]),
                                                                        "103": outcome(prices[2])}},
                    "1010": {"marketActive": True, "outcomes": {"1010": outcome(1.9), "1011": outcome(1.95)}},
                    "104": {"marketActive": True, "outcomes": {"104": outcome(1.8), "105": outcome(1.0)}},  # 1.0 invalid
                    "99999": {"marketActive": True, "outcomes": {}}}}}}


def test_parse_keeps_active_used_markets():
    rows = oddspapi.parse([raw_fixture("a", 1, 2, "2026-10-25T15:00:00Z"),
                           raw_fixture("b", 3, 4, "2026-10-25T15:00:00Z", suspended=True),
                           raw_fixture("c", 5, 6, "2026-10-25T15:00:00Z", active_market=False)], "betano.pt")
    assert [r["id"] for r in rows] == ["a", "c"]  # suspended game dropped
    a = rows[0]["odds"]
    assert a["1X2"] == {"Betano PT": {"1": 2.1, "X": 3.4, "2": 3.5}}
    assert a["OU2.5"] == {"Betano PT": {"Mais": 1.9, "Menos": 1.95}}
    assert "BTTS" not in a  # one outcome at 1.0 is not a price
    assert rows[0]["updated"] == "2026-10-25T11:30:00Z"  # odds change time, not the fixture's updatedAt
    assert "1X2" in rows[1]["odds"]  # marketActive=false alone does not drop a priced market


def test_pacing_uses_the_free_account_endpoint(monkeypatch):
    monkeypatch.setattr(oddspapi, "state", {"used": 100, "limit": 250, "reset_day": 24, "error": None,
                                            "skipped": [], "fetched": {}})
    monkeypatch.setattr(oddspapi, "_refresh_account", lambda: None)
    mid = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)  # ~half the period that started on 24 Oct
    assert oddspapi.can_spend(1, mid)
    oddspapi.state["used"] = 130
    assert not oddspapi.can_spend(1, mid)
    oddspapi.state.update(used=240)  # one more would leave 9, below the reserve of 10
    assert not oddspapi.can_spend(1, datetime(2026, 11, 23, tzinfo=timezone.utc))  # reserve of 10 kept
    oddspapi.state["used"] = None
    assert not oddspapi.can_spend(1, mid)  # usage unknown: never spend


@pytest.fixture
def pt_games(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    games = [{"id": "x1", "tournament": 239, "start": (now + pd.Timedelta(hours=5)).isoformat(),
              "kickoff": now + pd.Timedelta(hours=5), "home": "FC Porto", "away": "Sporting CP",
              "competition": "Portugal — Liga Portugal 2", "updated": "2026-10-24T20:00:00Z",
              "odds": {"1X2": {"Betano PT": {"1": 2.2, "X": 3.3, "2": 3.1}, "Bwin PT": {"1": 2.3, "X": 3.25, "2": 3.0}}}}]
    monkeypatch.setattr(oddspapi, "_fixtures_cache", games)
    monkeypatch.setattr(oddspapi, "_consumed", set())
    monkeypatch.setattr(settings, "oddspapi_key", "test")
    return games


def test_match_and_unmatched(pt_games):
    kick = pt_games[0]["kickoff"]
    odds, _ = oddspapi.match_odds("Porto", "Sp Lisbon", kick)  # football-data names vs OddsPapi names
    assert set(odds["1X2"]) == {"Betano PT", "Bwin PT"}
    assert oddspapi.unmatched() == []  # consumed by the game it was merged into
    oddspapi._consumed.clear()
    assert len(oddspapi.unmatched()) == 1


def test_portuguese_only_game_report(pt_games, monkeypatch):
    monkeypatch.setattr(settings, "pt_only", True)
    r = sports.from_oddspapi(pt_games[0], log=False)
    assert r["identification"]["competition"] == "Portugal — Liga Portugal 2" and r["identification"]["sport"] == "Futebol"
    assert r["predictions"]["probabilities"]["Vencedor"]  # consensus of the two Portuguese books
    assert all(m["bookmaker"] in ("Betano PT", "Bwin PT") for m in r["markets"] if m["bookmaker"])
    assert set(r["odds"]["Vencedor"]) == {"Betano PT", "Bwin PT"}  # public odds: licensed books only
    assert r["recommendation"]["kind"] != "valor"  # no backtest: never claimed as validated value
    # the shared cache is not altered by the analysis (consensus adds an average book to its own copy)
    assert set(pt_games[0]["odds"]["1X2"]) == {"Betano PT", "Bwin PT"}
