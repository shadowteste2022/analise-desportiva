from datetime import datetime, timedelta, timezone

import pytest

from app import mma
from app.data import api_sports as aps


def fight(fid, a, b, winner, day):
    return {"id": fid, "kickoff": f"2024-01-{day:02d}T20:00:00+00:00", "status": "FT", "event": "UFC", "category": "Lightweight",
            "home": a, "away": b, "winner": winner}


def test_elo_and_report():
    hist = [fight(1, "A", "B", "1", 1), fight(2, "A", "C", "1", 2), fight(3, "B", "C", "1", 3), fight(4, "A", "B", "1", 4)]
    elo, n, wins = mma.ratings(hist)
    assert elo["A"] > elo["B"] > elo["C"] and n["A"] == 3 and wins["A"] == 3
    assert sum(elo.values()) == pytest.approx(1500 * 3)  # Elo moves points between fighters, never creates them
    kick = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
    r = mma.report({"id": 9, "kickoff": kick, "status": "NS", "event": "UFC 400", "category": "Lightweight",
                    "home": "A", "away": "C", "winner": None}, elo, n, wins)
    p = r["predictions"]["probabilities"]["Vencedor"]
    assert p["1"] > 0.5 and p["1"] + p["2"] == pytest.approx(1) and r["identification"]["sport"] == "MMA"
    new = mma.report({"id": 10, "kickoff": kick, "status": "NS", "event": "", "category": "", "home": "A", "away": "Novo",
                      "winner": None}, elo, n, wins)
    assert new["predictions"] is None  # a fighter without MIN_FIGHTS gets no forecast


def test_normalize_mma_and_afl():
    f = {"id": 1, "timestamp": 1734202800, "status": {"short": "FT"}, "slug": "UFC", "category": "Welterweight",
         "fighters": {"first": {"name": "X", "winner": True}, "second": {"name": "Y", "winner": False}}}
    assert mma.normalize([f])[0]["winner"] == "1"
    afl = {"game": {"id": 7}, "league": {"id": 1, "season": 2024}, "timestamp": "1727497800", "status": {"short": "FT"},
           "teams": {"home": {"name": "Sydney"}, "away": {"name": "Brisbane"}},
           "scores": {"home": {"score": 60}, "away": {"score": 120}}}
    g = aps.normalize([afl], "afl")[0]
    assert (g["league"], g["country"], g["hs"], g["as"]) == ("AFL", "Austrália", 60, 120)
