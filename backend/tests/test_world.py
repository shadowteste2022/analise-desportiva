import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app import world
from app.config import settings
from app.data import api_football as af
from app.models import dixon_coles


def raw(fid, home, away, kickoff, status="NS", hg=None, ag=None, league=999):
    return {"fixture": {"id": fid, "date": kickoff, "status": {"short": status}},
            "league": {"id": league, "name": "Liga Teste", "country": "Testland", "season": 2026},
            "teams": {"home": {"name": home}, "away": {"name": away}}, "goals": {"home": hg, "away": ag},
            "score": {"halftime": {"home": None if hg is None else 0, "away": None if ag is None else 0}}}


def test_normalize_record_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    games = af.normalize([raw(1, "A", "B", "2026-10-25T15:00:00+00:00", "FT", 2, 1),
                          raw(2, "C", "D", "2026-10-26T15:00:00+00:00"), {"broken": True}])
    assert [g["id"] for g in games] == [1, 2]  # malformed entry dropped
    af.record(games)
    af.record(games)  # same game seen twice is stored once
    df = af.history(999)
    assert len(df) == 1 and df.iloc[0].FTHG == 2 and df.iloc[0].HomeTeam == "A"


def test_quota_keeps_a_reserve(monkeypatch):
    monkeypatch.setattr(af, "state", {"used": 89, "limit": 100, "error": None, "skipped": 0})
    assert af.can_spend()
    af.state["used"] = 90
    assert not af.can_spend()  # 10 of the 100 daily requests are never spent
    af.state["used"] = None
    assert not af.can_spend()  # usage unknown: never spend


def test_world_reports_skip_known_games_and_forecast(league, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "api_football_key", "test")
    kick = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    hist = league.assign(HTHG=league.FTHG // 2, HTAG=league.FTAG // 2)[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]]
    games = [g | {"kickoff": kick} for g in af.normalize([raw(10, "Team01", "Team02", kick),
                                                          raw(11, "Manchester United", "Bournemouth", kick),
                                                          raw(12, "Unknown FC", "Team03", kick)])]
    monkeypatch.setattr(af, "upcoming", lambda: games)
    monkeypatch.setattr(af, "backfill", lambda leagues: None)
    monkeypatch.setattr(af, "history", lambda lg: hist)
    known = [{"date": kick[:10], "home": "Man United", "away": "Bournemouth"}]  # already analysed elsewhere
    out = world.reports(known)
    assert [r["identification"]["home"] for r in out] == ["Team01", "Unknown FC"]
    p = out[0]["predictions"]["probabilities"]
    assert sum(p["1X2"].values()) == pytest.approx(1) and "HTFT" in p and "EG" in p
    assert out[0]["data_status"]["confidence"] == "baixa"  # history is old (synthetic seasons before 2026)
    assert out[1]["predictions"] is None  # a team without history gets no forecast
