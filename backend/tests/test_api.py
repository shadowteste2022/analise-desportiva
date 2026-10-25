import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import analysis
from app.data import football_data as fd, odds_api
from app.main import app


@pytest.fixture
def client(league, monkeypatch):
    played = league[league.Date < league.Date.max() - pd.Timedelta(days=6)]
    upcoming = league[league.Date >= league.Date.max() - pd.Timedelta(days=6)].drop(columns=["FTHG", "FTAG"])
    monkeypatch.setattr(fd, "load_results", lambda lg, seasons=None, **k: (played, {
        "source": "synthetic", "matches": len(played), "last_result": played.Date.max().date().isoformat(),
        "stale": False, "files": [{"fetched_at": "2026-01-01T00:00:00+00:00", "stale": False}]}))
    monkeypatch.setattr(fd, "load_fixtures", lambda: (upcoming.reset_index(drop=True), {
        "source": "synthetic", "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(), "stale": False}))
    analysis._models.clear()
    return TestClient(app)


def test_fixtures_and_report(client):
    data = client.get("/api/fixtures").json()
    assert data["fixtures"]
    fx = data["fixtures"][0]
    r = client.get("/api/match", params={"key": fx["key"]}).json()
    for section in ("identification", "data_status", "statistics", "predictions", "markets", "conclusion"):
        assert section in r
    by = {(m["market"], m["selection"]): m for m in r["markets"]}
    assert by[("BTTS", "Sim")]["category"] == "SEM DADOS SUFICIENTES"  # no BTTS odds in the feed
    one = by[("1X2", "1")]
    assert one["best_odds"] and one["ev"] == pytest.approx(one["model_prob"] * one["best_odds"] - 1, abs=1e-3)
    assert "Lesões" in " ".join(r["conclusion"]["uncertainties"])
    hist = client.get("/api/history").json()
    assert any(h["match_key"] == fx["key"] for h in hist)
    assert ("DC", "1X") in by and by[("DC", "1X")]["min_odds"] and r["recommendation"]["verdict"]
    assert "markets" in client.get("/api/performance").json()


def test_static_export(client, tmp_path):
    from app import export
    res = export.export(tmp_path / "site")
    site = tmp_path / "site"
    fixtures = __import__("json").loads((site / "data" / "fixtures.json").read_text(encoding="utf-8"))
    assert res["fixtures"] == len(fixtures["fixtures"]) > 0 and fixtures["generated_at"]
    key = fixtures["fixtures"][0]["key"]
    assert (site / "data" / "match" / f"{export.slug(key)}.json").exists()
    assert export.slug("P1|2026-10-24|Nott'm Forest|Man City") == "P1_2026_10_24_Nott_m_Forest_Man_City"  # = JS slug
    assert "window.STATIC = true" in (site / "index.html").read_text(encoding="utf-8")
    assert (site / "data" / "performance.json").exists()


def test_report_with_live_lines_and_blend(client, monkeypatch):
    """A fixture with live odds for a goal line the CSVs lack (OU3.5) must blend with the O/U 2.5 weights."""
    from app.models import blend
    monkeypatch.setattr(blend, "load", lambda: {"weights": {"1X2": {"a": 0.0, "b": 1.04}, "OU2.5": {"a": 0.0, "b": 1.04}}})
    live = {"OU3.5": {"A · live": {"Mais": 3.0, "Menos": 1.40}, "B · live": {"Mais": 2.8, "Menos": 1.45}}}
    real = analysis._fixture_odds
    monkeypatch.setattr(analysis, "_fixture_odds", lambda row, lg, meta: (lambda o, s: ({**o, **live}, s))(*real(row, lg, meta)))
    fx = client.get("/api/fixtures").json()["fixtures"][0]
    r = client.get("/api/match", params={"key": fx["key"]}).json()
    assert "OU3.5" in r["predictions"]["blend"]["markets"]
    assert r["predictions"]["blend"]["weights"]["OU3.5"] == {"a": 0.0, "b": 1.04}
    ou35 = next(m for m in r["markets"] if m["market"] == "OU3.5" and m["selection"] == "Menos")
    assert ou35["market_prob_no_vig"] is not None and abs(ou35["model_prob"] - ou35["market_prob_no_vig"]) < 0.05


def test_stale_odds_are_dropped(league):
    row = league.iloc[-1]
    odds, src = analysis._fixture_odds(row, "T0", {"fetched_at": "2020-01-01T00:00:00+00:00", "stale": True})
    assert odds == {} and "descartadas" in src[0]["note"]


def test_value_requires_validated_model():
    rows = [{"category": "VALOR POTENCIAL", "reasons": []}, {"category": "SEM VALOR IDENTIFICADO", "reasons": []}]
    analysis._apply_validation(rows, analysis.validation("NOPE"))  # no backtest on disk
    assert [r["category"] for r in rows] == ["VALOR INCONCLUSIVO", "SEM VALOR IDENTIFICADO"]
    assert "não validado" in rows[0]["reasons"][0]


def test_unknown_match_and_league(client):
    assert client.get("/api/match", params={"key": "x"}).status_code == 404
    assert client.get("/api/fixtures", params={"league": "E0"}).status_code == 404
    assert client.get("/api/fixtures", params={"league": "<script>"}).status_code == 422


def test_admin_requires_token(client):
    assert client.post("/api/admin/refresh").status_code == 401
    assert client.post("/api/admin/refresh", headers={"X-Admin-Token": "test-token"}).status_code == 200


def test_odds_api_matching():
    events = [{"home_team": "Sporting Lisbon", "away_team": "FC Porto", "bookmakers": [{
        "title": "Pinnacle", "last_update": "2026-10-23T10:00:00Z", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Sporting Lisbon", "price": 2.1}, {"name": "FC Porto", "price": 3.5},
                                        {"name": "Draw", "price": 3.3}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5},
                                           {"name": "Under", "price": 1.95, "point": 2.5}]}]}]}]
    odds, last = odds_api.match_odds(events, "Sp Lisbon", "Porto")
    assert odds["1X2"]["Pinnacle"] == {"1": 2.1, "X": 3.3, "2": 3.5}
    assert odds["OU2.5"]["Pinnacle"]["Mais"] == 1.9 and last
    assert odds_api.match_odds(events, "Benfica", "Braga") == ({}, None)


def test_played_fixtures_are_hidden():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-10-23", "2026-10-24", "2026-10-24", "2026-10-25"]),
                       "Time": ["20:00", "12:30", "17:30", None]})
    now = pd.Timestamp("2026-10-24 15:00", tz=fd.LOCAL_TZ)
    kept = fd.upcoming(df, now)
    # yesterday's game is gone, 12:30 kicked off 2.5 h ago, 17:30 is still to come, no time -> kept
    assert kept.Time.tolist()[:1] == ["17:30"] and len(kept) == 2


def test_clean_rejects_bad_rows():
    raw = pd.DataFrame({"Div": ["P1", "P1"], "Date": ["07/08/2026", "bad"], "HomeTeam": ["A", "B"],
                        "AwayTeam": ["C", "D"], "FTHG": ["1", "x"], "B365H": ["0.5", "2.0"]})
    df = fd.clean(raw)
    assert len(df) == 1 and df.Date.iloc[0] == pd.Timestamp("2026-08-07") and pd.isna(df.B365H.iloc[0])
    assert fd.season_codes(pd.Timestamp("2026-10-23").date(), 3) == ["2425", "2526", "2627"]


def test_predictions_without_odds_are_logged_once():
    from app import db
    row = {"match_key": "T0|2030-01-01|A|B", "league": "T0", "match_date": "2030-01-01", "home": "A", "away": "B",
           "market": "DC", "selection": "1X", "model_prob": 0.7, "odds": None, "ev": None,
           "category": "SEM DADOS SUFICIENTES", "confidence": "alta", "model_version": "t"}
    db.log_predictions([row]); db.log_predictions([row])  # hourly re-run, still no price
    assert sum(r["match_key"] == row["match_key"] for r in db.history(5000)) == 1
