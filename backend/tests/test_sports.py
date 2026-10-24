import pandas as pd
import pytest

from app import markets as mk, sports
from app.config import settings
from app.data import odds_api


def book(title, home, away, h2h, total=None):
    markets = [{"key": "h2h", "outcomes": [{"name": n, "price": p} for n, p in zip([home, away, "Draw"], h2h)]}]
    if total:
        point, over, under = total
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "price": over, "point": point},
                                                      {"name": "Under", "price": under, "point": point}]})
    return {"title": title, "last_update": "2026-10-24T10:00:00Z", "markets": markets}


def event(start, books, home="Lakers", away="Celtics"):
    return {"home_team": home, "away_team": away, "commence_time": start, "bookmakers": books}


def nba_event(start="2026-10-25T00:30:00Z"):
    return event(start, [book("A", "Lakers", "Celtics", [1.80, 2.10], (220.5, 1.90, 1.90)),
                         book("B", "Lakers", "Celtics", [1.85, 2.00], (220.5, 1.95, 1.87)),
                         book("C", "Lakers", "Celtics", [1.70, 2.25])])


def test_event_odds_and_consensus():
    odds, last = sports.event_odds(nba_event())
    assert set(odds) == {"Vencedor", "OU220.5"} and last == "2026-10-24T10:00:00Z"
    assert odds["Vencedor"]["A"] == {"1": 1.80, "2": 2.10}  # no draw in basketball
    probs = sports.consensus(odds)
    assert sum(probs["Vencedor"].values()) == pytest.approx(1)
    assert odds["Vencedor"]["Média do mercado"]["1"] == pytest.approx((1.80 + 1.85 + 1.70) / 3)
    three_way = sports.event_odds(event("2026-10-25T00:30:00Z", [book("A", "X1", "X2", [2.0, 3.5, 3.4])],
                                        "X1", "X2"))[0]
    assert list(three_way["Vencedor"]["A"]) == ["1", "X", "2"]


def test_report_never_claims_value(monkeypatch):
    r = sports.analyze(nba_event(), "basketball_nba", log=False)
    ident = r["identification"]
    assert ident["sport"] == "Basquetebol" and ident["competition"] == "NBA"
    assert ident["date"] == "2026-10-25" and ident["time"] == "01:30"  # 00:30 UTC = 01:30 in Lisbon (summer time)
    assert r["statistics"] is None and r["predictions"]["matrix"] is None
    assert r["data_status"]["confidence"] == "baixa"  # 3 bookmakers
    assert all(x["category"] != mk.VALOR for x in r["markets"])
    best = max(r["markets"], key=lambda x: x["ev"] or -9)
    assert best["best_odds"] == 2.25 and "consenso" in r["conclusion"]["uncertainties"][0]


def test_describe_tennis():
    assert sports.describe("tennis_atp_paris_masters") == ("Ténis", "ATP Paris Masters")
    assert sports.describe("icehockey_nhl") == ("Hóquei no gelo", "NHL")


def test_sport_keys_expand_active_tournaments(monkeypatch):
    monkeypatch.setattr(odds_api, "_refresh_quota", lambda: None)
    monkeypatch.setattr(odds_api, "active", ["tennis_atp_paris_masters", "tennis_wta_finals", "basketball_nba"])
    monkeypatch.setattr(settings, "other_sports", "basketball_nba,tennis_atp_")
    assert sports.sport_keys() == ["basketball_nba", "tennis_atp_paris_masters"]


def test_reports_keep_only_upcoming(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    events = [nba_event((now - pd.Timedelta(hours=5)).isoformat()), nba_event((now + pd.Timedelta(hours=3)).isoformat()),
              nba_event((now + pd.Timedelta(days=30)).isoformat())]
    monkeypatch.setattr(settings, "odds_api_key", "test")
    monkeypatch.setattr(sports, "sport_keys", lambda: ["basketball_nba"])
    monkeypatch.setattr(odds_api, "fetch_sport", lambda k: events)
    out = sports.reports(log=False)
    assert len(out) == 1
