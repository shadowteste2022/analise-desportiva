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


CATALOG = {
    "basketball_nba": {"group": "Basketball", "title": "NBA", "description": "US Basketball", "has_outrights": False},
    "basketball_wnba": {"group": "Basketball", "title": "WNBA", "description": "US Basketball", "has_outrights": False},
    "soccer_uefa_womens_champions_league": {"group": "Soccer", "title": "UEFA Women's Champions League",
                                            "description": "", "has_outrights": False},
    "soccer_epl": {"group": "Soccer", "title": "EPL", "description": "", "has_outrights": False},
    "golf_masters_tournament_winner": {"group": "Golf", "title": "Masters", "description": "", "has_outrights": True},
    "politics_us_presidential_election_winner": {"group": "Politics", "title": "US", "description": "",
                                                 "has_outrights": False},
    "tennis_atp_paris_masters": {"group": "Tennis", "title": "ATP Paris Masters", "description": "",
                                 "has_outrights": False},
}


def test_sport_keys(monkeypatch):
    monkeypatch.setattr(odds_api, "_refresh_quota", lambda: None)
    monkeypatch.setattr(odds_api, "catalog", CATALOG)
    monkeypatch.setattr(odds_api, "active", list(CATALOG))
    monkeypatch.setattr(settings, "other_sports", "*")
    # all match-market competitions; no outrights, no politics, no football league already from football-data
    assert sports.sport_keys() == ["basketball_nba", "basketball_wnba", "soccer_uefa_womens_champions_league",
                                   "tennis_atp_paris_masters"]
    monkeypatch.setattr(settings, "other_sports", "basketball_nba,tennis_atp_")
    assert sports.sport_keys() == ["basketball_nba", "tennis_atp_paris_masters"]


def test_names_in_portuguese_and_women_marked(monkeypatch):
    monkeypatch.setattr(odds_api, "catalog", CATALOG)
    assert sports.describe("basketball_wnba") == ("Basquetebol", "WNBA (Feminino)")
    assert sports.describe("soccer_uefa_womens_champions_league") == ("Futebol", "UEFA Women's Champions League (Feminino)")
    assert sports.describe("basketball_nba") == ("Basquetebol", "NBA")


def test_reports_list_all_fixtures_and_price_what_fits(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    evs = [{**nba_event((now + pd.Timedelta(hours=h)).isoformat()), "id": f"e{h}"} for h in (-5, 3, 6, 24 * 30)]
    fixtures = [{k: v for k, v in e.items() if k != "bookmakers"} for e in evs]  # /events has no odds
    monkeypatch.setattr(settings, "odds_api_key", "test")
    monkeypatch.setattr(sports, "sport_keys", lambda: ["basketball_nba"])
    monkeypatch.setattr(odds_api, "events", lambda k: fixtures)
    monkeypatch.setattr(odds_api, "fetch_sport", lambda k: [evs[1]])  # budget only priced the e3 game
    out = sports.reports(log=False)
    assert len(out) == 2  # past and +30 days dropped
    with_odds = [r for r in out if r["markets"]]
    assert len(with_odds) == 1 and out[1]["recommendation"]["kind"] == "sem_dados"
