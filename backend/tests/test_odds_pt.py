import pandas as pd

from app import analysis
from app.data import odds_api_io

EVENT = {"id": 77, "home": "Sporting CP", "away": "FC Porto", "date": "2026-10-25T20:30:00Z", "bookmakers": {
    "Betano PT": [
        {"name": "ML", "updatedAt": "2026-10-24T10:00:00Z", "odds": [{"home": "2.10", "draw": "3.40", "away": "3.60"}]},
        {"name": "Totals", "updatedAt": "2026-10-24T10:05:00Z", "odds": [
            {"hdp": 2.5, "over": "1.95", "under": "1.85"}, {"hdp": 2.25, "over": "1.80", "under": "2.00"}]},
        {"name": "Both Teams To Score", "updatedAt": "2026-10-24T10:00:00Z", "odds": [{"yes": 1.75, "no": 2.05}]}],
    "Betclic PT": [{"name": "ML", "updatedAt": "2026-10-24T09:00:00Z", "odds": [{"home": 2.05, "draw": 3.5, "away": None}]}]}}


def test_parse_maps_markets():
    odds, last = odds_api_io.parse(EVENT)
    assert odds["1X2"] == {"Betano PT": {"1": 2.10, "X": 3.40, "2": 3.60}}  # Betclic row incomplete -> dropped
    assert odds["OU2.5"]["Betano PT"] == {"Mais": 1.95, "Menos": 1.85}
    assert "OU2.25" not in odds  # Asian line skipped
    assert odds["BTTS"]["Betano PT"] == {"Sim": 1.75, "Não": 2.05}
    assert last == "2026-10-24T10:05:00Z"


def test_find_event_by_names_and_day():
    assert odds_api_io.find_event([EVENT], "Sp Lisbon", "Porto", pd.Timestamp("2026-10-25"))["id"] == 77
    assert odds_api_io.find_event([EVENT], "Sp Lisbon", "Porto", pd.Timestamp("2026-11-02")) is None
    assert odds_api_io.find_event([EVENT], "Benfica", "Braga", pd.Timestamp("2026-10-25")) is None


def test_pt_only_never_uses_unlicensed_books(monkeypatch):
    from app.config import settings
    from app.data import srij
    assert srij.licensed("Betano PT") and srij.licensed("Bwin PT · live") and srij.licensed("Placard")
    assert not srij.licensed("Bet365") and not srij.licensed("Betclic FR · live")
    monkeypatch.setattr(settings, "pt_only", True)
    assert analysis._best({"Bet365": {"1": 2.30}}, "1") == (None, None, False)
    assert analysis._best({"Bet365": {"1": 2.30}, "Bwin PT": {"1": 2.1}}, "1") == (2.1, "Bwin PT", True)
    odds = {"1X2": {"Bet365": {"1": 2.3, "X": 3.4, "2": 3.2}, "Média do mercado": {"1": 2.2, "X": 3.3, "2": 3.1},
                    "Betano PT": {"1": 2.15, "X": 3.3, "2": 3.0}}}
    assert analysis.public_odds(odds) == {"1X2": {"Betano PT": {"1": 2.15, "X": 3.3, "2": 3.0}}}
    rows = analysis.evaluate_markets({"1X2": {"1": 0.45, "X": 0.28, "2": 0.27}},
                                     {"1X2": {k: v for k, v in odds["1X2"].items() if k != "Betano PT"}}, "alta")
    assert all(r["best_odds"] is None and "SRIJ" in r["reasons"][0] for r in rows)


def test_best_prefers_srij_licensed():
    books = {"Bet365": {"1": 2.30}, "Betano PT": {"1": 2.10}, "Betclic PT": {"1": 2.15}}
    assert analysis._best(books, "1") == (2.15, "Betclic PT", True)
    assert analysis._best({"Bet365": {"1": 2.30}}, "1") == (2.30, "Bet365", False)
    # exchanges quote before commission: never the "best odds"
    assert analysis._best({"Bet365": {"1": 2.30}, "Matchbook · live": {"1": 2.6}, "Betfair · live": {"1": 2.7}}, "1") \
        == (2.30, "Bet365", False)
