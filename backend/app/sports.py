"""Other sports (basketball, ice hockey, baseball, tennis) from The Odds API, analysed by market consensus.

There is no free results-and-odds history like football-data for these sports, so there is no statistical model of
our own yet: the probability of each outcome is the bookmakers' consensus without margin (Shin on the average
price). The football backtest showed our model gets zero weight next to that consensus anyway. "Value" here is
therefore line shopping, a best price above the consensus, classified with the same rules as football; with no
backtest, nothing is ever labelled VALOR POTENCIAL.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import analysis, db, markets as mk
from app.config import settings
from app.data import odds_api
from app.data.football_data import AGGREGATES, LOCAL_TZ

MODEL_VERSION = "consenso-mercado-v1"
NAMES = {"basketball_nba": ("Basquetebol", "NBA"), "basketball_euroleague": ("Basquetebol", "Euroleague"),
         "icehockey_nhl": ("Hóquei no gelo", "NHL"), "baseball_mlb": ("Basebol", "MLB"),
         "americanfootball_nfl": ("Futebol americano", "NFL")}
NOTES = ["Probabilidades = consenso das casas sem margem (Shin); ainda não há modelo estatístico próprio para esta "
         "modalidade.",
         "Sem backtest: nenhuma seleção é classificada como VALOR POTENCIAL.",
         "Odds de casas internacionais (The Odds API): confirme se a casa tem licença do SRIJ antes de apostar a "
         "partir de Portugal."]
HORIZON_DAYS = 7


def is_other_sport(key: str) -> bool:
    """Match keys start with the competition: Odds API sport keys contain "_" (basketball_nba),
    football-data league codes never do (P1, SC0)."""
    return "_" in key.split("|")[0]


def describe(key: str) -> tuple[str, str]:
    if key in NAMES:
        return NAMES[key]
    if key.startswith("tennis_"):  # e.g. tennis_atp_paris_masters -> ATP Paris Masters
        tour, *rest = key.split("_")[1:]
        return "Ténis", " ".join([tour.upper()] + [w.capitalize() for w in rest])
    return key.split("_")[0].capitalize(), key


def sport_keys() -> list[str]:
    """Configured sports; entries ending in "_" expand to every tournament the API reports as active."""
    odds_api._refresh_quota()  # also fills odds_api.active, at no credit cost
    keys = []
    for k in (x.strip() for x in settings.other_sports.split(",") if x.strip()):
        keys += sorted(a for a in odds_api.active if a.startswith(k)) if k.endswith("_") else [k]
    return keys


def event_odds(ev: dict) -> tuple[dict, str | None]:
    """{market: {bookmaker: {selection: odds}}} with markets named like football ("Vencedor", "OU<line>")."""
    out: dict = {}
    last = None
    home, away = ev["home_team"], ev["away_team"]
    for bk in ev.get("bookmakers", []):
        last = max(last or "", bk.get("last_update") or "")
        for m in bk.get("markets", []):
            prices = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
            if m.get("key") == "h2h":
                key = "Vencedor"
                sel = {"1": prices.get(home), "X": prices["Draw"], "2": prices.get(away)} if "Draw" in prices \
                    else {"1": prices.get(home), "2": prices.get(away)}
            elif m.get("key") == "totals":
                points = {o.get("point") for o in m.get("outcomes", [])}
                if len(points) != 1:
                    continue
                key, sel = f"OU{points.pop()}", {"Mais": prices.get("Over"), "Menos": prices.get("Under")}
            else:
                continue
            if all(isinstance(v, (int, float)) and v > 1 for v in sel.values()):
                out.setdefault(key, {})[bk["title"]] = {s: float(v) for s, v in sel.items()}
    return out, last or None


def consensus(odds: dict) -> dict:
    """Adds the average-price book to each market with 2+ bookmakers and returns its no-vig probabilities."""
    probs = {}
    for market, books in odds.items():
        if len(books) < 2:
            continue
        sels = list(next(iter(books.values())))
        avg = {s: float(np.mean([b[s] for b in books.values()])) for s in sels}
        books[AGGREGATES["Avg"]] = avg
        probs[market] = mk.devig(avg)[0]
    return probs


def confidence(n_books: int) -> str | None:
    """How broad the consensus is; separate from the probability of the outcome."""
    return "alta" if n_books >= 8 else "média" if n_books >= 4 else "baixa" if n_books >= 2 else None


def analyze(ev: dict, sport_key: str, log: bool = True) -> dict:
    sport, competition = describe(sport_key)
    home, away = ev["home_team"], ev["away_team"]
    kick = pd.Timestamp(ev["commence_time"]).tz_convert(LOCAL_TZ)
    key = f"{sport_key}|{kick.date().isoformat()}|{home}|{away}"
    odds, last = event_odds(ev)
    n_books = len(odds.get("Vencedor", {}))
    probs = consensus(odds)
    conf = confidence(n_books)
    rows = analysis.evaluate_markets(probs, odds, conf) if probs else []
    validation = analysis.validation(sport_key)
    analysis._apply_validation(rows, validation)
    report = {
        "key": key,
        "identification": {"sport": sport, "league": sport_key, "competition": competition, "home": home,
                           "away": away, "date": kick.date().isoformat(), "time": kick.strftime("%H:%M"),
                           "kickoff": kick.isoformat(),
                           "venue": "—" if sport == "Ténis" else f"Casa de {home} (pavilhão/estádio não indicado)"},
        "data_status": {"odds": [{"source": "the-odds-api.com", "fetched_at": last,
                                  "note": f"odds de {n_books} casas"}],
                        "bookmakers": n_books, "confidence": conf, "model_validation": validation,
                        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "statistics": None,
        "predictions": {"probabilities": probs, "model_probabilities": None, "blend": None, "expected_goals": None,
                        "matrix": None, "top_scores": [],
                        "method": f"Consenso do mercado sem margem (Shin) — média de {n_books} casas"}
        if probs else None,
        "markets": rows, "odds": odds, "conclusion": analysis._conclusion(rows, conf, list(NOTES)),
        "recommendation": analysis.recommend(rows, home, away, sport)}
    if log and rows:
        db.log_predictions([{"match_key": key, "league": sport_key, "match_date": kick.date().isoformat(),
                             "home": home, "away": away, "market": r["market"], "selection": r["selection"],
                             "model_prob": r["model_prob"], "odds": r["best_odds"], "ev": r["ev"],
                             "category": r["category"], "confidence": conf, "model_version": MODEL_VERSION}
                            for r in rows])
        db.snapshot_odds(key, {m: {b: s for b, s in books.items() if b != AGGREGATES["Avg"]}
                               for m, books in odds.items()}, last or db.now())
    return report


def reports(log: bool = True) -> list[dict]:
    """Upcoming events (from 2 h ago to HORIZON_DAYS ahead) for every configured sport, soonest first."""
    if not odds_api.enabled():
        return []
    now = pd.Timestamp.now(tz=timezone.utc)
    out = []
    for sk in sport_keys():
        for ev in odds_api.fetch_sport(sk):
            try:
                start = pd.Timestamp(ev["commence_time"])
                if not (now - pd.Timedelta(hours=2) < start < now + pd.Timedelta(days=HORIZON_DAYS)):
                    continue
                out.append(analyze(ev, sk, log))
            except (KeyError, ValueError, TypeError):  # one malformed event must not hide the rest
                continue
    return sorted(out, key=lambda r: (r["identification"]["date"], r["identification"]["time"]))


def find(key: str) -> dict | None:
    return next((r for r in reports(log=False) if r["key"] == key), None)
