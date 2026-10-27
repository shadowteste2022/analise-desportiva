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
from app.data import odds_api, oddspapi
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
HORIZON_DAYS = 8  # covers the site's rolling week (today and the next 6 days)


def is_other_sport(key: str) -> bool:
    """Match keys start with the competition: Odds API sport keys contain "_" (basketball_nba),
    football-data league codes never do (P1, SC0)."""
    return "_" in key.split("|")[0]


GROUP_PT = {"Soccer": "Futebol", "Basketball": "Basquetebol", "Ice Hockey": "Hóquei no gelo", "Baseball": "Basebol",
            "Tennis": "Ténis", "American Football": "Futebol americano", "Aussie Rules": "Futebol australiano",
            "Rugby League": "Râguebi (league)", "Rugby Union": "Râguebi", "Cricket": "Críquete",
            "Mixed Martial Arts": "MMA", "Boxing": "Boxe", "Handball": "Andebol", "Volleyball": "Voleibol",
            "Lacrosse": "Lacrosse", "Golf": "Golfe", "Darts": "Dardos", "Snooker": "Snooker", "Table Tennis": "Ténis de mesa"}
EXCLUDED_GROUPS = {"Politics"}  # not sport
INDIVIDUAL = {"Ténis", "MMA", "Boxe", "Dardos", "Snooker", "Ténis de mesa", "Golfe"}  # no home venue


def is_women(*texts: str) -> bool:
    t = " ".join(texts).lower()
    return "women" in t or "wnba" in t or "wta" in t or "female" in t or "ladies" in t


def describe(key: str) -> tuple[str, str]:
    """(sport, competition) in Portuguese, women's competitions marked."""
    info = odds_api.catalog.get(key)
    if info:
        sport = GROUP_PT.get(info.get("group", ""), info.get("group", ""))
        comp = info.get("title") or key
    elif key in NAMES:
        sport, comp = NAMES[key]
    elif key.startswith("tennis_"):  # e.g. tennis_atp_paris_masters -> ATP Paris Masters
        tour, *rest = key.split("_")[1:]
        sport, comp = "Ténis", " ".join([tour.upper()] + [w.capitalize() for w in rest])
    else:
        sport, comp = key.split("_")[0].capitalize(), key
    if is_women(key, comp, (info or {}).get("description", "")) and "femin" not in comp.lower():
        comp += " (Feminino)"
    return sport, comp


def sport_keys() -> list[str]:
    """Configured sports. "*" = every in-season competition with match markets (outright-only ones, like golf
    winners, are left out) except the football leagues already covered by football-data; entries ending in "_"
    expand to every active key with that prefix."""
    odds_api._refresh_quota()  # also fills odds_api.active/catalog, at no credit cost
    covered = set(odds_api.SPORTS.values())
    keys = []
    for k in (x.strip() for x in settings.other_sports.split(",") if x.strip()):
        if k == "*":
            keys += sorted(s for s, info in odds_api.catalog.items()
                           if not info.get("has_outrights") and info.get("group") not in EXCLUDED_GROUPS
                           and s not in covered)
        else:
            keys += sorted(a for a in odds_api.active if a.startswith(k)) if k.endswith("_") else [k]
    return list(dict.fromkeys(keys))


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
                if len(points) != 1 or not isinstance(next(iter(points)), (int, float)) or next(iter(points)) % 1 != 0.5:
                    continue  # whole lines can push (stake returned), which the EV maths does not model
                key, sel = f"OU{points.pop()}", {"Mais": prices.get("Over"), "Menos": prices.get("Under")}
            else:
                continue
            if all(isinstance(v, (int, float)) and v > 1 for v in sel.values()):
                out.setdefault(key, {})[bk["title"]] = {s: float(v) for s, v in sel.items()}
    return out, last or None


def consensus(odds: dict) -> dict:
    """Adds the average-price book to each market and returns its no-vig probabilities. A single bookmaker is enough
    for a forecast (its prices without the margin); its confidence is "muito baixa" (see confidence)."""
    probs = {}
    for market, books in odds.items():
        if not books:
            continue
        sels = list(next(iter(books.values())))
        avg = {s: float(np.mean([b[s] for b in books.values()])) for s in sels}
        books[AGGREGATES["Avg"]] = avg
        probs[market] = mk.devig(avg)[0]
    return probs


def confidence(n_books: int) -> str | None:
    """How broad the consensus is; separate from the probability of the outcome."""
    return ("alta" if n_books >= 8 else "média" if n_books >= 4 else "baixa" if n_books >= 2
            else "muito baixa" if n_books == 1 else None)


def analyze(ev: dict, sport_key: str, log: bool = True) -> dict:
    sport, competition = describe(sport_key)
    home, away = ev["home_team"], ev["away_team"]
    kick = pd.Timestamp(ev["commence_time"]).tz_convert(LOCAL_TZ)
    odds, last = event_odds(ev)
    sources = [{"source": "the-odds-api.com", "fetched_at": last,
                "note": f"odds de {len(odds.get('Vencedor', {}))} casas internacionais" if odds else
                "sem odds nesta atualização (orçamento de créditos ou casas ainda sem preço)"}]
    if sport == "Futebol" and oddspapi.enabled():  # same game at the Portuguese books
        pt, updated = oddspapi.match_odds(home, away, kick)
        for market, books in _app_markets(pt).items():
            odds.setdefault(market, {}).update(books)
        if pt:
            sources.append({"source": "OddsPapi", "fetched_at": updated, "note": "Betano PT (licenciada SRIJ)"})
    return build(sport_key, sport, competition, home, away, kick, odds, sources, NOTES, log)


def _app_markets(odds: dict) -> dict:
    """OddsPapi names the match result "1X2"; in this module it is "Vencedor", as for the other sports. Copied,
    because consensus() adds the average book to the dicts it gets and these come from a shared cache."""
    return {("Vencedor" if m == "1X2" else m): {b: dict(s) for b, s in books.items()} for m, books in odds.items()}


def from_oddspapi(f: dict, log: bool = True) -> dict:
    """A game only the Portuguese books list (e.g. Liga Portugal 2, Liga BPI): probabilities from the no-vig
    consensus of the Portuguese books (none with Betano PT alone)."""
    kick = f["kickoff"].tz_convert(LOCAL_TZ)
    sources = [{"source": "OddsPapi", "fetched_at": f.get("updated"), "note": "Betano PT (licenciada SRIJ)"}]
    notes = NOTES[:2] + ["Probabilidades de uma única casa (Betano PT) sem a margem: referência mais fraca do que o "
                         "consenso de muitas casas."]
    return build(f"oddspapi_{f['tournament']}", "Futebol", f["competition"], f["home"], f["away"], kick,
                 _app_markets(f["odds"]), sources, notes, log)


def build(sport_key: str, sport: str, competition: str, home: str, away: str, kick, odds: dict, sources: list,
          notes: list, log: bool) -> dict:
    key = f"{sport_key}|{kick.date().isoformat()}|{home}|{away}"
    last = max((s["fetched_at"] or "" for s in sources), default="") or None
    n_books = len(odds.get("Vencedor", {}))
    probs = consensus(odds)
    if "X" in probs.get("Vencedor", {}):  # three-way result (football): double chance and draw no bet follow from it
        probs.update(mk.derived_markets(probs["Vencedor"]))
    conf = confidence(n_books)
    rows = analysis.evaluate_markets(probs, odds, conf) if probs else []
    validation = analysis.validation(sport_key)
    analysis._apply_validation(rows, validation)
    analysis.add_evidence(rows, None, validation, None)  # no own model: the evidence is the market alone
    report = {
        "key": key,
        "identification": {"sport": sport, "league": sport_key, "competition": competition, "home": home,
                           "away": away, "date": kick.date().isoformat(), "time": kick.strftime("%H:%M"),
                           "kickoff": kick.isoformat(),
                           "venue": "—" if sport in INDIVIDUAL else f"Casa de {home} (pavilhão/estádio não indicado)"},
        "data_status": {"odds": sources, "bookmakers": n_books, "confidence": conf, "model_validation": validation,
                        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "statistics": None,
        "predictions": {"probabilities": probs, "model_probabilities": None, "blend": None, "expected_goals": None,
                        "matrix": None, "top_scores": [],
                        "method": f"Consenso do mercado sem margem (Shin) — média de {n_books} casas"}
        if probs else None,
        "markets": rows, "odds": analysis.public_odds(odds), "conclusion": analysis._conclusion(rows, conf, list(notes)),
        "recommendation": analysis.recommend(rows, home, away, sport)}
    report["explanation"] = analysis.explain(report)
    if log and rows:
        db.log_predictions([{"match_key": key, "league": sport_key, "match_date": kick.date().isoformat(),
                             "home": home, "away": away, "market": r["market"], "selection": r["selection"],
                             "model_prob": r["model_prob"], "odds": r["best_odds"], "ev": r["ev"],
                             "category": r["category"], "confidence": conf, "model_version": MODEL_VERSION}
                            for r in rows])
        db.snapshot_odds(key, {m: {b: s for b, s in books.items() if b != AGGREGATES["Avg"]}
                               for m, books in odds.items()}, db.now())  # observation time, as for football
    return report


def _in_window(ev: dict, now: pd.Timestamp) -> bool:
    try:
        start = pd.Timestamp(ev["commence_time"])
    except (KeyError, ValueError, TypeError):
        return False
    return now - pd.Timedelta(hours=2) < start < now + pd.Timedelta(days=HORIZON_DAYS)


def reports(log: bool = True) -> list[dict]:
    """Every upcoming event (from 2 h ago to HORIZON_DAYS ahead) of every configured sport. Fixtures come from the
    free /events endpoint; odds are then requested sport by sport, soonest event first, within the credit budget.
    Events whose odds did not fit in the budget are still listed, without odds."""
    if not odds_api.enabled():
        return []
    now = pd.Timestamp.now(tz=timezone.utc)
    fixtures = {sk: [e for e in odds_api.events(sk) if _in_window(e, now)] for sk in sport_keys()}
    order = sorted((sk for sk, evs in fixtures.items() if evs),
                   key=lambda sk: min(pd.Timestamp(e["commence_time"]) for e in fixtures[sk]))
    out = []
    for sk in order:
        priced = {e.get("id"): e for e in odds_api.fetch_sport(sk, "h2h")}  # result only: 1 credit, twice the sports
        for ev in fixtures[sk]:
            try:
                out.append(analyze(priced.get(ev.get("id")) or {**ev, "bookmakers": []}, sk, log))
            except (KeyError, ValueError, TypeError):  # one malformed event must not hide the rest
                continue
    return sorted(out, key=lambda r: (r["identification"]["date"], r["identification"]["time"]))


def find(key: str) -> dict | None:
    return next((r for r in reports(log=False) if r["key"] == key), None)
