"""Forecasts for the other sports from API-Sports fixtures and results (see data/api_sports.py).

Hockey and baseball (few goals/runs): the Dixon-Coles Poisson model, with the regulation draw shared out as the
games always end with a winner. Basketball, American football, rugby, handball and volleyball (sets): the points
model (models/points.py). History = each league's 2024 season plus the games recorded since, so the confidence is
capped at "baixa" while the old season carries most of the weight.
"""
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import analysis, markets as mk, rankings
from app.config import settings
from app.data import api_sports as aps
from app.data.football_data import LOCAL_TZ
from app.data.odds_api import _similarity
from app.models import dixon_coles, points

log = logging.getLogger(__name__)
XI = 0.003  # time decay per day: these leagues play more often than football
MIN_GAMES = 50
# sport -> (model, draws possible, totals shown)
CONFIG = {"basketball": ("points", False, True), "american-football": ("points", False, True),
          "rugby": ("points", True, True), "handball": ("points", True, True), "volleyball": ("points", False, False),
          "hockey": ("goals", False, True), "baseball": ("goals", False, True), "afl": ("points", True, True)}
NOTES = ["Histórico da liga: época de 2024 mais os jogos registados desde 26/10/2026 — as equipas mudaram desde 2024, "
         "por isso a confiança fica no máximo «baixa» até haver jogos recentes suficientes.",
         "Lesões, ausências e calendário (jogos seguidos, viagens) não entram na previsão."]


def _goals_forecast(model, home, away, draws):
    m = model.score_matrix(home, away)
    i, j = np.indices(m.shape)
    p1, px, p2 = m[i > j].sum(), m[i == j].sum(), m[i < j].sum()
    res = {"1": p1, "X": px, "2": p2} if draws else {"1": p1 + px * p1 / (p1 + p2), "2": p2 + px * p2 / (p1 + p2)}
    lam, mu = model.expected_goals(home, away)
    base = math.floor(lam + mu)
    lines = [x + 0.5 for x in range(max(0, base - 2), base + 2)]
    totals = {f"OU{x:g}": {"Mais": float(m[i + j > x].sum()), "Menos": float(m[i + j <= x].sum())} for x in lines}
    ratings = {t: {"attack": round(float(np.exp(model.attack[model.teams.index(t)])), 3),
                   "defence": round(float(np.exp(model.defence[model.teams.index(t)])), 3)} for t in (home, away)}
    return {k: float(v) for k, v in res.items()}, totals, (lam, mu), ratings, round(float(np.exp(model.home_adv)), 3)


def _points_forecast(model, home, away, draws, with_totals):
    eh, ea = model.expected(home, away)
    total = eh + ea
    step = max(1, round(model.sd_total / 3))
    lines = [round(total) + 0.5 + k * step for k in range(-2, 3)]
    avg = (model.mean_home + model.mean_away) / 2
    ratings = {t: {"attack": round(1 + model.attack[model.teams.index(t)] / avg, 3),
                   "defence": round(1 - model.defence[model.teams.index(t)] / avg, 3)} for t in (home, away)}
    return (model.result(home, away, draws), model.totals(home, away, lines) if with_totals else {}, (eh, ea), ratings,
            round(model.mean_home / model.mean_away, 3))


def report(sport: str, game: dict, model, df: pd.DataFrame) -> dict:
    kind, draws, with_totals = CONFIG[sport]
    name = aps.SPORTS[sport][1]
    kick = pd.Timestamp(game["kickoff"]).tz_convert(LOCAL_TZ)
    home, away = game["home"], game["away"]
    league = f"aps_{sport}_{game['league_id']}"
    key = f"{league}|{kick.date().isoformat()}|{home}|{away}"
    when = kick.tz_localize(None).normalize()
    comp = f"{game['country']} — {game['league']}" if game["country"] else game["league"]
    out = {"key": key, "markets": [], "odds": {}, "alerts": [], "recommendation": None, "statistics": None,
           "predictions": None,
           "identification": {"sport": name, "league": league, "competition": comp, "home": home, "away": away,
                              "date": kick.date().isoformat(), "time": kick.strftime("%H:%M"), "kickoff": kick.isoformat(),
                              "venue": f"Casa de {home} (pavilhão/estádio não indicado)"}}
    known = model is not None and model.knows(home) and model.knows(away)
    n_h, n_a = (model.matches.get(home, 0), model.matches.get(away, 0)) if model else (0, 0)
    conf = mk.confidence(n_h, n_a) if known else None
    if conf and len(df[df.Date >= pd.Timestamp("2026-07-01")]) < 0.5 * len(df):
        conf = "baixa"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["data_status"] = {"results": {"source": "API-Sports", "matches": len(df),
                                      "last_result": df.Date.max().date().isoformat() if len(df) else None, "fetched_at": now},
                          "odds": [], "confidence": conf, "analysed_at": now, "team_matches": {home: n_h, away: n_a}}
    notes = list(NOTES)
    if conf is None:
        out["conclusion"] = analysis._conclusion([], None, notes)
        return out
    res, totals, (eh, ea), ratings, home_adv = (_goals_forecast(model, home, away, draws) if kind == "goals"
                                                else _points_forecast(model, home, away, draws, with_totals))
    out["statistics"] = {
        "form": {home: analysis._form(df, home, when), away: analysis._form(df, away, when)},
        "home_away": {home: analysis._form(df, home, when, 10, "home"), away: analysis._form(df, away, when, 10, "away")},
        "h2h": analysis._h2h(df, home, away, when),
        "rest_days": {home: analysis._rest_days(df, home, when), away: analysis._rest_days(df, away, when)},
        "ratings": ratings, "home_advantage": home_adv}
    probs = {"Vencedor": res, **totals}
    out["predictions"] = {
        "expected_goals": {home: round(eh, 1), away: round(ea, 1)}, "model_expected_goals": None, "probabilities": probs,
        "model_probabilities": probs, "blend": None, "top_scores": [], "matrix": None, "halftime": None,
        "corners": None, "cards": None, "shots": None, "shots_on_target": None,
        "method": ("Modelo de Poisson (Dixon-Coles)" if kind == "goals" else "Modelo de força de ataque e defesa em pontos")
                  + f" com decaimento temporal, treinado em {len(df)} jogos desta liga até {out['data_status']['results']['last_result']}"}
    out["conclusion"] = analysis._conclusion([], conf, notes)
    out["explanation"] = analysis.explain(out)
    return out


def _fit(kind: str, df: pd.DataFrame):
    return dixon_coles.fit(df, XI) if kind == "goals" else points.fit(df, XI)


def reports(known: list[dict]) -> list[dict]:
    """Every upcoming game of every API-Sports sport not already forecast by another source."""
    if not aps.enabled():
        return []
    out = []
    for sport in CONFIG:  # MMA (fighters, not teams) has its own module
        try:
            found = aps.upcoming(sport)
        except Exception:  # one sport failing must not stop the others
            log.exception("API-Sports %s: lista de jogos falhou", sport)
            continue
        found = [g for g in found if not any(k["date"] == g["kickoff"][:10] and _similarity(g["home"], k["home"]) >= 0.75
                                             and _similarity(g["away"], k["away"]) >= 0.75 for k in known)]
        counts = pd.Series([g["league_id"] for g in found]).value_counts()
        seasons = {g["league_id"]: g["season"] for g in found}
        aps.backfill(sport, [(lg, seasons[lg]) for lg in counts.index])  # leagues with the most games first
        models = {}
        for g in found:
            lg = g["league_id"]
            if lg not in models:
                df = aps.history(sport, lg)
                try:
                    models[lg] = (_fit(CONFIG[sport][0], df) if len(df) >= MIN_GAMES else None, df)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    log.warning("API-Sports %s: modelo da liga %s falhou (%s)", sport, lg, e)
                    models[lg] = (None, df)
            try:
                out.append(report(sport, g, *models[lg]))
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                log.exception("API-Sports %s: falha ao analisar %s – %s", sport, g.get("home"), g.get("away"))
        for g in {g["league_id"]: g for g in found}.values():  # strength ranking of every league with a model
            model, df = models[g["league_id"]]
            current = set(df[df.Date >= pd.Timestamp("2026-07-01")][["HomeTeam", "AwayTeam"]].to_numpy().ravel())
            current |= {x[s] for x in found if x["league_id"] == g["league_id"] for s in ("home", "away")}
            comp = f"{g['country']} — {g['league']}" if g["country"] else g["league"]
            rankings.register(model, CONFIG[sport][0], df, aps.SPORTS[sport][1], comp,
                              f"aps_{sport}_{g['league_id']}", current)
    return out
