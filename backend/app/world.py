"""Forecasts for the world's football leagues from API-Football fixtures and results (see data/api_football.py).

Same Dixon-Coles model and derived markets as the football-data leagues, without odds or blending: the history is
each league's 2024 season plus the games recorded since, so the confidence shown is capped at "baixa" until the
recorded games carry most of the weight.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import analysis, markets as mk
from app.config import settings
from app.data import api_football as af
from app.data.football_data import LOCAL_TZ
from app.data.odds_api import _similarity
from app.models import dixon_coles

log = logging.getLogger(__name__)
NOTES = ["Histórico da liga: época de 2024 mais os jogos registados desde 26/10/2026 — as equipas mudaram desde 2024, "
         "por isso a confiança fica no máximo «baixa» até haver jogos recentes suficientes.",
         "Lesões, suspensões, onzes e meteorologia não entram na previsão.",
         "Sem estatísticas de cantos, cartões e remates para estas ligas."]


def _duplicate(game: dict, known: list[dict]) -> bool:
    """Same game already analysed from another source (same day, both team names alike)."""
    day = game["kickoff"][:10]
    return any(k["date"] == day and _similarity(game["home"], k["home"]) >= 0.75
               and _similarity(game["away"], k["away"]) >= 0.75 for k in known)


def report(game: dict, model, df: pd.DataFrame) -> dict:
    kick = pd.Timestamp(game["kickoff"]).tz_convert(LOCAL_TZ)
    home, away = game["home"], game["away"]
    key = f"apif_{game['league_id']}|{kick.date().isoformat()}|{home}|{away}"
    when = kick.tz_localize(None).normalize()
    ident = {"sport": "Futebol", "league": f"apif_{game['league_id']}", "competition": f"{game['country']} — {game['league']}",
             "home": home, "away": away, "date": kick.date().isoformat(), "time": kick.strftime("%H:%M"),
             "kickoff": kick.isoformat(), "venue": f"Casa de {home} (estádio não indicado)"}
    known = model is not None and model.knows(home) and model.knows(away)
    conf = mk.confidence(model.matches.get(home, 0), model.matches.get(away, 0)) if known else None
    recent = df[df.Date >= pd.Timestamp("2026-07-01")]
    if conf and len(recent) < 0.5 * len(df):  # mostly 2024 data: teams have changed since
        conf = "baixa"
    status = {"results": {"source": "API-Football", "matches": len(df),
                          "last_result": df.Date.max().date().isoformat() if len(df) else None,
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
              "odds": [], "confidence": conf, "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "team_matches": {home: model.matches.get(home, 0) if model else 0, away: model.matches.get(away, 0) if model else 0}}
    out = {"key": key, "identification": ident, "data_status": status, "markets": [], "odds": {}, "alerts": [],
           "recommendation": None, "statistics": None, "predictions": None}
    notes = list(NOTES)
    if not known or conf is None:
        notes.insert(0, "Pelo menos uma das equipas tem menos de %d jogos no histórico desta liga: sem previsão."
                     % settings.min_team_matches)
        out["conclusion"] = analysis._conclusion([], None, notes)
        return out
    out["statistics"] = {
        "form": {home: analysis._form(df, home, when), away: analysis._form(df, away, when)},
        "home_away": {home: analysis._form(df, home, when, 10, "home"), away: analysis._form(df, away, when, 10, "away")},
        "h2h": analysis._h2h(df, home, away, when),
        "rest_days": {home: analysis._rest_days(df, home, when), away: analysis._rest_days(df, away, when)},
        "ratings": {t: {"attack": round(float(np.exp(model.attack[model.teams.index(t)])), 3),
                        "defence": round(float(np.exp(model.defence[model.teams.index(t)])), 3)} for t in (home, away)},
        "home_advantage": round(float(np.exp(model.home_adv)), 3)}
    m = model.score_matrix(home, away)
    lam, mu = model.expected_goals(home, away)
    probs = mk.market_probs(m)
    probs.update(mk.derived_markets(probs["1X2"], m))
    probs.update(mk.score_markets(m))
    share = analysis._ht_share(df)
    if share:
        probs.update(mk.halftime_markets(lam, mu, share))
        probs.update(mk.halves_markets(lam, mu, share))
    top = sorted(((float(m[i, j]), f"{i}-{j}") for i in range(7) for j in range(7)), reverse=True)[:10]
    out["predictions"] = {
        "expected_goals": {home: round(lam, 2), away: round(mu, 2)}, "model_expected_goals": None,
        "probabilities": probs, "model_probabilities": probs, "blend": None,
        "top_scores": [{"score": s, "prob": round(p, 4)} for p, s in top], "matrix": np.round(m[:6, :6], 4).tolist(),
        "halftime": {"goal_share": round(share, 3), "expected_goals": {home: round(lam * share, 2), away: round(mu * share, 2)}}
        if share else None, "corners": None, "cards": None, "shots": None, "shots_on_target": None,
        "method": "Dixon-Coles com decaimento temporal (ξ=%.4f/dia), treinado em %d jogos desta liga até %s"
                  % (settings.time_decay_xi, len(df), status["results"]["last_result"])}
    out["conclusion"] = analysis._conclusion([], conf, notes)
    return out


def reports(known: list[dict]) -> list[dict]:
    """World games not already covered by the other sources, each with the model of its league when its history
    allows one."""
    if not af.enabled():
        return []
    games = [g for g in af.upcoming() if not _duplicate(g, known)]
    leagues = sorted({g["league_id"] for g in games}, key=lambda lg: -sum(g["league_id"] == lg for g in games))
    af.backfill(leagues)  # leagues with the most games first
    models, out = {}, []
    for g in games:
        lg = g["league_id"]
        if lg not in models:
            df = af.history(lg)
            try:
                models[lg] = (dixon_coles.fit(df, settings.time_decay_xi) if len(df) >= 50 else None, df)
            except (ValueError, RuntimeError) as e:
                log.warning("API-Football: modelo da liga %s falhou (%s)", lg, e)
                models[lg] = (None, df)
        try:
            out.append(report(g, *models[lg]))
        except (KeyError, ValueError, TypeError):  # one malformed game must not hide the rest
            log.exception("API-Football: falha ao analisar %s – %s", g.get("home"), g.get("away"))
    return out
