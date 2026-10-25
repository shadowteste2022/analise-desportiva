"""Strength rankings for every league with a model, in every sport: how much better or worse than an average team each
side is per game, according to the model.

Why rankings and not tables: the free API plans give neither this season's standings nor the games played before the
results started being recorded, so a points table would be wrong. The model's ratings are what the site actually
knows about every team.
"""
import numpy as np
import pandas as pd

MIN_MATCHES = 5
UNIT = {"Futebol": "golos", "Hóquei no gelo": "golos", "Andebol": "golos", "Basebol": "corridas", "Voleibol": "sets"}
REGISTRY: dict[str, dict] = {}  # filled by the report builders of each export run


def build(model, kind: str, df: pd.DataFrame, sport: str, competition: str, league: str, current: set[str]) -> dict:
    """kind: "goals" (Dixon-Coles) or "points" (points model). Scored/conceded per game against an average team."""
    rows = []
    if kind == "goals":
        base = float((df.FTHG + df.FTAG).mean() / 2)
        for k, t in enumerate(model.teams):
            a, d = float(np.exp(model.attack[k])), float(np.exp(model.defence[k]))
            rows.append((t, base * a, base * d))
    else:
        avg = (model.mean_home + model.mean_away) / 2
        for k, t in enumerate(model.teams):
            rows.append((t, avg + float(model.attack[k]), avg - float(model.defence[k])))
    teams = [{"team": t, "scored": round(s, 2), "conceded": round(c, 2), "net": round(s - c, 2),
              "matches": int(model.matches.get(t, 0)), "current": t in current}
             for t, s, c in rows if model.matches.get(t, 0) >= MIN_MATCHES]
    teams.sort(key=lambda r: -r["net"])
    return {"league": league, "competition": competition, "sport": sport, "unit": UNIT.get(sport, "pontos"),
            "matches": len(df), "last_result": df.Date.max().date().isoformat() if len(df) else None, "teams": teams}


def register(model, kind: str, df: pd.DataFrame, sport: str, competition: str, league: str, current: set[str]) -> None:
    if model is not None and league not in REGISTRY:
        REGISTRY[league] = build(model, kind, df, sport, competition, league, current)


def all_rankings() -> list[dict]:
    order = ["Futebol", "Basquetebol", "Hóquei no gelo", "Futebol americano", "Andebol", "Voleibol", "Basebol", "Râguebi",
             "Futebol australiano"]
    return sorted((r for r in REGISTRY.values() if len(r["teams"]) >= 4),
                  key=lambda r: ((order.index(r["sport"]) if r["sport"] in order else 99), r["competition"]))
