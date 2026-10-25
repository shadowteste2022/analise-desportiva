"""League tables now and at the end of the season: the rest of the season simulated SIMS times with the league's
Dixon-Coles model (each remaining game drawn from its scoreline distribution).

Only leagues played as a plain double round robin (each team meets every other home and away, no split or
play-offs), where the remaining games follow from the ones already played.
"""
import logging
from datetime import date

import numpy as np
import pandas as pd

from app import analysis
from app.config import settings
from app.data import football_data as fd

log = logging.getLogger(__name__)
SIMS = 10_000
# split seasons, play-offs or uneven schedules: the remaining fixtures cannot be inferred from a double round robin
NOT_ROUND_ROBIN = {"SC0", "SC1", "SC2", "SC3", "B1", "G1"}


def current_season(df: pd.DataFrame, today: date) -> pd.DataFrame:
    start = pd.Timestamp(today.year if today.month >= 7 else today.year - 1, 7, 1)
    return df[df.Date >= start]


def standings(season: pd.DataFrame) -> pd.DataFrame:
    """Points, games, goals; sorted by points, goal difference, goals scored."""
    teams = sorted(set(season.HomeTeam) | set(season.AwayTeam))
    t = pd.DataFrame(0, index=teams, columns=["J", "V", "E", "D", "GM", "GS", "Pts"])
    for r in season.itertuples():
        for team, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            res = "V" if gf > ga else "E" if gf == ga else "D"
            t.loc[team, ["J", res, "GM", "GS"]] += [1, 1, gf, ga]
    t["Pts"] = 3 * t.V + t.E
    t["DG"] = t.GM - t.GS
    return t.sort_values(["Pts", "DG", "GM"], ascending=False)


def simulate(model, season: pd.DataFrame, sims: int = SIMS, seed: int = 0) -> dict | None:
    table = standings(season)
    teams = list(table.index)
    n = len(teams)
    played = set(zip(season.HomeTeam, season.AwayTeam))
    if table.J.max() > 2 * (n - 1) or any(not model.knows(t) for t in teams):
        return None
    remaining = [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]
    rng = np.random.default_rng(seed)
    idx = {t: k for k, t in enumerate(teams)}
    pts = np.tile(table.Pts.to_numpy(float), (sims, 1))
    gd = np.tile(table.DG.to_numpy(float), (sims, 1))
    gf = np.tile(table.GM.to_numpy(float), (sims, 1))
    for h, a in remaining:
        m = model.score_matrix(h, a)
        flat = m.ravel() / m.sum()
        draw = rng.choice(flat.size, size=sims, p=flat)
        hg, ag = np.divmod(draw, m.shape[1])
        i, j = idx[h], idx[a]
        pts[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd[:, i] += hg - ag; gd[:, j] += ag - hg
        gf[:, i] += hg; gf[:, j] += ag
    # final position per simulation: points, then goal difference, then goals (tiny weights break ties in order)
    score = pts + gd * 1e-3 + gf * 1e-6
    pos = (-score).argsort(axis=1).argsort(axis=1)  # 0 = first
    relegated = 3 if n >= 16 else 2 if n >= 10 else 1
    rows = []
    for k, t in enumerate(teams):
        r = table.loc[t]
        rows.append({"team": t, "played": int(r.J), "won": int(r.V), "drawn": int(r.E), "lost": int(r.D),
                     "gf": int(r.GM), "ga": int(r.GS), "gd": int(r.DG), "points": int(r.Pts),
                     "expected_points": round(float(pts[:, k].mean()), 1),
                     "p_title": round(float((pos[:, k] == 0).mean()), 4),
                     "p_top4": round(float((pos[:, k] < 4).mean()), 4),
                     "p_relegation": round(float((pos[:, k] >= n - relegated).mean()), 4),
                     "expected_position": round(float(pos[:, k].mean() + 1), 1)})
    return {"teams": rows, "remaining_games": len(remaining), "games_per_team": 2 * (n - 1), "relegation_places": relegated,
            "sims": sims}


def all_tables(today: date | None = None) -> list[dict]:
    today = today or date.today()
    out = []
    for lg in settings.league_list:
        if lg in NOT_ROUND_ROBIN or lg in fd.NEW_LEAGUES:
            continue
        try:
            model, df, meta = analysis.get_model(lg)
            season = current_season(df, today)
            if season.empty:
                continue
            sim = simulate(model, season)
        except (RuntimeError, ValueError, KeyError) as e:
            log.warning("Classificação de %s indisponível (%s)", lg, e)
            continue
        if sim:
            out.append({"league": lg, "competition": fd.LEAGUE_NAMES.get(lg, lg), "last_result": meta["last_result"], **sim})
    return out
