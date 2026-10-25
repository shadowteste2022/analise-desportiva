import pandas as pd

from app import tables
from app.models import dixon_coles


def test_standings_and_simulation(league):
    model = dixon_coles.fit(league, 0.0019)
    season = league[league.Date >= league.Date.max() - pd.Timedelta(days=60)]  # part of the last season
    table = tables.standings(season)
    assert table.Pts.sum() == 3 * (season.FTHG != season.FTAG).sum() + 2 * (season.FTHG == season.FTAG).sum()
    out = tables.simulate(model, season, sims=2000)
    rows = {r["team"]: r for r in out["teams"]}
    assert out["remaining_games"] == 90 - len(season)  # 10 teams, double round robin
    assert sum(r["p_title"] for r in rows.values()) == __import__("pytest").approx(1, abs=1e-9)
    assert rows["Team09"]["p_title"] > rows["Team00"]["p_title"]  # strongest team (conftest) wins more often
    assert all(r["expected_points"] >= r["points"] for r in rows.values())
