from app import rankings
from app.models import dixon_coles, points
from tests.test_multisport import points_league


def test_rankings_order_by_strength(league):
    r = rankings.build(dixon_coles.fit(league, 0.0019), "goals", league, "Futebol", "Liga", "x", {"Team09"})
    names = [t["team"] for t in r["teams"]]
    assert names[0] == "Team09" and names[-1] in ("Team00", "Team01") and r["unit"] == "golos"  # conftest strengths (two weakest are close)
    assert r["teams"][0]["current"] and not r["teams"][-1]["current"]
    df = points_league()
    p = rankings.build(points.fit(df, 0.003), "points", df, "Basquetebol", "NBA", "y", set())
    assert p["teams"][0]["team"] == "T11" and p["teams"][0]["net"] > 0 > p["teams"][-1]["net"] and p["unit"] == "pontos"
