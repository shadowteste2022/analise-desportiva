from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app import multisport
from app.data import api_sports as aps
from app.models import points


def points_league(seed=0, teams=12, games=400):
    rng = np.random.default_rng(seed)
    strength = np.linspace(-8, 8, teams)
    rows, day = [], pd.Timestamp("2026-01-01")
    for k in range(games):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"Date": day + pd.Timedelta(days=k // 6), "HomeTeam": f"T{h:02d}", "AwayTeam": f"T{a:02d}",
                     "FTHG": int(rng.normal(112 + strength[h] - strength[a] / 2, 10)),
                     "FTAG": int(rng.normal(108 + strength[a] - strength[h] / 2, 10))})
    return pd.DataFrame(rows)


def test_points_model_recovers_strength():
    m = points.fit(points_league(), 0.003)
    res = m.result("T11", "T00", draws=False)
    assert res["1"] > 0.8 and res["1"] + res["2"] == pytest.approx(1)  # strongest at home vs weakest
    hd = m.result("T05", "T06", draws=True)
    assert sum(hd.values()) == pytest.approx(1) and 0 < hd["X"] < 0.1
    tot = m.totals("T05", "T06", [200.5, 240.5])
    assert tot["OU200.5"]["Mais"] > tot["OU240.5"]["Mais"]
    assert 8 < m.sd_margin < 20  # residual spread close to the simulated noise (10 per side)


def test_normalize_handles_every_sport_shape():
    ts = int(datetime(2026, 10, 26, 18, tzinfo=timezone.utc).timestamp())
    basket = {"id": 1, "timestamp": ts, "status": {"short": "FT"}, "league": {"id": 12, "name": "NBA", "season": "2026-2027"},
              "country": {"name": "USA"}, "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
              "scores": {"home": {"total": 110}, "away": {"total": 99}}}
    nfl = {"game": {"id": 2, "date": {"timestamp": ts}, "status": {"short": "NS"}},
           "league": {"id": 1, "name": "NFL", "season": 2026, "country": {"name": "USA"}},
           "teams": {"home": {"name": "C"}, "away": {"name": "D"}}, "scores": {"home": {"total": None}, "away": {"total": None}}}
    hockey = {"id": 3, "timestamp": ts, "status": {"short": "AOT"}, "league": {"id": 57, "name": "NHL", "season": 2026},
              "country": {"name": "USA"}, "teams": {"home": {"name": "E"}, "away": {"name": "F"}}, "scores": {"home": 3, "away": 2}}
    out = aps.normalize([basket, nfl, hockey, {"broken": 1}])
    assert [(g["id"], g["hs"], g["as"], g["status"]) for g in out] == [(1, 110, 99, "FT"), (2, None, None, "NS"), (3, 3, 2, "AOT")]
    assert aps.history_season("2026-2027") == "2024-2025" and aps.history_season(2026) == 2024


def test_multisport_report(monkeypatch):
    df = points_league()
    kick = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    game = {"id": 9, "kickoff": kick, "status": "NS", "league_id": 12, "league": "NBA", "country": "USA",
            "season": "2026-2027", "home": "T11", "away": "T00", "hs": None, "as": None}
    r = multisport.report("basketball", game, points.fit(df, 0.003), df)
    p = r["predictions"]["probabilities"]
    assert r["identification"]["sport"] == "Basquetebol" and set(p["Vencedor"]) == {"1", "2"}
    assert any(k.startswith("OU") for k in p) and "Pontos esperados" in " ".join(r["explanation"])
    unknown = multisport.report("basketball", {**game, "home": "Nova"}, points.fit(df, 0.003), df)
    assert unknown["predictions"] is None
