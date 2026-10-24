import numpy as np
import pytest

from app import backtest
from app.data import football_data as fd
from app.models import blend


def test_backtest_walk_forward(league, monkeypatch):
    monkeypatch.setattr(fd, "load_results", lambda lg, seasons=None, **k: (
        league, {"source": "synthetic", "last_result": league.Date.max().date().isoformat()}))
    monkeypatch.setattr(backtest, "BLEND_MIN_PREDICTIONS", 50)
    reports = backtest.run_all(["T0"], test_seasons=2)
    r = reports[0]
    # first test season only trains the combination: just the second season is evaluated
    assert {x[:4] for x in r["by_season"]} == {str(league.Date.dt.year.max())}
    assert r["predictions"] > 50
    a = r["accuracy"]["1X2"]
    # the market here is the true model plus 5% margin removed, so it should be at least as good
    assert a["market_closing_no_vig"]["log_loss"] <= a["model_same_matches"]["log_loss"] + 0.02
    assert {"dixon_coles_only", "market_prematch_no_vig"} <= a.keys()
    assert sum(c["n"] for c in r["calibration"]) == r["predictions"] * 7  # 3 + 2 + 2 outcomes
    w = blend.load()["weights"]["1X2"]
    assert w["b"] > w["a"]  # market is the truth here, so the pool leans on it
    assert reports[-1]["league"] == "TODAS"
    o = r["betting"]["overall"]
    if o["bets"]:
        assert o["won"] + o["lost"] == o["bets"]
        assert o["max_drawdown_units"] >= 0
    assert any("não garantem" in w for w in r["warnings"])


def test_backtest_needs_history(league, monkeypatch):
    one = league[league.Date < league.Date.min() + __import__("pandas").Timedelta(days=200)]
    monkeypatch.setattr(fd, "load_results", lambda lg, seasons=None, **k: (one, {"source": "s", "last_result": ""}))
    with pytest.raises(ValueError):
        backtest.run_all(["T0"], test_seasons=1)


def test_pinnacle_clv():
    row = {"PSCH": 2.0, "PSCD": 3.5, "PSCA": 4.0, "PC>2.5": 1.9, "PC<2.5": float("nan")}
    fair = backtest._pinnacle_fair(row)
    assert set(fair) == {"1X2"} and sum(fair["1X2"].values()) == pytest.approx(1)  # O/U incomplete -> skipped
    bets = __import__("pandas").DataFrame({"clv_pinnacle": [0.05, -0.02, np.nan]})
    s = backtest._pinnacle_stats(bets)
    assert s["clv_pinnacle_n"] == 2 and s["avg_clv_pinnacle"] == pytest.approx(0.015)
    assert s["share_beating_pinnacle"] == 0.5


def test_blend_pool():
    rng = np.random.default_rng(0)
    truth = rng.dirichlet([3, 2, 2], size=3000)
    y = np.eye(3)[[rng.choice(3, p=p) for p in truth]]
    noisy = np.array([rng.dirichlet(p * 20) for p in truth])  # model = truth + noise
    w = blend.fit(noisy, truth, y)
    assert w["b"] > 0.8 and w["a"] < 0.3
    assert np.allclose(blend.combine(noisy, truth, 0.4, 0.6).sum(axis=1), 1)
    probs = {"1X2": {"1": 0.5, "X": 0.3, "2": 0.2}, "BTTS": {"Sim": 0.5, "Não": 0.5}}
    odds = {"1X2": {fd.AGGREGATES["Avg"]: {"1": 2.5, "X": 3.2, "2": 2.9}}}
    out, used = blend.apply(probs, odds, {"1X2": {"a": 0.0, "b": 1.0}, "OU2.5": {"a": 0.5, "b": 0.5}})
    assert used == ["1X2"] and out["BTTS"] == probs["BTTS"]
    assert out["1X2"]["1"] == pytest.approx(blend.market_fair(odds, "1X2")["1"])  # b=1, a=0 -> market
