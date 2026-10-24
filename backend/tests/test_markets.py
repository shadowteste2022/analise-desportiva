import numpy as np
import pytest

from app import markets as mk


def test_formulas():
    assert mk.implied(2.0) == 0.5
    assert mk.fair_odds(0.25) == 4.0
    assert mk.ev(0.55, 2.0) == pytest.approx(0.10)


def test_devig_removes_margin():
    fair, margin = mk.devig({"1": 2.0, "X": 3.4, "2": 3.8})
    assert sum(fair.values()) == pytest.approx(1)
    assert margin == pytest.approx(1 / 2 + 1 / 3.4 + 1 / 3.8 - 1)
    assert mk.devig({"1": 2.0, "X": None, "2": 3.0}) == (None, None)


def test_devig_methods_on_longshot():
    odds = {"1": 1.25, "X": 6.5, "2": 13.0}  # heavy favourite vs long shot, ~4% margin
    res = {m: mk.devig(odds, m)[0] for m in ("proportional", "power", "shin")}
    for p in res.values():
        assert sum(p.values()) == pytest.approx(1)
    # Shin and power move probability from the long shot to the favourite
    assert res["shin"]["2"] < res["proportional"]["2"] and res["power"]["2"] < res["proportional"]["2"]
    assert res["shin"]["1"] > res["proportional"]["1"]
    even = {m: mk.devig({"Mais": 1.9, "Menos": 1.9}, m)[0]["Mais"] for m in ("proportional", "power", "shin")}
    assert all(v == pytest.approx(0.5) for v in even.values())


def test_market_probs_consistent():
    m = np.random.default_rng(1).random((11, 11))
    m /= m.sum()
    p = mk.market_probs(m)
    assert sum(p["1X2"].values()) == pytest.approx(1)
    for k in ("OU0.5", "OU2.5", "BTTS"):
        assert sum(p[k].values()) == pytest.approx(1)
    assert p["OU0.5"]["Mais"] >= p["OU1.5"]["Mais"] >= p["OU2.5"]["Mais"]
    assert p["BTTS"]["Sim"] == pytest.approx(m[1:, 1:].sum())


@pytest.mark.parametrize("p,odds,conf,market_p,expected", [
    (0.5, None, "alta", None, mk.SEM_DADOS),
    (0.5, 2.5, None, None, mk.SEM_DADOS),
    (0.80, 1.20, "alta", 0.80, mk.SEM_VALOR),  # very likely event, but negative EV: not a bet
    (0.50, 2.10, "alta", 0.48, mk.VALOR),  # EV +5%, close to market
    (0.51, 2.0, "alta", 0.50, mk.INCONCLUSIVO),  # EV +2% < 3% safety margin
    (0.60, 2.0, "alta", 0.45, mk.INCONCLUSIVO),  # 15pp away from market -> suspicious
    (0.55, 2.0, "baixa", 0.50, mk.INCONCLUSIVO),
    (0.55, 2.0, "média", 0.50, mk.VALOR),
    (0.60, 5.25, "alta", 0.60, mk.INCONCLUSIVO),  # one book at 5.25 when the market says 60%: outlier
])
def test_classify(p, odds, conf, market_p, expected):
    assert mk.classify(p, odds, conf, market_p)[0] == expected


def test_confidence_thresholds():
    assert mk.confidence(5, 50) is None
    assert mk.confidence(15, 50) == "baixa"
    assert mk.confidence(25, 50) == "média"
    assert mk.confidence(45, 50) == "alta"
