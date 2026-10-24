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


def _row(market, sel, p, odds, ev, cat, reasons=("motivo",), market_p=0.5):
    return {"market": market, "selection": sel, "model_prob": p, "best_odds": odds, "bookmaker": "Bk", "ev": ev,
            "category": cat, "reasons": list(reasons), "market_prob_no_vig": market_p}


def test_recommendation_only_with_value():
    from app.analysis import recommend
    rows = [_row("1X2", "1", 0.62, 1.55, -0.04, mk.SEM_VALOR), _row("1X2", "X", 0.22, 4.6, 0.012, mk.INCONCLUSIVO,
            ["EV estimado +1.2%", "abaixo da margem de segurança de 3%"]), _row("1X2", "2", 0.16, 5.5, -0.12, mk.SEM_VALOR)]
    r = recommend(rows, "Benfica", "Porto")
    assert r["kind"] == "nenhuma" and r["label"] == "Empate" and "margem de segurança" in r["text"]
    assert r["most_likely"] == {"label": "Vitória Benfica", "prob": 0.62}  # likely, but not recommended
    rows.append(_row("OU2.5", "Mais", 0.58, 1.9, 0.10, mk.INCONCLUSIVO))  # higher EV but not valid value
    rows.append(_row("BTTS", "Sim", 0.55, 1.95, 0.0725, mk.VALOR))
    r = recommend(rows, "Benfica", "Porto")
    assert r["kind"] == "valor" and r["label"] == "Ambas marcam" and r["odds"] == 1.95
    assert recommend([], "A", "B")["kind"] == "sem_dados"


def test_opportunities_skip_negative_ev_and_bad_data():
    from app.analysis import opportunities
    rows = [_row("1X2", "1", 0.62, 1.55, -0.04, mk.SEM_VALOR),
            _row("1X2", "X", 0.30, 5.0, 0.50, mk.INCONCLUSIVO, ["EV estimado +50%", f"odd 50% {mk.OUTLIER_TEXT} — erro"]),
            _row("OU2.5", "Mais", 0.55, 1.95, 0.0725, mk.INCONCLUSIVO), _row("BTTS", "Sim", 0.5, 2.05, 0.025, mk.INCONCLUSIVO)]
    r = {"identification": {"home": "Benfica", "away": "Porto", "sport": "Futebol"}, "markets": rows}
    opps = opportunities(r)
    assert [o["label"] for o in opps] == ["Mais de 2.5 golos", "Ambas marcam"]  # outlier and negative EV left out
    assert opps[0]["licensed_pt"] is False and opps[0]["odds"] == 1.95
    rows += [_row("OU3.5", "Menos", 0.8, 1.83, 0.46, mk.INCONCLUSIVO, market_p=None),  # nothing to check it against
             _row("OU1.5", "Mais", 0.9, 1.4, 0.26, mk.INCONCLUSIVO, ["EV", f"modelo diverge 20% {mk.DIVERGENCE_TEXT}"])]
    assert [o["label"] for o in opportunities(r)] == ["Mais de 2.5 golos", "Ambas marcam"]


def test_market_reference_from_bookmaker_consensus():
    from app.models import blend
    odds = {"OU3.5": {"A": {"Mais": 3.0, "Menos": 1.40}, "B": {"Mais": 2.8, "Menos": 1.45}},
            "OU4.5": {"A": {"Mais": 5.0, "Menos": 1.15}}}  # a single bookmaker is no consensus
    fair = blend.market_fair(odds, "OU3.5")
    assert sum(fair.values()) == pytest.approx(1) and fair["Menos"] > 0.6
    assert blend.market_fair(odds, "OU4.5") is None
    w = {"1X2": {"a": 0, "b": 1}, "OU2.5": {"a": 0.0, "b": 1.0}}
    assert blend.weights_for(w, "OU3.5") == w["OU2.5"] and blend.weights_for(w, "Vencedor") is None
    out, used = blend.apply({"OU3.5": {"Mais": 0.1, "Menos": 0.9}}, odds, w)
    assert used == ["OU3.5"] and out["OU3.5"]["Menos"] == pytest.approx(fair["Menos"])  # a=0 -> market


def test_pick_labels():
    from app.analysis import pick_label
    assert pick_label("OU2.5", "Mais", "A", "B") == "Mais de 2.5 golos"
    assert pick_label("OU220.5", "Menos", "A", "B", "Basquetebol") == "Menos de 220.5 pontos"
    assert pick_label("Vencedor", "2", "Lakers", "Celtics", "Basquetebol") == "Vitória Celtics"


def test_confidence_thresholds():
    assert mk.confidence(5, 50) is None
    assert mk.confidence(15, 50) == "baixa"
    assert mk.confidence(25, 50) == "média"
    assert mk.confidence(45, 50) == "alta"
