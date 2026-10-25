import numpy as np
import pytest
from scipy.stats import poisson

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


def _row(market, sel, p, odds, ev, cat, reasons=("motivo",), market_p=0.5, evidence=50):
    return {"market": market, "selection": sel, "model_prob": p, "best_odds": odds, "bookmaker": "Bk", "ev": ev,
            "ev_conservative": ev, "min_odds": round(1.03 / p, 2), "evidence": {"score": evidence, "label": "moderada"},
            "category": cat, "reasons": list(reasons), "market_prob_no_vig": market_p}


def test_recommendation_only_with_value():
    from app.analysis import recommend
    rows = [_row("1X2", "1", 0.62, 1.55, -0.04, mk.SEM_VALOR), _row("1X2", "X", 0.22, 4.9, 0.078, mk.INCONCLUSIVO,
            ["EV estimado +7.8%", "liga sem validação — margem de segurança não garante vantagem"]), _row("1X2", "2", 0.16, 5.5, -0.12, mk.SEM_VALOR)]
    tiny = recommend([_row("1X2", "X", 0.22, 4.6, 0.012, mk.INCONCLUSIVO)], "Benfica", "Porto")
    assert tiny["kind"] == "sem_valor" and "label" not in tiny  # +1.2% does not reach the minimum odds
    r = recommend(rows, "Benfica", "Porto")
    assert r["kind"] == "melhor_opcao" and r["label"] == "Empate" and "margem de segurança" in r["text"]
    assert r["most_likely"] == {"label": "Vitória Benfica", "prob": 0.62}  # likely, but not the best bet
    suspicious = rows + [_row("1X2", "2", 0.30, 5.2, 0.56, mk.INCONCLUSIVO, [f"odd 56% {mk.OUTLIER_TEXT}"])]
    assert recommend(suspicious, "Benfica", "Porto")["label"] == "Empate"  # a +56% data error is never the pick
    only_negative = [_row("1X2", "1", 0.62, 1.55, -0.04, mk.SEM_VALOR)]
    neg = recommend(only_negative, "Benfica", "Porto")
    assert neg["kind"] == "sem_valor" and "label" not in neg and "não apostar" in neg["text"]  # no long shot shown as the bet
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
    assert [o["label"] for o in opps] == ["Mais de 2.5 golos"]  # outlier, negative EV and +2.5% (< 3% margin) left out
    assert opps[0]["licensed_pt"] is False and opps[0]["odds"] == 1.95
    rows += [_row("OU3.5", "Menos", 0.8, 1.83, 0.46, mk.INCONCLUSIVO, market_p=None),  # nothing to check it against
             _row("OU1.5", "Mais", 0.9, 1.4, 0.26, mk.INCONCLUSIVO, ["EV", f"modelo diverge 20% {mk.DIVERGENCE_TEXT}"])]
    assert [o["label"] for o in opportunities(r)] == ["Mais de 2.5 golos"]


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


def test_derived_markets_are_consistent():
    from app.analysis import outcome
    m = np.outer(poisson.pmf(np.arange(11), 1.6), poisson.pmf(np.arange(11), 1.1)); m /= m.sum()
    base = mk.market_probs(m)
    d = mk.derived_markets(base["1X2"], m)
    p = base["1X2"]
    assert d["DC"]["1X"] == pytest.approx(p["1"] + p["X"]) and d["DNB"]["1"] == pytest.approx(p["1"] / (p["1"] + p["2"]))
    assert d["AH-1.5"]["1"] < p["1"] < d["AH+1.5"]["1"]  # winning by 2+ < winning < not losing by 2+
    assert d["GC0.5"]["Mais"] == pytest.approx(1 - m[0, :].sum()) and d["GF0.5"]["Mais"] == pytest.approx(1 - m[:, 0].sum())
    assert outcome("DC", "X2", 1, 1) and not outcome("DC", "12", 0, 0) and outcome("DNB", "1", 2, 2) is None
    assert outcome("AH-1.5", "1", 3, 1) and outcome("AH-1.5", "2", 2, 1) and outcome("AH+1.5", "1", 0, 1)
    assert outcome("GC1.5", "Mais", 2, 0) and outcome("GF0.5", "Menos", 3, 0)


def test_draw_no_bet_ev_counts_the_refund():
    from app.analysis import evaluate_markets
    probs = {"1X2": {"1": 0.5, "X": 0.25, "2": 0.25}, "DNB": {"1": 2 / 3, "2": 1 / 3}}
    odds = {"DNB": {"Betano PT": {"1": 1.6, "2": 2.6}, "Bwin PT": {"1": 1.55, "2": 2.55}}}
    r = next(x for x in evaluate_markets(probs, odds, "alta") if x["market"] == "DNB" and x["selection"] == "1")
    assert r["ev"] == pytest.approx(0.5 * 1.6 + 0.25 - 1, abs=1e-4)  # win pays, draw refunds, loss loses
    assert r["min_odds"] == pytest.approx(1.5 * 1.03, abs=0.01) and r["n_books"] == 2


def test_evidence_score_and_conservative_ranking():
    from app.analysis import NO_OPPORTUNITY, recommend
    strong = mk.evidence(6, "alta", {"validated": True}, 0.50, 0.51)
    weak = mk.evidence(1, None, {"validated": False, "reason": "sem backtest"}, None, 0.5)
    assert strong["score"] == 100 and strong["label"] == "forte" and weak["score"] == 8 and weak["label"] == "fraca"
    assert mk.evidence(6, "alta", {"validated": False, "model_log_loss": 1.0}, 0.50, 0.51)["label"] == "moderada"
    rows = [_row("1X2", "1", 0.5, 2.2, 0.10, mk.INCONCLUSIVO, evidence=10),  # bigger EV, thin evidence
            _row("OU2.5", "Mais", 0.55, 1.95, 0.07, mk.INCONCLUSIVO, evidence=90)]
    r = recommend(rows, "Benfica", "Porto")
    assert r["label"] == "Mais de 2.5 golos" and r["verdict"] == NO_OPPORTUNITY


def test_performance_uses_latest_estimate_per_selection():
    from app.analysis import performance
    rows = [  # newest first, as the database returns them
        {"match_key": "k1", "market": "1X2", "selection": "1", "model_prob": 0.6, "odds": 1.9, "ev": 0.14, "won": True},
        {"match_key": "k1", "market": "1X2", "selection": "1", "model_prob": 0.4, "odds": 2.0, "ev": -0.2, "won": True},
        {"match_key": "k1", "market": "1X2", "selection": "2", "model_prob": 0.2, "odds": 4.0, "ev": -0.2, "won": False},
        {"match_key": "k2", "market": "DNB", "selection": "1", "model_prob": 0.7, "odds": 1.4, "ev": -0.02, "won": None}]
    out = performance(rows)
    t = out["markets"]["todos"]
    assert t["selections"] == 2 and t["bets"] == 1 and t["profit"] == pytest.approx(0.9)  # void DNB left out
    assert t["brier"] == pytest.approx(((0.6 - 1) ** 2 + 0.2 ** 2) / 2)
