"""Walk-forward backtest over many seasons, with out-of-sample validation of the model+market combination.

1. Each week the Dixon-Coles model is refit on matches strictly before that week (same training window as live).
2. The model+market weights used in season S are fitted only on predictions from seasons before S, pooling all
   leagues; the first test season therefore only trains the combination and is left out of every metric.
3. Bets use the same decision code as the live app. Run: python -m app.backtest
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import markets as mk
from app.analysis import evaluate_markets, outcome
from app.config import settings
from app.data import football_data as fd
from app.models import blend, dixon_coles

WARMUP_SEASONS = 2
TEST_SEASONS = 8  # average/closing odds exist from 2019/20 on
BLEND_MIN_PREDICTIONS = 200
SELS = {"1X2": ["1", "X", "2"], "OU2.5": ["Mais", "Menos"], "BTTS": ["Sim", "Não"]}


def backtestable() -> list[str]:
    """Leagues with pre-match and closing average odds per season. The extra leagues only publish closing odds,
    so the model+market combination cannot be validated on them; they stay unvalidated (no VALOR POTENCIAL)."""
    return [lg for lg in settings.league_list if lg not in fd.NEW_LEAGUES]


def _season(d: pd.Series) -> pd.Series:
    return d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)


def _fair(row, closing: bool) -> dict:
    """No-vig probabilities from the market average (pre-match: blend input; closing: the benchmark)."""
    odds = fd.row_odds(row, closing=closing)
    return {m: f for m in SELS if (f := blend.market_fair(odds, m))}


PINNACLE_CLOSE = {"1X2": {"1": "PSCH", "X": "PSCD", "2": "PSCA"}, "OU2.5": {"Mais": "PC>2.5", "Menos": "PC<2.5"}}


def _pinnacle_fair(row, method: str | None = None) -> dict:
    """No-vig Pinnacle closing probabilities: the sharp benchmark for CLV (columns exist 2019/20 to early 2025/26)."""
    out = {}
    for market, cols in PINNACLE_CLOSE.items():
        fair, _ = mk.devig({s: row.get(c) for s, c in cols.items()}, method)
        if fair:
            out[market] = fair
    return out


def _scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier (sum over outcomes) and log loss for probability rows p against one-hot y."""
    return {"brier": round(float(((p - y) ** 2).sum(axis=1).mean()), 4),
            "log_loss": round(float(-np.log(np.clip((p * y).sum(axis=1), 1e-12, None)).mean()), 4), "n": len(p)}


def _max_drawdown(equity: np.ndarray) -> float:
    return float((np.maximum.accumulate(np.r_[0, equity]) - np.r_[0, equity]).max()) if len(equity) else 0.0


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 7) -> list[float] | None:
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _bet_stats(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    profit = bets.profit.to_numpy()
    return {"bets": len(bets), "won": int(bets.won.sum()), "lost": int((~bets.won).sum()),
            "profit_units": round(float(profit.sum()), 2), "yield": round(float(profit.mean()), 4),
            "yield_ci95": _bootstrap_ci(profit), "max_drawdown_units": round(_max_drawdown(profit.cumsum()), 2),
            "avg_odds": round(float(bets.odds.mean()), 2),
            "avg_clv": round(float(bets.clv.mean()), 4) if bets.clv.notna().any() else None,
            **_pinnacle_stats(bets)}


def _pinnacle_stats(bets: pd.DataFrame) -> dict:
    """CLV against Pinnacle's no-vig close: positive on average only if the prices taken beat the sharpest market."""
    c = bets.clv_pinnacle.dropna().to_numpy() if "clv_pinnacle" in bets else np.array([])
    if not len(c):
        return {"clv_pinnacle_n": 0}
    return {"clv_pinnacle_n": len(c), "avg_clv_pinnacle": round(float(c.mean()), 4),
            "median_clv_pinnacle": round(float(np.median(c)), 4), "clv_pinnacle_ci95": _bootstrap_ci(c),
            "share_beating_pinnacle": round(float((c > 0).mean()), 4)}


def _split(league: str, test_seasons: int):
    df, meta = fd.load_results(league, seasons=test_seasons + WARMUP_SEASONS)
    season = _season(df.Date)
    first_test = season.max() - test_seasons + 1
    test = df[season >= first_test]
    if test.empty or season.min() >= first_test:
        raise ValueError("Histórico insuficiente para separar treino e teste")
    return df, test, meta


def _walk(df: pd.DataFrame, test: pd.DataFrame, xg_weight: float) -> list[dict]:
    preds, x0 = [], None
    window = pd.Timedelta(days=365 * settings.seasons_back)  # same amount of history as the live model
    for week_start, week in test.groupby(test.Date.dt.to_period("W").dt.start_time):
        train = df[df.Date >= week_start - window]
        model = dixon_coles.fit(train, settings.time_decay_xi, until=week_start, x0=x0, xg_weight=xg_weight)
        x0 = dixon_coles.params(model)
        for _, row in week.iterrows():
            h, a = row.HomeTeam, row.AwayTeam
            if not (model.knows(h) and model.knows(a)):
                continue
            conf = mk.confidence(model.matches[h], model.matches[a])
            if conf is None:
                continue
            preds.append({"date": row.Date, "season": int(_season(pd.Series([row.Date]))[0]), "key": (h, a, row.Date),
                          "match": f"{h} - {a}", "model": mk.market_probs(model.score_matrix(h, a)),
                          "odds": fd.row_odds(row), "pre": _fair(row, False), "closing": _fair(row, True),
                          "pinnacle": _pinnacle_fair(row),
                          "conf": conf, "hg": int(row.FTHG), "ag": int(row.FTAG)})
    return preds


def _arrays(preds: list[dict], market: str, source: str):
    """(probs, one-hot outcomes) for predictions that have `source` for this market."""
    rows = [x for x in preds if market in x[source]]
    p = np.array([[x[source][market][s] for s in SELS[market]] for x in rows]).reshape(-1, len(SELS[market]))
    y = np.array([[outcome(market, s, x["hg"], x["ag"]) for s in SELS[market]] for x in rows], float)
    return p, y.reshape(-1, len(SELS[market]))


def fit_weights(preds: list[dict]) -> dict:
    out = {}
    for market in blend.MARKETS:
        rows = [x for x in preds if market in x["pre"]]
        if len(rows) < BLEND_MIN_PREDICTIONS:
            continue
        pm, y = _arrays(rows, market, "model")
        pk, _ = _arrays(rows, market, "pre")
        out[market] = blend.fit(pm, pk, y)
    return out


def _bets(preds: list[dict]) -> pd.DataFrame:
    bets = []
    for x in preds:
        for r in evaluate_markets(x["final"], x["odds"], x["conf"]):
            if r["category"] != mk.VALOR:
                continue
            won = outcome(r["market"], r["selection"], x["hg"], x["ag"])
            fair_close = x["closing"].get(r["market"], {}).get(r["selection"])
            fair_pin = x.get("pinnacle", {}).get(r["market"], {}).get(r["selection"])
            bets.append({"date": x["date"].date().isoformat(), "match": x["match"], "score": f"{x['hg']}-{x['ag']}",
                         "market": r["market"], "selection": r["selection"], "odds": r["best_odds"],
                         "bookmaker": r["bookmaker"], "model_prob": r["model_prob"], "ev": r["ev"], "won": won,
                         "profit": r["best_odds"] - 1 if won else -1.0,
                         "clv": r["best_odds"] * fair_close - 1 if fair_close else np.nan,
                         "clv_pinnacle": r["best_odds"] * fair_pin - 1 if fair_pin else np.nan})
    return pd.DataFrame(bets)


def run_all(leagues: list[str], test_seasons: int = TEST_SEASONS, xg_weight: float | None = None) -> list[dict]:
    """Backtest every league, validate the combination season by season, save reports and the live weights."""
    xg_weight = settings.xg_weight if xg_weight is None else xg_weight
    per_league = {}
    for lg in leagues:
        df, test, meta = _split(lg, test_seasons)
        per_league[lg] = (_walk(df, test, xg_weight), meta)
        print(f"{lg}: {len(per_league[lg][0])} previsões", flush=True)
    every = [x for preds, _ in per_league.values() for x in preds]
    seasons = sorted({x["season"] for x in every})
    weights_by_season = {s: fit_weights([x for x in every if x["season"] < s]) for s in seasons[1:]}
    for x in every:
        w = weights_by_season.get(x["season"])
        x["final"] = blend.apply(x["model"], x["odds"], w)[0] if w else None
    if not any(x["final"] is not None for x in every):
        raise ValueError("Épocas insuficientes para validar a combinação fora da amostra")

    final = {"weights": fit_weights(every), "fitted_on": {"leagues": leagues, "seasons": seasons,
                                                         "predictions": len(every)},
             "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "out_of_sample_by_season": {str(s): w for s, w in weights_by_season.items()}}
    blend.save(final)
    reports = []
    for lg, (preds, meta) in per_league.items():
        evaluated = [x for x in preds if x["final"] is not None]
        if evaluated:
            reports.append(_report(lg, test_seasons, evaluated, _bets(evaluated), meta, xg_weight))
            save(reports[-1])
    reports.append(_report("TODAS", test_seasons, [x for x in every if x["final"] is not None],
                           _bets([x for x in every if x["final"] is not None]),
                           {"source": fd.SOURCE, "last_result": max(m["last_result"] for _, m in per_league.values())},
                           xg_weight))
    save(reports[-1])
    return reports


def _report(league: str, test_seasons: int, preds: list[dict], bets: pd.DataFrame, meta: dict,
            xg_weight: float) -> dict:
    accuracy, calib = {}, []
    for market in SELS:
        p, y = _arrays(preds, market, "final")
        entry = {"model": _scores(p, y)}
        both = [x for x in preds if market in x["closing"] and market in x["pre"]]
        if both:  # same matches for every source, otherwise the comparison is meaningless
            for name, src in (("model_same_matches", "final"), ("dixon_coles_only", "model"),
                              ("market_prematch_no_vig", "pre"), ("market_closing_no_vig", "closing")):
                entry[name] = _scores(*_arrays(both, market, src))
        accuracy[market] = entry
        calib.append((p.ravel(), y.ravel()))
    p_all, y_all = np.concatenate([c[0] for c in calib]), np.concatenate([c[1] for c in calib])
    bins = np.minimum((p_all * 10).astype(int), 9)
    calibration = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "predicted": round(float(p_all[bins == b].mean()), 4),
                    "observed": round(float(y_all[bins == b].mean()), 4), "n": int((bins == b).sum())}
                   for b in range(10) if (bins == b).any()]
    by_season = {}
    for s in sorted({x["season"] for x in preds}):
        sp = [x for x in preds if x["season"] == s and "1X2" in x["closing"] and "1X2" in x["pre"]]
        if sp:
            by_season[f"{s}/{(s + 1) % 100:02d}"] = {
                "n": len(sp), **{k: _scores(*_arrays(sp, "1X2", src))["log_loss"] for k, src in
                                 (("final", "final"), ("dixon_coles", "model"), ("prematch", "pre"),
                                  ("closing", "closing"))}}

    betting = {"overall": _bet_stats(bets)}
    if not bets.empty:
        betting["by_market"] = {m: _bet_stats(g) for m, g in bets.groupby("market")}
        betting["equity"] = [round(float(v), 2) for v in bets.profit.cumsum()]
    warnings = ["Resultados históricos simulados não garantem desempenho futuro."]
    o = betting["overall"]
    if o["bets"] < 300:
        warnings.append(f"Amostra de {o['bets']} apostas é pequena: conclusões sobre rentabilidade não são fiáveis.")
    if o.get("yield_ci95") and o["yield_ci95"][0] <= 0:
        warnings.append("O intervalo de confiança de 95% do yield inclui zero ou valores negativos: "
                        "não há evidência estatística de rentabilidade.")
    a = accuracy["1X2"]
    if "market_closing_no_vig" in a and a["model_same_matches"]["log_loss"] > a["market_closing_no_vig"]["log_loss"]:
        warnings.append("O modelo é menos preciso do que as odds de fecho sem margem (log loss 1X2): "
                        "vantagens aparentes são provavelmente ruído ou erro do modelo.")
    return {"league": league, "competition": fd.LEAGUE_NAMES.get(league, "Todas as ligas"),
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": {"test_seasons": test_seasons, "xi": settings.time_decay_xi, "xg_weight": xg_weight,
                       "devig": settings.devig_method,
                       "ev_margin": settings.ev_margin, "max_market_divergence": settings.max_market_divergence,
                       "min_team_matches": settings.min_team_matches,
                       "blend": "pool logarítmico modelo × mercado pré-jogo; pesos de cada época ajustados só com "
                                "épocas anteriores (todas as ligas); 1.ª época de teste usada só para treino",
                       "odds": "melhor odd pré-jogo entre casas nomeadas (sem Betfair Exchange/agregados); stake fixa 1u"},
            "period": {"from": preds[0]["date"].date().isoformat(), "to": preds[-1]["date"].date().isoformat()},
            "predictions": len(preds), "accuracy": accuracy, "by_season": by_season, "calibration": calibration,
            "betting": betting,
            "bets": bets.astype(object).where(bets.notna(), None).to_dict("records")
            if not bets.empty and league != "TODAS" else [],
            "data": {"source": meta["source"], "last_result": meta["last_result"]}, "warnings": warnings}


DEVIG_METHODS = ("proportional", "power", "shin")
ODDS_BANDS = [1, 2, 3, 5, 10, 1000]


def clv_robustness(leagues: list[str], test_seasons: int = TEST_SEASONS) -> dict:
    """Pinnacle CLV of the saved simulated bets under each margin-removal method, overall and by odds band.
    Re-reads each bet's original CSV row, so the backtest itself does not need re-running."""
    rows = []
    for lg in leagues:
        path = settings.data_dir / "backtests" / f"{lg}.json"
        if not path.exists():
            continue
        df, _ = fd.load_results(lg, seasons=test_seasons + WARMUP_SEASONS)
        index = {(r.Date.date().isoformat(), r.HomeTeam, r.AwayTeam): r for _, r in df.iterrows()}
        for b in json.loads(path.read_text(encoding="utf-8"))["bets"]:
            home, away = b["match"].split(" - ", 1)
            row = index.get((b["date"], home, away))
            if row is None:
                continue
            clv = {}
            for m in DEVIG_METHODS:
                fair = _pinnacle_fair(row, m).get(b["market"], {}).get(b["selection"])
                clv[m] = b["odds"] * fair - 1 if fair else np.nan
            rows.append({"league": lg, "odds": b["odds"], **clv})
    t = pd.DataFrame(rows).dropna()
    if t.empty:
        raise ValueError("Sem apostas com odds de fecho da Pinnacle: corra primeiro o backtest")
    t["band"] = pd.cut(t.odds, ODDS_BANDS).astype(str)

    def summary(g: pd.DataFrame) -> dict:
        return {"n": len(g), **{m: {"mean": round(float(g[m].mean()), 4), "median": round(float(g[m].median()), 4),
                                    "ci95": _bootstrap_ci(g[m].to_numpy()),
                                    "share_positive": round(float((g[m] > 0).mean()), 4)} for m in DEVIG_METHODS}}
    return {"overall": summary(t), "by_odds_band": {b: summary(g) for b, g in t.groupby("band", sort=False)},
            "by_league": {lg: summary(g) for lg, g in t.groupby("league")}}


def _match_log_loss(x: dict, source: str) -> float:
    p = x["model"]["1X2"] if source == "model" else x["closing"]["1X2"]
    sel = "1" if x["hg"] > x["ag"] else "X" if x["hg"] == x["ag"] else "2"
    return float(-np.log(max(p[sel], 1e-12)))


def compare_xg(leagues: list[str], xg_weight: float, test_seasons: int = 1) -> dict:
    """Paired comparison on the same matches: goals-only model vs xG-blended model vs closing market (1X2 log loss).
    Differences are per match, so the bootstrap CI reflects only the change between the models."""
    rows = []
    for lg in leagues:
        df, test, _ = _split(lg, test_seasons)
        base, xg = _walk(df, test, 0.0), _walk(df, test, xg_weight)
        xg_by = {p["key"]: p for p in xg}
        for p in base:
            q = xg_by.get(p["key"])
            if q and "1X2" in p["closing"]:
                rows.append({"league": lg, "base": _match_log_loss(p, "model"), "xg": _match_log_loss(q, "model"),
                             "market": _match_log_loss(p, "market")})
    t = pd.DataFrame(rows)

    def summary(g: pd.DataFrame) -> dict:
        return {"n": len(g), "log_loss_goals": round(g.base.mean(), 4), "log_loss_xg": round(g.xg.mean(), 4),
                "log_loss_market": round(g.market.mean(), 4),
                "xg_minus_goals": round((g.xg - g.base).mean(), 4),
                "xg_minus_goals_ci95": _bootstrap_ci((g.xg - g.base).to_numpy()),
                "xg_minus_market": round((g.xg - g.market).mean(), 4),
                "xg_minus_market_ci95": _bootstrap_ci((g.xg - g.market).to_numpy())}
    return {"xg_weight": xg_weight, "pooled": summary(t), "by_league": {lg: summary(g) for lg, g in t.groupby("league")}}


def save(result: dict) -> None:
    path = settings.data_dir / "backtests" / f"{result['league']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


def load_all() -> list[dict]:
    folder = settings.data_dir / "backtests"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("leagues", nargs="*", default=backtestable())
    ap.add_argument("--test-seasons", type=int, default=TEST_SEASONS)
    ap.add_argument("--compare-xg", type=float, metavar="W",
                    help="compara o modelo só-golos com o modelo com peso W de xG (não grava nada)")
    ap.add_argument("--clv-robustness", action="store_true",
                    help="CLV vs fecho Pinnacle das apostas guardadas com 3 métodos de remoção de margem")
    args = ap.parse_args()
    if args.clv_robustness:
        print(json.dumps(clv_robustness(args.leagues, args.test_seasons), indent=1))
        raise SystemExit
    if args.compare_xg is not None:
        print(json.dumps(compare_xg(args.leagues, args.compare_xg, args.test_seasons), indent=1))
        raise SystemExit
    for res in run_all(args.leagues, args.test_seasons):
        a, o = res["accuracy"]["1X2"], res["betting"]["overall"]
        print(res["league"], res["period"], res["predictions"], {k: v["log_loss"] for k, v in a.items() if k != "model"},
              {k: o.get(k) for k in ("bets", "yield", "yield_ci95", "avg_clv")}, sep=" | ")
