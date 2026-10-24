"""Pre-match report per fixture (sections A-E) built only from fetched data; missing data is reported, never filled in."""
import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import db, markets as mk
from app.config import settings
from app.data import football_data as fd, odds_api, odds_api_io
from app.models import blend, dixon_coles

MODEL_VERSION = "dixon-coles-v1"
EXCLUDED_FROM_BEST = set(fd.AGGREGATES.values()) | {"Betfair Exchange"}  # not bettable as-is / pre-commission
MEMO_SECONDS = 300
STAT_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HxG", "AxG"]
_models: dict[str, tuple] = {}
_validation_cache: dict[tuple, dict] = {}
_lock = threading.Lock()


def get_model(league: str):
    """Fitted model + results for a league. Results are re-read at most every MEMO_SECONDS; refit only on change."""
    with _lock:
        hit = _models.get(league)
    if hit and time.time() - hit[0] < MEMO_SECONDS:
        return hit[2], hit[3], hit[4]
    df, meta = fd.load_results(league)
    df = df[[c for c in STAT_COLS if c in df.columns]]  # the 100+ odds columns make row iteration slow
    sig = (len(df), meta["last_result"])
    model = hit[2] if hit and hit[1] == sig else dixon_coles.fit(df, settings.time_decay_xi,
                                                                    xg_weight=settings.xg_weight)
    with _lock:
        _models[league] = (time.time(), sig, model, df, meta)
    return model, df, meta


def _iso(ts) -> str | None:
    return None if pd.isna(ts) else ts.isoformat()


def _form(df: pd.DataFrame, team: str, before, n: int = 5, venue: str | None = None) -> dict:
    d = df[df.Date < before]
    if venue == "home":
        d = d[d.HomeTeam == team]
    elif venue == "away":
        d = d[d.AwayTeam == team]
    else:
        d = d[(d.HomeTeam == team) | (d.AwayTeam == team)]
    d = d.tail(n)
    if d.empty:
        return {"matches": 0}
    home = d.HomeTeam == team
    gf = np.where(home, d.FTHG, d.FTAG)
    ga = np.where(home, d.FTAG, d.FTHG)
    res = "".join("V" if f > a else "E" if f == a else "D" for f, a in zip(gf, ga))
    out = {"matches": len(d), "form": res, "points_per_game": round((res.count("V") * 3 + res.count("E")) / len(d), 2),
           "goals_for": round(gf.mean(), 2), "goals_against": round(ga.mean(), 2),
           "games": [{"date": r.Date.date().isoformat(), "home": r.HomeTeam, "away": r.AwayTeam,
                      "score": f"{int(r.FTHG)}-{int(r.FTAG)}"} for r in d.itertuples()]}
    if {"HxG", "AxG"} <= set(d.columns) and d[["HxG", "AxG"]].notna().all(axis=None):
        out["xg_for"] = round(np.where(home, d.HxG, d.AxG).mean(), 2)
        out["xg_against"] = round(np.where(home, d.AxG, d.HxG).mean(), 2)
    return out


def _h2h(df: pd.DataFrame, home: str, away: str, before, n: int = 6) -> list[dict]:
    d = df[(df.Date < before) & (((df.HomeTeam == home) & (df.AwayTeam == away)) |
                                 ((df.HomeTeam == away) & (df.AwayTeam == home)))].tail(n)
    return [{"date": r.Date.date().isoformat(), "home": r.HomeTeam, "away": r.AwayTeam,
             "score": f"{int(r.FTHG)}-{int(r.FTAG)}"} for r in d.itertuples()]


def _rest_days(df: pd.DataFrame, team: str, when) -> int | None:
    d = df[(df.Date < when) & ((df.HomeTeam == team) | (df.AwayTeam == team))]
    return None if d.empty else int((when - d.Date.max()).days)


def _best(books: dict[str, dict[str, float]], sel: str) -> tuple[float | None, str | None, bool]:
    """Best price, preferring SRIJ-licensed books: only those can legally take bets from Portugal."""
    cands = [(s[sel], b) for b, s in books.items() if b not in EXCLUDED_FROM_BEST and sel in s]
    pt = [c for c in cands if c[1] in odds_api_io.bookmakers()]
    if pt:
        return (*max(pt), True)
    return (*max(cands), False) if cands else (None, None, False)


def outcome(market: str, sel: str, hg: int, ag: int) -> bool:
    if market == "1X2":
        return {"1": hg > ag, "X": hg == ag, "2": hg < ag}[sel]
    if market == "BTTS":
        return (hg > 0 and ag > 0) == (sel == "Sim")
    line = float(market[2:])
    return (hg + ag > line) == (sel == "Mais")


def evaluate_markets(probs: dict, odds: dict, conf: str | None) -> list[dict]:
    """One row per market selection with every number the decision depends on."""
    rows = []
    for market, sels in probs.items():
        books = odds.get(market, {})
        avg = books.get(fd.AGGREGATES["Avg"])
        nv_source = avg if avg and len(avg) == len(sels) else None
        nv, avg_margin = mk.devig(nv_source) if nv_source else (None, None)
        for sel, p in sels.items():
            best, bookie, licensed = _best(books, sel)
            margin = mk.devig(books[bookie])[1] if bookie and len(books[bookie]) == len(sels) else None
            market_p = nv[sel] if nv else None
            cat, reasons = mk.classify(p if conf else None, best, conf, market_p)
            if best and not licensed:
                reasons.append("odd de referência de operador sem licença SRIJ — não disponível legalmente em Portugal")
            best_ev = mk.ev(p, best) if best and conf else None  # no EV without a reliable estimate
            rows.append({
                "market": market, "selection": sel, "model_prob": round(p, 4),
                "fair_odds": round(mk.fair_odds(p), 2) if p > 0 else None,
                "best_odds": best, "bookmaker": bookie, "licensed_pt": licensed,
                "implied_prob": round(mk.implied(best), 4) if best else None,
                "market_prob_no_vig": round(market_p, 4) if market_p else None,
                "bookmaker_margin": round(margin, 4) if margin is not None else None,
                "market_avg_margin": round(avg_margin, 4) if avg_margin is not None else None,
                "ev": round(best_ev, 4) if best_ev is not None else None,
                "confidence": conf, "category": cat, "reasons": reasons})
    return rows


def validation(league: str) -> dict:
    """Out-of-sample evidence for this league. Without it, no selection is labelled VALOR POTENCIAL."""
    path = settings.data_dir / "backtests" / f"{league}.json"
    if not path.exists():
        return {"validated": False, "reason": "sem backtest para esta liga — modelo não validado"}
    key = (league, path.stat().st_mtime)
    if key not in _validation_cache:
        bt = json.loads(path.read_text(encoding="utf-8"))
        a = bt["accuracy"]["1X2"]
        model, market = a["model_same_matches"]["log_loss"], (a.get("market_closing_no_vig") or {}).get("log_loss")
        o = bt["betting"]["overall"]
        # Pinnacle's close is the sharp benchmark; the market average is beaten by line shopping alone
        clv = o.get("avg_clv_pinnacle") if o.get("clv_pinnacle_n") else o.get("avg_clv")
        accurate = market is not None and model <= market
        # beating the closing line is the stricter test: equal log loss alone is noise
        ok = accurate and clv is not None and clv > 0
        _validation_cache[key] = {
            "validated": ok, "model_log_loss": model, "market_log_loss": market, "avg_clv": clv,
            "reason": "backtest: modelo tão preciso como as odds de fecho e com CLV positivo" if ok else
                      f"backtest nesta liga: {'modelo menos preciso do que as odds de fecho' if not accurate else 'CLV médio não positivo'}"
                      " — vantagem não demonstrada"}
    return _validation_cache[key]


def _apply_validation(rows: list[dict], v: dict) -> None:
    for r in rows:
        if r["category"] == mk.VALOR and not v["validated"]:
            r["category"] = mk.INCONCLUSIVO
            r["reasons"].append(v["reason"])


def _fixture_odds(row, league: str, fx_meta: dict) -> tuple[dict, list[dict]]:
    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(fx_meta["fetched_at"])).total_seconds() / 3600
    too_old = age_h > settings.odds_max_age_hours
    odds = {} if too_old else fd.row_odds(row)
    sources = [{"source": fd.SOURCE, "fetched_at": fx_meta["fetched_at"],
                "note": f"odds descartadas: cache com {age_h:.0f}h" if too_old else
                        "odds pré-jogo recolhidas pelo football-data.co.uk; hora exata de recolha não publicada"}]
    if odds_api.enabled():
        live, last = odds_api.match_odds(odds_api.fetch(league), row["HomeTeam"], row["AwayTeam"])
        for market, books in live.items():
            for b, s in books.items():
                odds.setdefault(market, {})[f"{b} · live"] = s
        sources.append({"source": "the-odds-api.com", "fetched_at": last,
                        "note": "sem correspondência de equipas" if not live else "odds ao vivo"})
    if odds_api_io.enabled():
        pt, last = odds_api_io.match_odds(row["HomeTeam"], row["AwayTeam"], row["Date"])
        for market, books in pt.items():
            odds.setdefault(market, {}).update(books)
        sources.append({"source": "odds-api.io", "fetched_at": last,
                        "note": "operadores licenciados SRIJ" if pt else "jogo não encontrado nos operadores portugueses"})
    return odds, sources


def pick_label(market: str, sel: str, home: str, away: str, sport: str = "Futebol") -> str:
    """Human wording for a selection: team names instead of 1/X/2, "Mais de 2.5 golos"."""
    if market in ("1X2", "Vencedor"):
        return {"1": f"Vitória {home}", "X": "Empate", "2": f"Vitória {away}"}.get(sel, sel)
    if market == "BTTS":
        return "Ambas marcam" if sel == "Sim" else "Não marcam ambas"
    if market.startswith("OU"):
        unit = {"Futebol": "golos", "Hóquei no gelo": "golos", "Andebol": "golos", "Basquetebol": "pontos",
                "Futebol americano": "pontos", "Futebol australiano": "pontos", "Râguebi": "pontos",
                "Râguebi (league)": "pontos", "Ténis": "jogos", "Basebol": "corridas", "Críquete": "corridas",
                "MMA": "rondas", "Boxe": "rondas", "Voleibol": "pontos"}.get(sport, "")
        return f"{sel} de {market[2:]} {unit}".strip()
    return f"{market} {sel}"


def recommend(rows: list[dict], home: str, away: str, sport: str = "Futebol") -> dict:
    """The answer to "what should I bet on here?": a pick only when the value rules are met; otherwise an explicit
    no-bet with the option closest to value and why it falls short. The most likely outcome is shown apart, because a
    likely outcome is not a good bet unless the price pays more than its probability."""
    def label(r):
        return pick_label(r["market"], r["selection"], home, away, sport)
    priced = [r for r in rows if r["ev"] is not None]
    value = [r for r in priced if r["category"] == mk.VALOR]
    main = [r for r in rows if r["market"] in ("1X2", "Vencedor")]
    likely = max(main, key=lambda r: r["model_prob"]) if main else None
    out = {"most_likely": likely and {"label": label(likely), "prob": likely["model_prob"]}}
    if value:
        b = max(value, key=lambda r: r["ev"])
        return {**out, "kind": "valor", "label": label(b), "odds": b["best_odds"], "bookmaker": b["bookmaker"],
                "ev": b["ev"], "prob": b["model_prob"],
                "text": f"Aposta com valor potencial: {label(b)} @ {b['best_odds']} ({b['bookmaker']}), EV estimado "
                        f"{b['ev']:+.1%}. É uma estimativa, não uma garantia: use stakes pequenas."}
    if priced:
        b = max(priced, key=lambda r: r["ev"])
        # the classification reason, not the EV figure itself or the licensing note appended after it
        main_reasons = [x for x in b["reasons"] if not x.startswith("EV estimado") and "SRIJ" not in x]
        why = (main_reasons or b["reasons"] or [""])[0]
        return {**out, "kind": "nenhuma", "label": label(b), "odds": b["best_odds"], "bookmaker": b["bookmaker"],
                "ev": b["ev"], "prob": b["model_prob"],
                "text": f"Nenhuma aposta recomendada neste jogo. A opção mais perto de ter valor é {label(b)} "
                        f"@ {b['best_odds']} ({b['bookmaker']}), com EV estimado {b['ev']:+.1%}, mas não cumpre os "
                        f"critérios ({why})."}
    return {**out, "kind": "sem_dados", "text": "Sem odds ou dados suficientes: não há recomendação para este jogo."}


def _conclusion(rows: list[dict], conf: str | None, notes: list[str]) -> dict:
    value = [r for r in rows if r["category"] == mk.VALOR]
    no_value = sorted({f"{r['market']} {r['selection']}" for r in rows if r["category"] == mk.SEM_VALOR})
    pro, contra = [], []
    for r in value:
        pro.append(f"{r['market']} {r['selection']}: modelo {r['model_prob']:.1%} vs implícita {r['implied_prob']:.1%} "
                   f"@ {r['best_odds']} ({r['bookmaker']}), EV {r['ev']:+.1%}")
        if r["market_prob_no_vig"] is not None:
            contra.append(f"{r['market']} {r['selection']}: o mercado sem margem dá {r['market_prob_no_vig']:.1%}; "
                          "uma diferença destas é frequentemente erro do modelo, não do mercado")
    if value:
        summary = (f"{len(value)} mercado(s) com valor potencial estimado. Estimativas, não garantias: "
                   "confirme odds, lesões e onzes antes de qualquer decisão.")
    elif conf is None:
        summary = "Dados insuficientes para uma análise fiável deste jogo. Recomendação: não apostar."
    else:
        summary = "O modelo não demonstra vantagem em nenhum mercado analisado. Recomendação: não apostar."
    return {"summary": summary, "for": pro, "against": contra, "uncertainties": notes,
            "markets_without_value": no_value, "recommend_no_bet": not value}


def analyze(row, fx_meta: dict, log: bool = True) -> dict:
    league, home, away = row["Div"], row["HomeTeam"], row["AwayTeam"]
    when = row["Date"]
    key = fd.match_key(league, when, home, away)
    report = {"key": key, "identification": {
        "sport": "Futebol", "league": league, "competition": fd.LEAGUE_NAMES.get(league, league),
        "home": home, "away": away, "date": when.date().isoformat(),
        "time": row.get("Time") if isinstance(row.get("Time"), str) else None,
        "kickoff": _iso(fd.kickoffs(pd.DataFrame([row])).iloc[0]),
        "venue": f"Casa de {home} (estádio não indicado pela fonte)"}}
    notes = ["Lesões, suspensões e onzes prováveis: sem fonte integrada no MVP — verificar manualmente.",
             "Só jogos do campeonato contam; taças e competições europeias não entram no modelo nem no descanso.",
             "Meteorologia não considerada."]
    try:
        model, df, meta = get_model(league)
    except (RuntimeError, ValueError) as e:
        report.update(data_status={"error": str(e)}, statistics=None, predictions=None, markets=[],
                      conclusion=_conclusion([], None, notes + [str(e)]))
        return report

    known = model.knows(home) and model.knows(away)
    conf = mk.confidence(model.matches.get(home, 0), model.matches.get(away, 0)) if known else None
    odds, odds_sources = _fixture_odds(row, league, fx_meta)
    if meta["stale"] or fx_meta["stale"]:
        notes.append("A fonte de dados estava indisponível; foram usados dados em cache possivelmente desatualizados.")
    if not known:
        notes.append("Pelo menos uma equipa não tem histórico nesta liga (ex.: recém-promovida).")
    elif conf is None:
        notes.append(f"Menos de {settings.min_team_matches} jogos de histórico para pelo menos uma equipa.")
    report["data_status"] = {
        "results": {"source": meta["source"], "last_result": meta["last_result"], "matches": meta["matches"],
                    "fetched_at": max(f["fetched_at"] for f in meta["files"])},
        "odds": odds_sources, "team_matches": {home: model.matches.get(home, 0), away: model.matches.get(away, 0)},
        "confidence": conf, "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    report["statistics"] = {
        "form": {home: _form(df, home, when), away: _form(df, away, when)},
        "home_away": {home: _form(df, home, when, 10, "home"), away: _form(df, away, when, 10, "away")},
        "h2h": _h2h(df, home, away, when),
        "rest_days": {home: _rest_days(df, home, when), away: _rest_days(df, away, when)},
        "ratings": ({t: {"attack": round(float(np.exp(model.attack[model.teams.index(t)])), 3),
                         "defence": round(float(np.exp(model.defence[model.teams.index(t)])), 3)}
                     for t in (home, away)} if known else None),
        "home_advantage": round(float(np.exp(model.home_adv)), 3)}
    if conf is None:  # unknown team or too few matches: no forecast rather than an unreliable one
        report.update(predictions=None, markets=[], conclusion=_conclusion([], None, notes))
        return report

    m = model.score_matrix(home, away)
    lam, mu = model.expected_goals(home, away)
    model_probs = mk.market_probs(m)
    bw = blend.load()
    probs, blended = blend.apply(model_probs, odds, bw and bw["weights"])
    top = sorted(((float(m[i, j]), f"{i}-{j}") for i in range(7) for j in range(7)), reverse=True)[:10]
    report["predictions"] = {
        "expected_goals": {home: round(lam, 2), away: round(mu, 2)}, "probabilities": probs,
        "model_probabilities": model_probs,
        "blend": {"markets": blended, "weights": {k: bw["weights"][k] for k in blended}} if blended else None,
        "top_scores": [{"score": s, "prob": round(p, 4)} for p, s in top],
        "matrix": np.round(m[:6, :6], 4).tolist(),
        "method": "Dixon-Coles com decaimento temporal (ξ=%.4f/dia), peso do xG %.0f%%, treinado em %d jogos até %s"
                  % (settings.time_decay_xi, settings.xg_weight * 100, meta["matches"], meta["last_result"])}
    rows = evaluate_markets(probs, odds, conf)
    v = validation(league)
    _apply_validation(rows, v)
    report["data_status"]["model_validation"] = v
    report["markets"] = rows
    report["odds"] = odds
    report["conclusion"] = _conclusion(rows, conf, notes)
    report["recommendation"] = recommend(rows, home, away)
    if log:
        db.log_predictions([{"match_key": key, "league": league, "match_date": when.date().isoformat(),
                             "home": home, "away": away, "market": r["market"], "selection": r["selection"],
                             "model_prob": r["model_prob"], "odds": r["best_odds"], "ev": r["ev"],
                             "category": r["category"], "confidence": conf, "model_version": MODEL_VERSION}
                            for r in rows])
        db.snapshot_odds(key, odds, odds_sources[-1]["fetched_at"] or fx_meta["fetched_at"])
    return report


def list_fixtures(league: str | None = None) -> tuple[list[dict], dict]:
    fx, meta = fd.load_fixtures()
    fx = fx[fx.Div.isin(settings.league_list)]
    if league:
        fx = fx[fx.Div == league]
    odds_api_io.prefetch([r for _, r in fx.iterrows()])
    return [summarize(analyze(row, meta)) for _, row in fx.iterrows()], meta


def summarize(r: dict) -> dict:
    """One fixture-list row from a full report."""
    best = max((x for x in r["markets"] if x["ev"] is not None), key=lambda x: x["ev"], default=None)
    cats = [x["category"] for x in r["markets"]]
    top = next((c for c in (mk.VALOR, mk.INCONCLUSIVO, mk.SEM_VALOR) if c in cats), mk.SEM_DADOS)
    rec = r.get("recommendation") or {}
    return {"key": r["key"], **r["identification"], "category": top,
            "pick": {k: rec[k] for k in ("label", "odds", "ev")} if rec.get("kind") == "valor" else None,
            "confidence": (r.get("data_status") or {}).get("confidence"),
            "probs_1x2": ((r["predictions"] or {}).get("probabilities", {}).get("1X2")
                          or (r["predictions"] or {}).get("probabilities", {}).get("Vencedor")),
            "best": best and {k: best[k] for k in ("market", "selection", "best_odds", "ev", "category")},
            "value_count": cats.count(mk.VALOR)}


def find_fixture(key: str):
    fx, meta = fd.load_fixtures()
    for _, row in fx.iterrows():
        if fd.match_key(row["Div"], row["Date"], row["HomeTeam"], row["AwayTeam"]) == key:
            return row, meta
    return None, meta


def settled_history(limit: int = 500) -> list[dict]:
    """Logged predictions with the real outcome once the match is in the results feed."""
    rows = db.history(limit)
    results = {}
    for lg in {r["league"] for r in rows if "_" not in r["league"]}:  # other sports (basketball_nba…) have no results feed yet
        try:
            df, _ = fd.load_results(lg)
        except RuntimeError:
            continue
        for x in df.itertuples():
            results[fd.match_key(lg, x.Date, x.HomeTeam, x.AwayTeam)] = (int(x.FTHG), int(x.FTAG))
    for r in rows:
        score = results.get(r["match_key"])
        r["result"] = f"{score[0]}-{score[1]}" if score else None
        r["won"] = outcome(r["market"], r["selection"], *score) if score else None
    return rows
