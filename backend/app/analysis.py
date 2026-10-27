"""Pre-match report per fixture (sections A-E) built only from fetched data; missing data is reported, never filled in."""
import itertools
import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import db, markets as mk
from app.config import settings
from app.data import football_data as fd, odds_api, odds_api_io, oddspapi, srij
from scipy.optimize import minimize

from app.models import blend, dixon_coles

MODEL_VERSION = "dixon-coles-v1"
EXCLUDED_FROM_BEST = set(fd.AGGREGATES.values()) | {"Betfair Exchange"}  # not bettable as-is / pre-commission
# Betting exchanges and prediction markets quote prices before their 2-5% commission, so their "best odds" overstate EV
EXCHANGES = ("betfair", "matchbook", "smarkets", "betdaq", "polymarket", "kalshi", "novig", "prophetx", "betopenly",
             "limitless")


def _excluded(book: str) -> bool:
    return book in EXCLUDED_FROM_BEST or any(x in book.lower() for x in EXCHANGES)
MEMO_SECONDS = 300
NO_OPPORTUNITY = ("Não foi identificada uma oportunidade com valor estatístico suficientemente convincente nas "
                  "condições atuais.")
STAT_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HxG", "AxG", "HTHG", "HTAG", "HC", "AC", "HY", "AY",
             "HS", "AS", "HST", "AST"]
COUNT_WINDOW_DAYS = 365  # corner and card rates from the last year: squads and refereeing standards change
COUNT_MIN_MATCHES = 5
COUNT_LINES = {"HC": mk.CORNER_LINES, "HY": mk.CARD_LINES, "HS": mk.SHOT_LINES, "HST": mk.SHOT_TARGET_LINES}
COUNT_SHRINK = 10  # team rates pulled to the league mean as if it were 10 extra matches (chosen on 2024/25)
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


def _calibrate(model, lam: float, mu: float, probs: dict) -> tuple[float, float]:
    """Expected goals whose scoreline distribution reproduces the final (model + market) result and over 2.5
    probabilities. Scorelines, team goals, handicaps and half-time then agree with the headline numbers instead of
    showing the model alone (which the backtest gives ~zero weight)."""
    t1, tx, to = probs["1X2"]["1"], probs["1X2"]["X"], probs["OU2.5"]["Mais"]

    def loss(x):
        m = model.matrix(*np.exp(x))
        i, j = np.indices(m.shape)
        return (m[i > j].sum() - t1) ** 2 + (m[i == j].sum() - tx) ** 2 + (m[i + j > 2.5].sum() - to) ** 2
    res = minimize(loss, np.log([lam, mu]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-9})
    return tuple(float(v) for v in np.exp(res.x)) if res.success else (lam, mu)


def _ht_share(df: pd.DataFrame) -> float | None:
    """Share of the league's goals scored before half-time."""
    if not {"HTHG", "HTAG"} <= set(df.columns):
        return None
    d = df.dropna(subset=["HTHG", "HTAG"])
    total = (d.FTHG + d.FTAG).sum()
    return float((d.HTHG + d.HTAG).sum() / total) if len(d) >= 100 and total else None


def _count_rates(df: pd.DataFrame, home: str, away: str, when, hcol: str, acol: str) -> dict | None:
    """Expected count for each side (e.g. cards) from the last year: the home side's own count at home averaged with
    what the away side concedes away, and vice versa, each pulled towards the league mean (COUNT_SHRINK).
    Dispersion = variance / mean of the league's match totals. `league_over` = the league's real frequency per line."""
    if not {hcol, acol} <= set(df.columns):
        return None
    d = df[(df.Date < when) & (df.Date >= when - pd.Timedelta(days=COUNT_WINDOW_DAYS))].dropna(subset=[hcol, acol])
    h_home, a_away = d[d.HomeTeam == home], d[d.AwayTeam == away]
    if len(h_home) < COUNT_MIN_MATCHES or len(a_away) < COUNT_MIN_MATCHES or len(d) < 100:
        return None
    def shrunk(s, prior):
        return (s.sum() + prior * COUNT_SHRINK) / (len(s) + COUNT_SHRINK)
    lh, la = d[hcol].mean(), d[acol].mean()
    eh = (shrunk(h_home[hcol], lh) + shrunk(a_away[hcol], lh)) / 2
    ea = (shrunk(a_away[acol], la) + shrunk(h_home[acol], la)) / 2
    tot = d[hcol] + d[acol]
    return {"home": round(float(eh), 2), "away": round(float(ea), 2), "total": round(float(eh + ea), 2),
            "dispersion": round(max(1.0, float(tot.var() / tot.mean())), 3),
            "matches": {home: len(h_home), away: len(a_away)}, "league_total": round(float(tot.mean()), 2),
            "league_over": {line: float((tot > line).mean()) for line in COUNT_LINES[hcol]}}


def _rest_days(df: pd.DataFrame, team: str, when) -> int | None:
    d = df[(df.Date < when) & ((df.HomeTeam == team) | (df.AwayTeam == team))]
    return None if d.empty else int((when - d.Date.max()).days)


def _best(books: dict[str, dict[str, float]], sel: str) -> tuple[float | None, str | None, bool]:
    """Best price, preferring SRIJ-licensed books: only those can legally take bets from Portugal. With PT_ONLY,
    unlicensed books are never the best price (they still feed the market reference used for probabilities)."""
    cands = [(s[sel], b) for b, s in books.items() if not _excluded(b) and sel in s]
    pt = [c for c in cands if srij.licensed(c[1])]
    if pt:
        return (*max(pt), True)
    if settings.pt_only:
        return None, None, False
    return (*max(cands), False) if cands else (None, None, False)


def public_odds(odds: dict) -> dict:
    """The odds tables shown to visitors: with PT_ONLY, only SRIJ-licensed bookmakers."""
    if not settings.pt_only:
        return odds
    return {m: {b: s for b, s in books.items() if srij.licensed(b)} for m, books in odds.items()}


def public_alerts(moves: list[dict]) -> list[dict]:
    return [a for a in moves if not settings.pt_only or srij.licensed(a["bookmaker"])]


def _res(h: int, a: int) -> str:
    return "1" if h > a else "X" if h == a else "2"


STAT_PAIRS = {"CO": ("HC", "AC"), "CA": ("HY", "AY"), "RE": ("HS", "AS"), "RB": ("HST", "AST")}


def outcome(market: str, sel: str, hg: int, ag: int, stats: dict | None = None) -> bool | None:
    """Whether the selection won; None when the stake is returned (draw no bet on a draw) or the result needs a
    statistic the feed does not have (half-time score, corners, cards, shots in `stats`)."""
    stats = stats or {}
    if market[:2] in STAT_PAIRS:
        h, a = (stats.get(c) for c in STAT_PAIRS[market[:2]])
        return None if h is None or a is None else (h + a > float(market[2:])) == (sel == "Mais")
    if market in ("HT1X2", "HTFT", "2H1X2", "MP") or market.startswith("HTOU"):
        h, a = stats.get("HTHG"), stats.get("HTAG")
        if h is None or a is None:
            return None
        if market == "HT1X2":
            return sel == _res(h, a)
        if market == "HTFT":
            return sel == f"{_res(h, a)}/{_res(hg, ag)}"
        if market == "2H1X2":
            return sel == _res(hg - h, ag - a)
        if market == "MP":
            first, second = h + a, hg + ag - h - a
            return sel == ("1ª parte" if first > second else "2ª parte" if second > first else "Igual")
        return (h + a > float(market[4:])) == (sel == "Mais")
    res = _res(hg, ag)
    if market == "1X2":
        return sel == res
    if market == "DC":
        return res in sel
    if market == "DNB":
        return None if res == "X" else sel == res
    if market == "BTTS":
        return (hg > 0 and ag > 0) == (sel == "Sim")
    if market == "EG":
        return sel == (str(hg + ag) if hg + ag < 4 else "4+")
    if market == "CS":
        return (ag if sel == "1" else hg) == 0
    if market == "WM":
        d = hg - ag
        return sel == ("X" if d == 0 else f"{'1' if d > 0 else '2'} por {abs(d) if abs(d) < 3 else '3+'}")
    if market.startswith("AH"):
        return (hg - ag + float(market[2:]) > 0) == (sel == "1")
    if market[:2] in ("GC", "GF"):
        return ((hg if market[:2] == "GC" else ag) > float(market[2:])) == (sel == "Mais")
    line = float(market[2:])
    return (hg + ag > line) == (sel == "Mais")


def evaluate_markets(probs: dict, odds: dict, conf: str | None) -> list[dict]:
    """One row per market selection with every number the decision depends on."""
    rows = []
    draw = (probs.get("1X2") or probs.get("Vencedor") or {}).get("X")
    for market, sels in probs.items():
        # draw no bet returns the stake on a draw: EV per unit staked is the conditional EV times P(no draw)
        scale = 1 - draw if market == "DNB" and draw is not None else 1
        books = odds.get(market, {})
        avg = books.get(fd.AGGREGATES["Avg"])
        avg_margin = mk.devig(avg)[1] if avg and len(avg) == len(sels) else None
        nv = blend.market_fair(odds, market)  # average price, or the consensus of 2+ bookmakers
        nv = nv if nv and nv.keys() == sels.keys() else None
        for sel, p in sels.items():
            best, bookie, licensed = _best(books, sel)
            margin = mk.devig(books[bookie])[1] if bookie and len(books[bookie]) == len(sels) else None
            market_p = nv[sel] if nv else None
            cat, reasons = mk.classify(p if conf else None, best, conf, market_p)
            if best and not licensed:
                reasons.append("odd de referência de operador sem licença SRIJ — não disponível legalmente em Portugal")
            if best is None and books and settings.pt_only:
                reasons = ["sem odd de casa licenciada pelo SRIJ (o site só usa casas legais em Portugal)"]
            best_ev = mk.ev(p, best) * scale if best and conf else None  # no EV without a reliable estimate
            market_ev = mk.ev(market_p, best) * scale if best and market_p else None
            rows.append({
                "market": market, "selection": sel, "model_prob": round(p, 4),
                "fair_odds": round(mk.fair_odds(p), 2) if p > 0 else None,
                "best_odds": best, "bookmaker": bookie, "licensed_pt": licensed,
                "implied_prob": round(mk.implied(best), 4) if best else None,
                "market_prob_no_vig": round(market_p, 4) if market_p else None,
                "bookmaker_margin": round(margin, 4) if margin is not None else None,
                "market_avg_margin": round(avg_margin, 4) if avg_margin is not None else None,
                "ev": round(best_ev, 4) if best_ev is not None else None,
                # the lower of two estimates (ours and the no-vig market's): the value both agree on
                "ev_conservative": round(min(best_ev, market_ev), 4) if None not in (best_ev, market_ev) else None,
                # lowest price worth taking: fair odds plus the safety margin
                "min_odds": round(mk.fair_odds(p) * (1 + settings.ev_margin), 2) if p > 0 else None,
                "n_books": sum(1 for b, s in books.items() if not _excluded(b) and sel in s),
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


def add_evidence(rows: list[dict], history: str | None, v: dict, model_probs: dict | None) -> None:
    """Evidence quality per selection; model_probs are the model's own (pre-blend) estimates, None without a model."""
    for r in rows:
        model_p = (model_probs or {}).get(r["market"], {}).get(r["selection"])
        r["evidence"] = mk.evidence(r["n_books"], history, v, model_p, r["market_prob_no_vig"])


def rank_score(r: dict) -> float:
    """Ranking of checkable selections: conservative EV, discounted by evidence quality when positive."""
    e = r["ev_conservative"]
    return e * r["evidence"]["score"] / 100 if e > 0 else e


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
    if oddspapi.enabled():
        pt, last = oddspapi.match_odds(row["HomeTeam"], row["AwayTeam"], row["Date"])
        for market, books in pt.items():
            odds.setdefault(market, {}).update({b: dict(s) for b, s in books.items()})
        sources.append({"source": "OddsPapi", "fetched_at": last,
                        "note": "Betano PT (licenciada SRIJ)" if pt else "jogo sem odds da Betano PT"})
    return odds, sources


def pick_label(market: str, sel: str, home: str, away: str, sport: str = "Futebol") -> str:
    """Human wording for a selection: team names instead of 1/X/2, "Mais de 2.5 golos"."""
    if market in ("1X2", "Vencedor"):
        return {"1": f"Vitória {home}", "X": "Empate", "2": f"Vitória {away}"}.get(sel, sel)
    if market == "BTTS":
        return "Ambas marcam" if sel == "Sim" else "Não marcam ambas"
    if market == "DC":
        return {"1X": f"{home} ou empate", "12": f"{home} ou {away}", "X2": f"Empate ou {away}"}.get(sel, sel)
    if market == "DNB":
        return f"{home if sel == '1' else away} (empate anula)"
    if market.startswith("AH"):
        h = float(market[2:])
        return f"{home} {h:+g}" if sel == "1" else f"{away} {-h:+g}"
    if market[:2] in ("GC", "GF"):
        return f"{sel} de {market[2:]} golos — {home if market[:2] == 'GC' else away}"
    if market == "HT1X2":
        return {"1": f"{home} ao intervalo", "X": "Empate ao intervalo", "2": f"{away} ao intervalo"}.get(sel, sel)
    if market.startswith("HTOU"):
        return f"{sel} de {market[4:]} golos ao intervalo"
    if market[:2] in ("CO", "CA", "RE", "RB"):
        unit = {"CO": "cantos", "CA": "cartões amarelos", "RE": "remates", "RB": "remates à baliza"}[market[:2]]
        return f"{sel} de {market[2:]} {unit}"
    if market == "EG":
        return f"{sel} golos no jogo"
    if market == "CS":
        return f"{home if sel == '1' else away} sem sofrer golos"
    if market == "WM":
        return "Empate" if sel == "X" else f"{home if sel[0] == '1' else away} ganha {sel[2:]}"
    if market == "HTFT":
        names = {"1": home, "X": "Empate", "2": away}
        a, b = sel.split("/")
        return f"{names[a]} ao intervalo / {names[b]} no fim"
    if market == "2H1X2":
        return {"1": f"{home} ganha a 2.ª parte", "X": "2.ª parte empatada", "2": f"{away} ganha a 2.ª parte"}.get(sel, sel)
    if market == "MP":
        return {"Igual": "Mesmo número de golos nas duas partes"}.get(sel, f"Mais golos na {sel}")
    if market.startswith("OU"):
        unit = {"Futebol": "golos", "Hóquei no gelo": "golos", "Andebol": "golos", "Basquetebol": "pontos",
                "Futebol americano": "pontos", "Futebol australiano": "pontos", "Râguebi": "pontos",
                "Râguebi (league)": "pontos", "Ténis": "jogos", "Basebol": "corridas", "Críquete": "corridas",
                "MMA": "rondas", "Boxe": "rondas", "Voleibol": "pontos"}.get(sport, "")
        return f"{sel} de {market[2:]} {unit}".strip()
    return f"{market} {sel}"


def recommend(rows: list[dict], home: str, away: str, sport: str = "Futebol") -> dict:
    """The answer to "what should I bet on here?": a pick only when the price pays more than the probability (validated
    value, or positive conservative EV marked as not validated); otherwise an explicit no-bet. The most likely outcome is shown apart, because a
    likely outcome is not a good bet unless the price pays more than its probability."""
    def label(r):
        return pick_label(r["market"], r["selection"], home, away, sport)
    ok = checkable(rows)
    value = [r for r in ok if r["category"] == mk.VALOR]
    main = [r for r in rows if r["market"] in ("1X2", "Vencedor")]
    likely = max(main, key=lambda r: r["model_prob"]) if main else None
    out = {"most_likely": likely and {"label": label(likely), "prob": likely["model_prob"]}}
    if not ok:
        return {**out, "kind": "sem_dados", "verdict": NO_OPPORTUNITY,
                "text": "Sem odds que se possam comparar com o mercado: não há melhor aposta para este jogo."}
    b = max(value or ok, key=rank_score)
    pick = {"label": label(b), "odds": b["best_odds"], "bookmaker": b["bookmaker"], "ev": b["ev_conservative"],
            "ev_model": b["ev"], "prob": b["model_prob"], "licensed_pt": b.get("licensed_pt", False),
            "min_odds": b["min_odds"], "evidence": b["evidence"], "score": round(rank_score(b), 4)}
    if value:
        return {**out, **pick, "kind": "valor",
                "text": f"Aposta com valor potencial: {label(b)} @ {b['best_odds']} ({b['bookmaker']}), EV estimado "
                        f"{pick['ev']:+.1%}. É uma estimativa, não uma garantia: use stakes pequenas."}
    out["verdict"] = NO_OPPORTUNITY
    if pick["ev"] >= settings.ev_margin:  # the price reaches the minimum odds (fair odds + safety margin)
        # the classification reason, not the EV figure itself or the licensing note appended after it
        main_reasons = [x for x in b["reasons"] if not x.startswith("EV estimado") and "SRIJ" not in x]
        why = (main_reasons or b["reasons"] or [""])[0]
        return {**out, **pick, "kind": "melhor_opcao",
                "text": f"Melhor opção: {label(b)} @ {b['best_odds']} ({b['bookmaker']}), EV conservador {pick['ev']:+.1%}, "
                        f"evidência {b['evidence']['label']} ({b['evidence']['score']}/100). "
                        f"Não validada ({why}): é a aposta mais favorável deste jogo, não uma aposta garantida."}
    # no pick at all: the "least bad" option is usually a long shot, and showing it as the game's bet misleads
    likely_txt = f" Resultado mais provável: {out['most_likely']['label']} ({out['most_likely']['prob']:.0%})." if likely else ""
    return {**out, "kind": "sem_valor",
            "text": "Nenhuma aposta com valor: nenhuma odd disponível chega à odd mínima (odd justa + "
                    f"{settings.ev_margin:.0%} de margem de segurança).{likely_txt} Recomendação: não apostar."}


def checkable(rows: list[dict]) -> list[dict]:
    """Selections whose price can be checked against the market: a no-vig market reference exists, the price is not
    a probable data error or stale, and the model estimate is not far from the market. Anything else would offer
    value that is not really there (e.g. +56% EV from one bookmaker's wrong price)."""
    flags = (mk.OUTLIER_TEXT, mk.DIVERGENCE_TEXT)
    return [r for r in rows if r["ev"] is not None and r.get("market_prob_no_vig") is not None
            and not any(f in reason for f in flags for reason in r["reasons"])]


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
    if conf is None:  # unknown team or too few matches: no model forecast; the market's, when there is one
        market = {k: v for k, v in ((mkt, blend.market_fair(odds, mkt)) for mkt in odds) if v}
        if "1X2" in market:
            market.update(mk.derived_markets(market["1X2"]))
        report.update(markets=[], conclusion=_conclusion([], None, notes), predictions={
            "probabilities": market, "model_probabilities": None, "blend": None, "expected_goals": None,
            "matrix": None, "top_scores": [],
            "method": "Sem histórico suficiente de uma das equipas nesta liga: probabilidades do mercado sem margem"}
            if "1X2" in market else None)
        return report

    m = model.score_matrix(home, away)
    lam, mu = model.expected_goals(home, away)
    model_probs = mk.market_probs(m)
    bw = blend.load()
    probs, blended = blend.apply(model_probs, odds, bw and bw["weights"])
    model_lam, model_mu, model_m = lam, mu, m
    if {"1X2", "OU2.5"} <= set(blended):  # scorelines consistent with the final probabilities
        lam, mu = _calibrate(model, lam, mu, probs)
        m = model.matrix(lam, mu)
        # goal lines and BTTS without a market reference follow the calibrated distribution, not the model alone
        probs = {**probs, **{k: v for k, v in mk.market_probs(m).items() if k not in blended}}
    # double chance / draw no bet from the final result; team goals and handicaps from the (calibrated) matrix
    probs = {**probs, **mk.derived_markets(probs["1X2"], m)}
    model_probs = {**model_probs, **mk.derived_markets(model_probs["1X2"], model_m)}
    # half-time, corners and cards: model only (no odds from any source), skipped when the league lacks the data
    extra, share = {}, _ht_share(df)
    extra.update(mk.score_markets(m))
    if share:
        extra.update(mk.halftime_markets(lam, mu, share))
        extra.update(mk.halves_markets(lam, mu, share))
    corners, cards = _count_rates(df, home, away, when, "HC", "AC"), _count_rates(df, home, away, when, "HY", "AY")
    if corners:  # team corner rates did not beat the league frequency out of sample (2025/26): the league's is used
        extra.update({f"CO{line}": {"Mais": p, "Menos": 1 - p} for line, p in corners.pop("league_over").items()})
    # cards and shots: team rates beat the league frequency out of sample (2024/25 and 2025/26)
    shots, on_target = _count_rates(df, home, away, when, "HS", "AS"), _count_rates(df, home, away, when, "HST", "AST")
    for stat, prefix, lines in ((cards, "CA", mk.CARD_LINES), (shots, "RE", mk.SHOT_LINES),
                                (on_target, "RB", mk.SHOT_TARGET_LINES)):
        if stat:
            stat.pop("league_over")
            extra.update(mk.count_markets(prefix, stat["total"], stat["dispersion"], lines))
    probs, model_probs = {**probs, **extra}, {**model_probs, **extra}
    top = sorted(((float(m[i, j]), f"{i}-{j}") for i in range(7) for j in range(7)), reverse=True)[:10]
    report["predictions"] = {
        "expected_goals": {home: round(lam, 2), away: round(mu, 2)}, "probabilities": probs,
        "model_expected_goals": {home: round(model_lam, 2), away: round(model_mu, 2)},
        "model_probabilities": model_probs,
        "blend": {"markets": blended, "weights": {k: blend.weights_for(bw["weights"], k) for k in blended}}
        if blended else None,
        "top_scores": [{"score": s, "prob": round(p, 4)} for p, s in top],
        "halftime": {"goal_share": round(share, 3), "expected_goals": {home: round(lam * share, 2), away: round(mu * share, 2)}}
        if share else None,
        "corners": corners, "cards": cards, "shots": shots, "shots_on_target": on_target,
        "matrix": np.round(m[:6, :6], 4).tolist(),
        "method": "Dixon-Coles com decaimento temporal (ξ=%.4f/dia), peso do xG %.0f%%, treinado em %d jogos até %s"
                  % (settings.time_decay_xi, settings.xg_weight * 100, meta["matches"], meta["last_result"])}
    rows = evaluate_markets(probs, odds, conf)
    v = validation(league)
    _apply_validation(rows, v)
    add_evidence(rows, conf, v, model_probs)
    report["data_status"]["model_validation"] = v
    report["markets"] = rows
    report["odds"] = public_odds(odds)
    report["conclusion"] = _conclusion(rows, conf, notes)
    report["explanation"] = explain(report)
    report["recommendation"] = recommend(rows, home, away)
    if log:
        db.log_predictions([{"match_key": key, "league": league, "match_date": when.date().isoformat(),
                             "home": home, "away": away, "market": r["market"], "selection": r["selection"],
                             "model_prob": r["model_prob"], "odds": r["best_odds"], "ev": r["ev"],
                             "category": r["category"], "confidence": conf, "model_version": MODEL_VERSION}
                            for r in rows])
        db.snapshot_odds(key, odds, db.now())  # when we saw these prices: one source's time would mislabel the others
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
    best = max(checkable(r["markets"]), key=rank_score, default=None)  # same filter and order as the best bet
    cats = [x["category"] for x in r["markets"]]
    top = next((c for c in (mk.VALOR, mk.INCONCLUSIVO, mk.SEM_VALOR) if c in cats), mk.SEM_DADOS)
    rec = r.get("recommendation") or {}
    eg = (r["predictions"] or {}).get("expected_goals")
    return {"key": r["key"], **r["identification"], "category": top,
            "pick": {k: rec.get(k) for k in ("kind", "label", "odds", "ev", "bookmaker", "licensed_pt", "score")}
            if rec.get("label") else None,
            "confidence": (r.get("data_status") or {}).get("confidence"),
            "probs_1x2": ((r["predictions"] or {}).get("probabilities", {}).get("1X2")
                          or (r["predictions"] or {}).get("probabilities", {}).get("Vencedor")),
            "over25": (r["predictions"] or {}).get("probabilities", {}).get("OU2.5", {}).get("Mais"),
            "btts": (r["predictions"] or {}).get("probabilities", {}).get("BTTS", {}).get("Sim"),
            # expected goals (football): the page turns them into live probabilities from the score and the minute
            "xg": list(eg.values()) if eg and r["identification"]["sport"] == "Futebol" else None,
            "best": best and {k: best[k] for k in ("market", "selection", "best_odds", "ev", "category")},
            "value_count": cats.count(mk.VALOR), "opps": opportunities(r)}


def opportunities(r: dict, n: int = 3) -> list[dict]:
    """The game's best-EV selections for the day's ranking. Only selections that can be checked against the market
    (a no-vig market reference exists) and pass both checks: not a probable data error or stale price, and not a
    model estimate far from the market. Those would top the ranking with value that is not really there."""
    ident = r["identification"]
    rows = [x for x in checkable(r["markets"]) if x["ev_conservative"] >= settings.ev_margin]
    rows.sort(key=rank_score, reverse=True)
    return [{"label": pick_label(x["market"], x["selection"], ident["home"], ident["away"], ident["sport"]),
             "odds": x["best_odds"], "bookmaker": x["bookmaker"], "licensed_pt": x.get("licensed_pt", False),
             "prob": x["model_prob"], "ev": x["ev_conservative"], "evidence": x["evidence"]["score"],
             "min_odds": x["min_odds"], "score": round(rank_score(x), 4), "category": x["category"]} for x in rows[:n]]


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
        stat_cols = [c for c in ("HTHG", "HTAG", "HC", "AC", "HY", "AY", "HS", "AS", "HST", "AST") if c in df.columns]
        for x in df.to_dict("records"):
            stats = {c: int(x[c]) for c in stat_cols if pd.notna(x[c])}
            results[fd.match_key(lg, x["Date"], x["HomeTeam"], x["AwayTeam"])] = (int(x["FTHG"]), int(x["FTAG"]), stats)
    for r in rows:
        res = results.get(r["match_key"])
        r["result"] = f"{res[0]}-{res[1]}" if res else None
        r["won"] = outcome(r["market"], r["selection"], *res) if res else None
    return rows


def performance(rows: list[dict]) -> dict:
    """Live track record of the logged predictions once results are in: calibration (log loss, Brier) per market
    group and the simulated result of every positive-EV selection at the logged price. Only the latest logged
    estimate per selection counts, so hourly re-logging does not weigh a game more."""
    latest = {}
    for r in rows:  # rows come newest first
        if r.get("won") is not None:
            latest.setdefault((r["match_key"], r["market"], r["selection"]), r)
    groups: dict[str, list] = {}
    for r in latest.values():
        g = r["market"] if r["market"] in ("1X2", "Vencedor", "OU2.5", "BTTS") else "outros"
        groups.setdefault(g, []).append(r)
        groups.setdefault("todos", []).append(r)
    out = {}
    for g, rs in groups.items():
        p = np.clip([r["model_prob"] for r in rs], 1e-6, 1 - 1e-6)
        y = np.array([bool(r["won"]) for r in rs], dtype=float)
        bets = [r for r in rs if r.get("ev") is not None and r["ev"] > 0 and r.get("odds")]
        profit = sum(r["odds"] - 1 if r["won"] else -1 for r in bets)
        out[g] = {"selections": len(rs), "games": len({r["match_key"] for r in rs}),
                  "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
                  "brier": round(float(((p - y) ** 2).mean()), 4),
                  "bets": len(bets), "profit": round(profit, 2), "roi": round(profit / len(bets), 4) if bets else None}
    calib = []
    allp = [(r["model_prob"], bool(r["won"])) for r in latest.values()]
    for k in range(10):
        b = [(pp, w) for pp, w in allp if min(int(pp * 10), 9) == k]
        if b:
            calib.append({"from": k / 10, "to": (k + 1) / 10, "n": len(b),
                          "predicted": round(float(np.mean([pp for pp, _ in b])), 4),
                          "observed": round(float(np.mean([w for _, w in b])), 4)})
    # how often the most likely result happened: the number most visitors look for
    best: dict[str, dict] = {}
    for r in latest.values():
        if r["market"] in ("1X2", "Vencedor") and r["model_prob"] > best.get(r["match_key"], {}).get("model_prob", -1):
            best[r["match_key"]] = r
    hits = {"games": len(best), "hit": sum(bool(r["won"]) for r in best.values()),
            "rate": round(sum(bool(r["won"]) for r in best.values()) / len(best), 4) if best else None,
            "expected": round(float(np.mean([r["model_prob"] for r in best.values()])), 4) if best else None}
    return {"markets": out, "calibration": calib, "most_likely": hits,
            "note": "Previsões ao vivo registadas por este site; amostras pequenas não permitem conclusões."}


def odds_timeline(match_key: str) -> dict | None:
    """Best price per selection over time for the match result market, from the stored snapshots (one per change).
    Each bookmaker's last price is carried forward until it changes; with PT_ONLY only licensed books count."""
    rows = [r for r in db.odds_series(match_key) if r["market"] in ("1X2", "Vencedor") and not _excluded(r["bookmaker"])
            and (not settings.pt_only or srij.licensed(r["bookmaker"]))]
    if not rows:
        return None
    last, series = {}, {}
    for t, group in itertools.groupby(rows, key=lambda r: r["fetched_at"]):
        for r in group:
            last[(r["selection"], r["bookmaker"])] = r["odds"]
        for sel in {s for s, _ in last}:
            best = max(v for (s, _), v in last.items() if s == sel)
            pts = series.setdefault(sel, [])
            if not pts or pts[-1][1] != best:
                pts.append([t, best])
    return {"market": rows[0]["market"], "series": series}


def explain(r: dict) -> list[str]:
    """Plain-language reasons behind a forecast, from the numbers already in the report."""
    pr, st, ident = r.get("predictions"), r.get("statistics"), r["identification"]
    if not pr:
        return []
    home, away, sport = ident["home"], ident["away"], ident["sport"]
    unit = {"Basquetebol": "pontos", "Futebol americano": "pontos", "Râguebi": "pontos", "Futebol australiano": "pontos",
            "Basebol": "corridas",
            "Voleibol": "sets"}.get(sport, "golos")
    probs = pr["probabilities"]
    res = probs.get("1X2") or probs.get("Vencedor")
    if not res:
        return []
    def label(s):
        return pick_label("1X2", s, home, away, sport)
    def size(x):  # high-scoring sports differ by a few percent: keep a decimal there
        d = abs(x - 1)
        return f"{d:.0%}" if d >= 0.1 else f"{d:.1%}".replace(".", ",")
    def attack(x):
        return "na média da liga" if abs(x - 1) < 0.005 else f"{size(x)} {'acima' if x > 1 else 'abaixo'} da média da liga"
    def conceded(x):  # defence rating above 1 = concedes more
        return (f"{unit} na média da liga" if abs(x - 1) < 0.005
                else f"{size(x)} {'mais' if x > 1 else 'menos'} {unit} do que a média")
    out = []
    if st and st.get("ratings"):
        ra, rb = st["ratings"][home], st["ratings"][away]
        out.append(f"Ataque: o do {home} está {attack(ra['attack'])}; o do {away}, {attack(rb['attack'])}.")
        out.append(f"Defesa: o {home} sofre {conceded(ra['defence'])}; o {away}, {conceded(rb['defence'])}.")
        out.append(f"Jogar em casa vale cerca de {st['home_advantage'] - 1:+.0%} de {unit} nesta liga.")
    if st and st.get("form"):
        f1, f2 = st["form"].get(home, {}), st["form"].get(away, {})
        if f1.get("matches") and f2.get("matches"):
            out.append(f"Forma recente: {home} {f1['form']} ({f1['points_per_game']:.1f} pontos por jogo), "
                       f"{away} {f2['form']} ({f2['points_per_game']:.1f}).")
    k = max(res, key=res.get)
    if pr.get("expected_goals"):
        lam, mu = pr["expected_goals"].values()
        out.append(f"{unit.capitalize()} esperados {lam:.2f} – {mu:.2f}: o resultado mais provável é {label(k)} ({res[k]:.0%}).")
    model = (pr.get("model_probabilities") or {}).get("1X2")
    if model and pr.get("blend"):
        s = max(res, key=lambda x: abs(res[x] - model[x]))
        if abs(res[s] - model[s]) >= 0.05:
            out.append(f"O mercado dá {res[s]:.0%} a «{label(s)}», o nosso modelo {model[s]:.0%}. O mercado costuma refletir "
                       "informação recente (lesões, onzes, treinador) que o modelo não vê, por isso a previsão final segue-o.")
        else:
            out.append("O nosso modelo e o consenso do mercado concordam (diferença inferior a 5 pontos).")
    elif not pr.get("model_probabilities"):
        out.append(f"Probabilidades do consenso do mercado sem a margem ({r['data_status'].get('bookmakers') or 1} casa(s)): "
                   "ainda sem modelo próprio para esta competição.")
    conf = (r.get("data_status") or {}).get("confidence")
    if conf in ("baixa", "muito baixa"):
        out.append("Confiança baixa: há pouca informação sobre estas equipas, a previsão pode estar longe da realidade.")
    return out
