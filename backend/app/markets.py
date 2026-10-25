"""Market probabilities from a scoreline matrix, margin removal, EV and opportunity classification."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import nbinom, poisson

from app.config import settings

VALOR = "VALOR POTENCIAL"
INCONCLUSIVO = "VALOR INCONCLUSIVO"
SEM_VALOR = "SEM VALOR IDENTIFICADO"
SEM_DADOS = "SEM DADOS SUFICIENTES"
TOTAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
OUTLIER_TEXT = "acima da odd justa do próprio mercado"  # marks a probable data error in a selection's reasons
DIVERGENCE_TEXT = "do mercado sem margem"  # marks a model estimate far from the market (probable model error)


def market_probs(m: np.ndarray) -> dict[str, dict[str, float]]:
    """All MVP markets from one distribution, so they are mutually consistent."""
    i, j = np.indices(m.shape)
    out = {"1X2": {"1": m[i > j].sum(), "X": m[i == j].sum(), "2": m[i < j].sum()}}
    for line in TOTAL_LINES:
        over = m[i + j > line].sum()
        out[f"OU{line}"] = {"Mais": over, "Menos": 1 - over}
    btts = 1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0]
    out["BTTS"] = {"Sim": btts, "Não": 1 - btts}
    return {k: {s: float(p) for s, p in v.items()} for k, v in out.items()}


def implied(odds: float) -> float:
    return 1.0 / odds


def devig(odds: dict[str, float], method: str | None = None) -> tuple[dict[str, float], float] | tuple[None, None]:
    """Margin removal over a complete set of mutually exclusive outcomes. Returns (fair probabilities, margin).

    proportional: q_i / Σq (spreads the margin evenly).
    power: q_i^k with k chosen so Σ = 1; takes relatively more margin off long shots.
    shin: Shin (1993) insider-trading model, the usual correction for the favourite-longshot bias.
    """
    if not odds or any(o is None or not o > 1 for o in odds.values()):
        return None, None
    method = method or settings.devig_method
    keys = list(odds)
    q = np.array([1 / odds[k] for k in keys])
    total = q.sum()
    p = q / total  # proportional; also the answer when there is no margin to remove
    if total > 1 and method != "proportional":
        try:
            if method == "power":
                p = q ** brentq(lambda k: (q ** k).sum() - 1, 1, 50)
            elif method == "shin":
                def shin(z):
                    return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / total) - z) / (2 * (1 - z))
                p = shin(brentq(lambda z: shin(z).sum() - 1, 0, 0.99))
            else:
                raise KeyError(method)
        except ValueError:  # no root in range (extreme margins): keep proportional rather than fail the analysis
            pass
    return {k: float(v) for k, v in zip(keys, p / p.sum())}, float(total - 1)


def ev(p: float, odds: float) -> float:
    return p * odds - 1


def fair_odds(p: float) -> float | None:
    return 1 / p if p > 0 else None


def confidence(n_home: int, n_away: int) -> str | None:
    """Statistical confidence from the sample behind the team ratings. Separate from the event probability."""
    n = min(n_home, n_away)
    if n < settings.min_team_matches:
        return None
    return "alta" if n >= 40 else "média" if n >= 20 else "baixa"


def classify(p: float | None, odds: float | None, conf: str | None,
             market_p: float | None = None) -> tuple[str, list[str]]:
    """Category plus the reasons behind it. High probability alone never makes a bet interesting."""
    if p is None or conf is None:
        return SEM_DADOS, ["modelo sem histórico suficiente para estas equipas"]
    if odds is None:
        return SEM_DADOS, ["sem odd disponível para este mercado"]
    e = ev(p, odds)
    if e <= 0:
        return SEM_VALOR, [f"EV estimado {e:+.1%} ≤ 0"]
    reasons = [f"EV estimado {e:+.1%}"]
    if market_p is not None and odds * market_p - 1 > settings.odds_outlier:
        reasons.append(f"odd {odds * market_p - 1:.0%} {OUTLIER_TEXT} — provável erro ou odd desatualizada num operador")
        return INCONCLUSIVO, reasons
    if market_p is not None and abs(p - market_p) > settings.max_market_divergence:
        reasons.append(f"modelo diverge {abs(p - market_p):.0%} {DIVERGENCE_TEXT} — provável informação "
                       "que o modelo não vê (lesões, rotação, notícias)")
        return INCONCLUSIVO, reasons
    if e < settings.ev_margin:
        reasons.append(f"abaixo da margem de segurança de {settings.ev_margin:.0%}")
        return INCONCLUSIVO, reasons
    if conf == "baixa":
        reasons.append("confiança estatística baixa (amostra pequena)")
        return INCONCLUSIVO, reasons
    return VALOR, reasons


TEAM_LINES = (0.5, 1.5, 2.5)
HANDICAPS = (-1.5, 1.5)  # half lines only: whole and quarter lines can push, which the EV maths does not model


def derived_markets(p1x2: dict[str, float], m: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Markets that follow from the match result (double chance, draw no bet) and, with the scoreline matrix, team
    goals and handicaps. Draw no bet is given as the probability conditional on no draw: the stake is returned on a
    draw, so its fair odds are 1/p, and evaluate_markets scales its EV back by P(no draw)."""
    p1, px, p2 = p1x2["1"], p1x2["X"], p1x2["2"]
    out = {"DC": {"1X": p1 + px, "12": p1 + p2, "X2": px + p2}, "DNB": {"1": p1 / (p1 + p2), "2": p2 / (p1 + p2)}}
    if m is not None:
        i, j = np.indices(m.shape)
        for line in TEAM_LINES:
            for key, goals in (("GC", i), ("GF", j)):
                over = m[goals > line].sum()
                out[f"{key}{line}"] = {"Mais": over, "Menos": 1 - over}
        for h in HANDICAPS:
            cover = m[i - j + h > 0].sum()
            out[f"AH{h:+g}"] = {"1": cover, "2": 1 - cover}
    return {k: {s: float(p) for s, p in v.items()} for k, v in out.items()}


def evidence(n_books: int, history: str | None, validation: dict, model_p: float | None,
             market_p: float | None) -> dict:
    """How much the estimate can be trusted (0-100), kept apart from the probability itself. Four parts of 25:
    breadth of the market, team history behind the model, out-of-sample validation, model/market agreement."""
    parts = {
        "casas": 25 if n_books >= 4 else 16 if n_books >= 2 else 8 if n_books == 1 else 0,
        "histórico": {"alta": 25, "média": 18, "baixa": 10}.get(history, 0),
        "validação": 25 if validation.get("validated") else 10 if validation.get("model_log_loss") is not None else 0,
        "acordo modelo/mercado": 0}
    if model_p is not None and market_p is not None:
        d = abs(model_p - market_p)
        parts["acordo modelo/mercado"] = 25 if d <= 0.03 else 15 if d <= 0.06 else 5 if d <= 0.12 else 0
    score = sum(parts.values())
    # "forte" needs out-of-sample validation: without it, agreement and breadth alone cannot show an edge
    strong = score >= 70 and validation.get("validated")
    return {"score": score, "label": "forte" if strong else "moderada" if score >= 40 else "fraca", "parts": parts}


HT_LINES = (0.5, 1.5, 2.5)
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


def halftime_markets(lam: float, mu: float, share: float) -> dict[str, dict[str, float]]:
    """First-half result and goals: independent Poisson with the full-time expected goals scaled by the share of goals
    the league scores before half-time (the low-score correction of Dixon-Coles is left out)."""
    g = np.arange(8)
    m = np.outer(poisson.pmf(g, lam * share), poisson.pmf(g, mu * share))
    m /= m.sum()
    i, j = np.indices(m.shape)
    out = {"HT1X2": {"1": m[i > j].sum(), "X": m[i == j].sum(), "2": m[i < j].sum()}}
    for line in HT_LINES:
        over = m[i + j > line].sum()
        out[f"HTOU{line}"] = {"Mais": over, "Menos": 1 - over}
    return {k: {s: float(p) for s, p in v.items()} for k, v in out.items()}


def count_markets(prefix: str, mean: float, dispersion: float, lines: tuple) -> dict[str, dict[str, float]]:
    """Over/under for a match count (corners, cards). Negative binomial with variance = mean x dispersion, because
    these counts vary more than a Poisson; Poisson when the league shows no extra variance."""
    out = {}
    for line in lines:
        k = int(np.floor(line))
        cdf = poisson.cdf(k, mean) if dispersion <= 1.05 else nbinom.cdf(k, mean / (dispersion - 1), 1 / dispersion)
        out[f"{prefix}{line}"] = {"Mais": float(1 - cdf), "Menos": float(cdf)}
    return out


SHOT_LINES = (18.5, 20.5, 22.5, 24.5, 26.5, 28.5)
SHOT_TARGET_LINES = (5.5, 6.5, 7.5, 8.5, 9.5, 10.5)


def score_markets(m: np.ndarray) -> dict[str, dict[str, float]]:
    """From the scoreline matrix: exact total goals, clean sheets and winning margin."""
    i, j = np.indices(m.shape)
    tot, diff = i + j, i - j
    out = {"EG": {**{str(k): m[tot == k].sum() for k in range(4)}, "4+": m[tot >= 4].sum()},
           "CS": {"1": m[j == 0].sum(), "2": m[i == 0].sum()},  # home / away keeps a clean sheet
           "WM": {"1 por 1": m[diff == 1].sum(), "1 por 2": m[diff == 2].sum(), "1 por 3+": m[diff >= 3].sum(),
                  "X": m[diff == 0].sum(),
                  "2 por 1": m[diff == -1].sum(), "2 por 2": m[diff == -2].sum(), "2 por 3+": m[diff <= -3].sum()}}
    return {k: {s: float(p) for s, p in v.items()} for k, v in out.items()}


def halves_markets(lam: float, mu: float, share: float) -> dict[str, dict[str, float]]:
    """Half-time/full-time, second-half result and the half with more goals, with each half's goals independent
    Poisson (first half = share of the expected goals, second half = the rest)."""
    g = np.arange(8)
    h1 = np.outer(poisson.pmf(g, lam * share), poisson.pmf(g, mu * share)); h1 /= h1.sum()
    h2 = np.outer(poisson.pmf(g, lam * (1 - share)), poisson.pmf(g, mu * (1 - share))); h2 /= h2.sum()
    i, j = np.indices(h1.shape)
    res = lambda a, b: np.where(a > b, 0, np.where(a == b, 1, 2))  # noqa: E731  0 = home, 1 = draw, 2 = away
    r1, r2, d1, d2 = res(i, j), res(i, j), i - j, i - j
    names = ["1", "X", "2"]
    htft = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            # P(HT result a, FT result b) = sum over first-half margin d1 in a, second-half margin d2 with sign(d1+d2) = b
            for x in np.unique(d1[r1 == a]):
                p1 = h1[d1 == x].sum()
                for y in np.unique(d2):
                    t = x + y
                    if (t > 0 and b == 0) or (t == 0 and b == 1) or (t < 0 and b == 2):
                        htft[a, b] += p1 * h2[d2 == y].sum()
    g1, g2 = (i + j), (i + j)
    p1_more = sum(h1[g1 == a].sum() * h2[g2 < a].sum() for a in range(1, 15))
    p2_more = sum(h2[g2 == a].sum() * h1[g1 < a].sum() for a in range(1, 15))
    out = {"HTFT": {f"{names[a]}/{names[b]}": htft[a, b] for a in range(3) for b in range(3)},
           "2H1X2": {"1": h2[r2 == 0].sum(), "X": h2[r2 == 1].sum(), "2": h2[r2 == 2].sum()},
           "MP": {"1ª parte": p1_more, "2ª parte": p2_more, "Igual": 1 - p1_more - p2_more}}
    return {k: {s: float(p) for s, p in v.items()} for k, v in out.items()}
