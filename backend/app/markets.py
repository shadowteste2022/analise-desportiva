"""Market probabilities from a scoreline matrix, margin removal, EV and opportunity classification."""
import numpy as np
from scipy.optimize import brentq

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
