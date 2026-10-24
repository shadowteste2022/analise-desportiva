"""Model + market combination: logarithmic opinion pool p ∝ p_model^a · p_market^b, fitted per market type.

Why this form: two weights per market are hard to overfit, they read directly (a ≈ 0 means the model adds nothing
the market does not already know), and it never counts the same information twice because the market enters once.
The market input is the pre-match average price without margin: what is known when a bet would be placed.
"""
import json

import numpy as np
from scipy.optimize import minimize

from app import markets as mk
from app.config import settings
from app.data import football_data as fd

MARKETS = ("1X2", "OU2.5")  # the markets with a historical market price in the CSVs


def combine(pm: np.ndarray, pk: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * np.log(np.clip(pm, 1e-12, None)) + b * np.log(np.clip(pk, 1e-12, None))
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit(pm: np.ndarray, pk: np.ndarray, y: np.ndarray) -> dict:
    """Weights minimising log loss on (n, k) model probs, market probs and one-hot outcomes."""
    def nll(w):
        return -np.log(np.clip((combine(pm, pk, *w) * y).sum(axis=1), 1e-12, None)).mean()
    a, b = minimize(nll, [0.5, 0.5], method="L-BFGS-B", bounds=[(0, 3), (0, 3)]).x
    return {"a": round(float(a), 4), "b": round(float(b), 4), "n": len(y)}


NOT_BOOKS = set(fd.AGGREGATES.values()) | {"Betfair Exchange"}  # aggregates, and an exchange before commission


def market_fair(odds: dict, market: str) -> dict | None:
    """No-vig market probabilities: the CSV average price when there is one, otherwise the consensus (average price)
    of 2+ bookmakers pricing the whole market, e.g. the live lines other than 2.5 goals."""
    books = odds.get(market, {})
    avg = books.get(fd.AGGREGATES["Avg"])
    if not avg:
        full = [s for b, s in books.items() if b not in NOT_BOOKS]
        full = [s for s in full if full and s.keys() == full[0].keys()]
        if len(full) < 2:
            return None
        avg = {k: float(np.mean([s[k] for s in full])) for k in full[0]}
    return mk.devig(avg)[0]


def weights_for(weights: dict, market: str) -> dict | None:
    """Validated weights exist for 1X2 and O/U 2.5 only; other goal lines and BTTS (also two-way goal markets)
    reuse the O/U 2.5 ones. Either way the backtest gives the model ~zero weight."""
    if market in weights:
        return weights[market]
    return weights.get("OU2.5") if market.startswith("OU") or market == "BTTS" else None


def apply(probs: dict, odds: dict, weights: dict | None) -> tuple[dict, list[str]]:
    """Blended copy of `probs` for markets with weights and a market price; returns the markets it changed."""
    if not weights:
        return probs, []
    out, used = dict(probs), []
    for market in probs:
        w = weights_for(weights, market)
        fair = market_fair(odds, market) if w else None
        if not fair or fair.keys() != probs[market].keys():
            continue
        sels = list(probs[market])
        p = combine(np.array([probs[market][s] for s in sels]), np.array([fair[s] for s in sels]), w["a"], w["b"])
        out[market] = {s: float(v) for s, v in zip(sels, p)}
        used.append(market)
    return out, used


def load() -> dict | None:
    path = settings.data_dir / "blend.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(weights: dict) -> None:
    path = settings.data_dir / "blend.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=1), encoding="utf-8")
