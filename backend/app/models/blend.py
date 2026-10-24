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


def market_fair(odds: dict, market: str) -> dict | None:
    avg = odds.get(market, {}).get(fd.AGGREGATES["Avg"])
    return mk.devig(avg)[0] if avg else None


def apply(probs: dict, odds: dict, weights: dict | None) -> tuple[dict, list[str]]:
    """Blended copy of `probs` for markets with weights and a market price; returns the markets it changed."""
    if not weights:
        return probs, []
    out, used = dict(probs), []
    for market, w in weights.items():
        fair = market_fair(odds, market)
        if market not in probs or not fair or fair.keys() != probs[market].keys():
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
