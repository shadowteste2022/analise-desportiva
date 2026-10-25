"""Dixon-Coles (1997) goals model: Poisson with team attack/defence, home advantage,
low-score correction (rho) and exponential time decay so recent matches weigh more.

Why this model for the MVP: it yields a full scoreline distribution, from which 1X2, totals
and BTTS all follow consistently (no double counting across markets), it is well studied,
and with ~2*teams+2 parameters it is hard to overfit. A small ridge penalty shrinks teams
with few matches (e.g. promoted sides) towards the league average.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10
RIDGE = 0.01


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray  # higher = concedes more
    home_adv: float
    rho: float
    matches: dict[str, int]  # matches per team in the training window
    trained_until: pd.Timestamp

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.teams.index(home), self.teams.index(away)
        lam = np.exp(self.home_adv + self.attack[i] + self.defence[j])
        mu = np.exp(self.attack[j] + self.defence[i])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home goals = i, away goals = j), i,j in 0..MAX_GOALS. Exact, so no Monte Carlo needed."""
        return self.matrix(*self.expected_goals(home, away))

    def matrix(self, lam: float, mu: float) -> np.ndarray:
        """Scoreline distribution for given expected goals, with this model's low-score correction."""
        g = np.arange(MAX_GOALS + 1)
        ph = np.exp(g * np.log(lam) - lam - gammaln(g + 1))
        pa = np.exp(g * np.log(mu) - mu - gammaln(g + 1))
        m = np.outer(ph, pa)
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def knows(self, team: str) -> bool:
        return team in self.teams


def fit(df: pd.DataFrame, xi: float, until: pd.Timestamp | None = None,
        x0: dict | None = None, xg_weight: float = 0.0) -> DixonColes:
    """Fit on matches strictly before `until` (no future leakage). `x0` warm-starts from a previous fit.

    xg_weight > 0 fits attack/defence to (1-w)*goals + w*xG (Poisson quasi-likelihood) where HxG/AxG exist,
    falling back to goals elsewhere; rho stays on the real scorelines since it models discrete low scores.
    """
    until = until if until is not None else df.Date.max() + pd.Timedelta(days=1)
    d = df[df.Date < until]
    if len(d) < 50:
        raise ValueError("Jogos insuficientes para ajustar o modelo")
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    idx = {t: k for k, t in enumerate(teams)}
    h = d.HomeTeam.map(idx).to_numpy()
    a = d.AwayTeam.map(idx).to_numpy()
    hg, ag = d.FTHG.to_numpy(float), d.FTAG.to_numpy(float)
    th, ta = hg, ag  # Poisson targets
    if xg_weight and {"HxG", "AxG"} <= set(d.columns):
        has = (d.HxG.notna() & d.AxG.notna()).to_numpy()
        th = np.where(has, (1 - xg_weight) * hg + xg_weight * d.HxG.fillna(0).to_numpy(float), hg)
        ta = np.where(has, (1 - xg_weight) * ag + xg_weight * d.AxG.fillna(0).to_numpy(float), ag)
    w = np.exp(-xi * (until - d.Date).dt.days.to_numpy())
    n = len(teams)
    const = gammaln(th + 1) + gammaln(ta + 1)
    c00, c01, c10, c11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

    def nll(p):
        att, dfn, home, rho = p[:n] - p[:n].mean(), p[n:2 * n], p[2 * n], p[2 * n + 1]
        lam = np.exp(home + att[h] + dfn[a])
        mu = np.exp(att[a] + dfn[h])
        tau = np.ones_like(lam)
        tau[c00] = 1 - lam[c00] * mu[c00] * rho
        tau[c01] = 1 + lam[c01] * rho
        tau[c10] = 1 + mu[c10] * rho
        tau[c11] = 1 - rho
        ll = np.log(np.clip(tau, 1e-10, None)) + th * np.log(lam) - lam + ta * np.log(mu) - mu - const
        return -(w * ll).sum() + RIDGE * (p[:2 * n] ** 2).sum()

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = 0.25
    if x0:
        for t, k in idx.items():
            if t in x0["teams"]:
                p0[k], p0[n + k] = x0["teams"][t]
        p0[2 * n], p0[2 * n + 1] = x0["home"], x0["rho"]
    bounds = [(None, None)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
    p = minimize(nll, p0, method="L-BFGS-B", bounds=bounds).x
    counts = pd.concat([d.HomeTeam, d.AwayTeam]).value_counts().to_dict()
    return DixonColes(teams, p[:n] - p[:n].mean(), p[n:2 * n], float(p[2 * n]), float(p[2 * n + 1]),
                      {t: int(counts.get(t, 0)) for t in teams}, until)


def params(m: DixonColes) -> dict:
    """Serializable parameters for warm starts."""
    return {"teams": {t: (m.attack[k], m.defence[k]) for k, t in enumerate(m.teams)},
            "home": m.home_adv, "rho": m.rho}
