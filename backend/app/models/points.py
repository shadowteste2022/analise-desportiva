"""Points model for high-scoring sports (basketball, American football, rugby, handball, volleyball sets).

Each side's score = league mean at home/away + own attack - opponent's defence, fitted by weighted ridge least squares
with exponential time decay. The margin and the total are then normal around their expectations, with the spread
measured from the fit's own residuals. Why this model: in these sports scores are large and roughly normal, so a
Poisson model would be too narrow, and a linear rating model is the usual, well-tested baseline.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

RIDGE = 2.0  # pulls teams with few games towards the league average (like two average games)


@dataclass
class PointsModel:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray  # higher = concedes fewer
    mean_home: float
    mean_away: float
    sd_margin: float
    sd_total: float
    matches: dict[str, int]

    def knows(self, team: str) -> bool:
        return team in self.teams

    def expected(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.teams.index(home), self.teams.index(away)
        return (float(self.mean_home + self.attack[i] - self.defence[j]),
                float(self.mean_away + self.attack[j] - self.defence[i]))

    def result(self, home: str, away: str, draws: bool) -> dict[str, float]:
        """Home win / draw / away win. With draws, a draw is a margin within half a point of zero."""
        eh, ea = self.expected(home, away)
        mu, sd = eh - ea, self.sd_margin
        if not draws:
            p1 = float(norm.cdf(mu / sd))
            return {"1": p1, "2": 1 - p1}
        p1, p2 = float(1 - norm.cdf((0.5 - mu) / sd)), float(norm.cdf((-0.5 - mu) / sd))
        return {"1": p1, "X": 1 - p1 - p2, "2": p2}

    def totals(self, home: str, away: str, lines: list[float]) -> dict[str, dict[str, float]]:
        total = sum(self.expected(home, away))
        return {f"OU{line:g}": {"Mais": float(1 - norm.cdf((line - total) / self.sd_total)),
                                "Menos": float(norm.cdf((line - total) / self.sd_total))} for line in lines}


def fit(df: pd.DataFrame, xi: float) -> PointsModel:
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    n, idx = len(teams), {t: k for k, t in enumerate(teams)}
    w = np.exp(-xi * (df.Date.max() - df.Date).dt.days.to_numpy())
    h = df.HomeTeam.map(idx).to_numpy()
    a = df.AwayTeam.map(idx).to_numpy()
    g = len(df)
    # columns: mean_home, mean_away, attack[n], defence[n]; two rows per game (home score, away score)
    X = np.zeros((2 * g, 2 + 2 * n))
    X[:g, 0] = 1
    X[np.arange(g), 2 + h] = 1
    X[np.arange(g), 2 + n + a] = -1
    X[g + np.arange(g), 1] = 1
    X[g + np.arange(g), 2 + a] = 1
    X[g + np.arange(g), 2 + n + h] = -1
    y = np.concatenate([df.FTHG.to_numpy(float), df.FTAG.to_numpy(float)])
    sw = np.sqrt(np.concatenate([w, w]))
    ridge = np.sqrt(RIDGE) * np.eye(2 + 2 * n)[2:]
    A = np.vstack([X * sw[:, None], ridge])
    b = np.concatenate([y * sw, np.zeros(2 * n)])
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = X @ beta
    ph, pa = pred[:g], pred[g:]
    margin_res = (df.FTHG.to_numpy() - df.FTAG.to_numpy()) - (ph - pa)
    total_res = (df.FTHG.to_numpy() + df.FTAG.to_numpy()) - (ph + pa)
    sd = lambda r: float(np.sqrt(np.average(r ** 2, weights=w)))  # noqa: E731
    counts = pd.concat([df.HomeTeam, df.AwayTeam]).value_counts().to_dict()
    return PointsModel(teams, beta[2:2 + n], beta[2 + n:], float(beta[0]), float(beta[1]),
                       max(sd(margin_res), 0.5), max(sd(total_res), 0.5), counts)
