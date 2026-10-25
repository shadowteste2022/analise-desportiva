import os
import tempfile

# Must run before app.config is imported: isolate data and disable external services.
# Every setting the tests depend on is pinned here, so variables set by the CI workflow (e.g. ODDS_API_RESET_DAY)
# cannot change the results.
os.environ.update({"DATA_DIR": tempfile.mkdtemp(), "ADMIN_TOKEN": "test-token", "ODDS_API_KEY": "", "LEAGUES": "T0",
                   "ODDS_API_IO_KEY": "", "ODDS_API_RESET_DAY": "1", "ODDS_API_MONTHLY_BUDGET": "500",
                   "ODDS_API_MIN_REMAINING": "20", "OTHER_SPORTS": "*", "DEVIG_METHOD": "shin", "XG_WEIGHT": "0",
                   "ODDSPAPI_KEY": "", "PT_ONLY": "false",
                   "API_FOOTBALL_KEY": ""})

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import poisson  # noqa: E402

TEAMS = [f"Team{k:02d}" for k in range(10)]
STRENGTH = np.linspace(-0.4, 0.4, 10)  # true attack; defence = -attack


def make_league(seasons=4, seed=0, start="2022-08-06") -> pd.DataFrame:
    """Double round robin per season with goals from a known Poisson model and 5%-margin odds."""
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp(start)
    for _ in range(seasons):
        pairs = [(h, a) for h in range(10) for a in range(10) if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            d = day + pd.Timedelta(days=7 * (k // 5))
            lam, mu = np.exp(0.25 + STRENGTH[h] - STRENGTH[a] * 0.8), np.exp(STRENGTH[a] - STRENGTH[h] * 0.8)
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            g = np.arange(11)
            m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
            i, j = np.indices(m.shape)
            p = [m[i > j].sum(), m[i == j].sum(), m[i < j].sum()]
            over = m[i + j > 2.5].sum()
            row = {"Div": "T0", "Date": d, "Time": "15:00", "HomeTeam": TEAMS[h], "AwayTeam": TEAMS[a],
                   "FTHG": hg, "FTAG": ag}
            for c in ("", "C"):
                for book in ("B365", "Avg"):
                    row.update({f"{book}{c}H": 1 / (p[0] * 1.05), f"{book}{c}D": 1 / (p[1] * 1.05),
                                f"{book}{c}A": 1 / (p[2] * 1.05),
                                f"{book}{c}>2.5": 1 / (over * 1.05), f"{book}{c}<2.5": 1 / ((1 - over) * 1.05)})
            rows.append(row)
        day += pd.Timedelta(days=365)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def league():
    return make_league()
