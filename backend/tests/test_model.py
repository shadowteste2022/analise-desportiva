import numpy as np
import pandas as pd
import pytest

from app.models import dixon_coles as dc
from tests.conftest import STRENGTH, TEAMS


def test_recovers_team_order(league):
    m = dc.fit(league, xi=0.0)
    est = np.array([m.attack[m.teams.index(t)] - m.defence[m.teams.index(t)] for t in TEAMS])
    assert np.corrcoef(est, STRENGTH)[0, 1] > 0.9
    assert 0.1 < m.home_adv < 0.4


def test_score_matrix_is_distribution(league):
    m = dc.fit(league, xi=0.0019)
    s = m.score_matrix(TEAMS[9], TEAMS[0])
    assert s.sum() == pytest.approx(1) and (s >= 0).all()
    lam, mu = m.expected_goals(TEAMS[9], TEAMS[0])
    assert lam > mu  # strongest at home vs weakest


def test_no_future_leakage(league):
    cut = league.Date.iloc[200]
    m = dc.fit(league, xi=0.0019, until=cut)
    assert sum(m.matches.values()) == 2 * (league.Date < cut).sum()
    assert m.trained_until == cut


def test_too_little_data(league):
    with pytest.raises(ValueError):
        dc.fit(league.head(10), xi=0.0)


def test_xg_target(league):
    one = league[league.Date < league.Date.min() + pd.Timedelta(days=200)].copy()  # one season: noisy goals
    idx = {t: k for k, t in enumerate(TEAMS)}
    h, a = one.HomeTeam.map(idx).to_numpy(), one.AwayTeam.map(idx).to_numpy()
    one["HxG"] = np.exp(0.25 + STRENGTH[h] - STRENGTH[a] * 0.8)  # perfect xG = true scoring rates
    one["AxG"] = np.exp(STRENGTH[a] - STRENGTH[h] * 0.8)

    def corr(m):
        return np.corrcoef([m.attack[m.teams.index(t)] - m.defence[m.teams.index(t)] for t in TEAMS], STRENGTH)[0, 1]
    goals, xg = dc.fit(one, 0.0), dc.fit(one, 0.0, xg_weight=1.0)
    assert corr(xg) > corr(goals) and corr(xg) > 0.99
    # without xG columns filled in, the blend falls back to goals exactly
    blank = one.assign(HxG=np.nan, AxG=np.nan)
    assert np.allclose(dc.fit(blank, 0.0, xg_weight=0.5).attack, goals.attack, atol=1e-4)


def test_warm_start_matches_cold(league):
    cold = dc.fit(league, xi=0.0019)
    warm = dc.fit(league, xi=0.0019, x0=dc.params(cold))
    assert np.allclose(cold.attack, warm.attack, atol=1e-3)
