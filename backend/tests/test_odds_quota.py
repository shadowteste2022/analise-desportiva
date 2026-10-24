import json
import time
from datetime import datetime, timezone

import pytest

from app import db
from app.config import settings
from app.data import odds_api


@pytest.fixture
def quota(monkeypatch):
    monkeypatch.setattr(odds_api, "_refresh_quota", lambda: None)  # never call the network
    q = {"remaining": None, "used": None, "budget": None, "allowed_now": None, "skipped": []}
    monkeypatch.setattr(odds_api, "quota", q)
    monkeypatch.setattr(settings, "odds_api_monthly_budget", 500)
    return q


def test_period_fraction():
    mid = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    assert odds_api.period_fraction(mid, 1) == pytest.approx(15.5 / 31)
    # plan renews on the 15th: the 10th belongs to the period that started on 15 September
    assert odds_api.period_fraction(datetime(2026, 10, 10, tzinfo=timezone.utc), 15) == pytest.approx(25 / 30)
    assert odds_api.period_fraction(datetime(2026, 1, 3, tzinfo=timezone.utc), 5) == pytest.approx(29 / 31)


def test_spending_is_paced(quota):
    mid = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)  # half the budget (250) earned + one full refresh (2)
    quota.update(used=240, remaining=260)
    assert odds_api.can_spend(2, mid)
    quota.update(used=251, remaining=249)
    assert not odds_api.can_spend(2, mid)
    quota.update(used=10, remaining=21)  # on pace, but would break the 20-credit reserve
    assert not odds_api.can_spend(2, mid)


def test_unknown_usage_never_spends(quota):
    assert not odds_api.can_spend(2)


def test_skipped_league_reuses_saved_odds(quota, monkeypatch):
    monkeypatch.setattr(settings, "odds_api_key", "test")
    odds_api._cache.pop("E0", None)
    path = odds_api._disk("E0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time() - 5 * 3600, "events": [{"id": "x"}]}), encoding="utf-8")
    quota.update(used=500, remaining=0)  # budget exhausted
    assert odds_api.fetch("E0") == [{"id": "x"}]  # 5 h old: past the 1 h TTL but within ODDS_MAX_AGE_HOURS
    assert quota["skipped"] == ["E0"]
    path.write_text(json.dumps({"fetched_at": time.time() - 72 * 3600, "events": [{"id": "x"}]}), encoding="utf-8")
    assert odds_api.fetch("E0") == []  # too old to show


def test_snapshots_only_store_changes():
    key = "T0|2099-01-01|A|B"
    db.snapshot_odds(key, {"1X2": {"Bk": {"1": 2.0, "X": 3.4, "2": 3.6}}}, "2099-01-01T10:00:00")
    db.snapshot_odds(key, {"1X2": {"Bk": {"1": 2.0, "X": 3.4, "2": 3.6}}}, "2099-01-01T11:00:00")
    db.snapshot_odds(key, {"1X2": {"Bk": {"1": 2.3, "X": 3.4, "2": 3.6}}}, "2099-01-01T12:00:00")
    from contextlib import closing
    with closing(db.connect()) as con:
        rows = con.execute("SELECT selection, odds FROM odds_snapshots WHERE match_key = ? ORDER BY id", (key,)).fetchall()
    assert [tuple(r) for r in rows] == [("1", 2.0), ("X", 3.4), ("2", 3.6), ("1", 2.3)]
    moves = [m for m in db.odds_moves(key) if m["selection"] == "1"]
    assert moves and moves[0]["first_odds"] == 2.0 and moves[0]["last_odds"] == 2.3
