"""World football fixtures and results from API-Football (https://www.api-football.com). Requires API_FOOTBALL_KEY.

Why this source: one request lists every game of a date across ~1,000 competitions (men's and women's), which the
other free feeds do not. The free plan (100 requests/day, reset 00:00 UTC) only serves yesterday to tomorrow by
date, and whole seasons only for 2022-2024. So the history behind each league's model is its 2024 season, fetched
once per league as the league shows up, plus every finished game seen from now on (kept in DATA_DIR/api_football,
which the daily job commits). Forecasts improve as that history grows.
"""
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from app.config import settings

log = logging.getLogger(__name__)
BASE = "https://v3.football.api-sports.io"
RESERVE = 60  # requests per day left for the live-scores Worker (worker/live.js) and a margin
FIXTURES_TTL_H = 6
HISTORY_SEASON = 2024  # latest season the free plan serves by league
HISTORY_PER_RUN = 6  # league seasons fetched per run: hourly runs spread the backfill over the day
FIXTURES_KEEP = 15  # requests left for the fixture lists (3 dates every FIXTURES_TTL_H hours)
state = {"used": None, "limit": None, "error": None, "skipped": 0}


def enabled() -> bool:
    return bool(settings.api_football_key)


def _cache_dir():
    return settings.data_dir / "cache" / "api_football"


def _history_dir():
    return settings.data_dir / "api_football"


def _get(path: str, **params) -> dict:
    r = httpx.get(f"{BASE}/{path}", params=params, headers={"x-apisports-key": settings.api_football_key}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):  # plan limits and bad keys come back as HTTP 200 with an errors object
        raise ValueError(json.dumps(body["errors"], ensure_ascii=False))
    return body


def _refresh_quota() -> None:
    """Usage from /status, which costs no request. Once per process."""
    if state["used"] is not None or state["error"]:
        return
    try:
        req = _get("status")["response"]["requests"]
        state.update(used=int(req["current"]), limit=int(req["limit_day"]))
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        state["error"] = f"conta indisponível ({type(e).__name__})"
        log.warning("API-Football: %s", state["error"])


def can_spend(n: int = 1) -> bool:
    _refresh_quota()
    return state["used"] is not None and state["used"] + n <= state["limit"] - RESERVE


def _spend() -> None:
    state["used"] += 1


def normalize(raw: list[dict]) -> list[dict]:
    """The fields the app uses, one dict per game."""
    out = []
    for f in raw:
        try:
            fx, lg, teams, goals, ht = f["fixture"], f["league"], f["teams"], f["goals"], f["score"]["halftime"]
            out.append({"id": fx["id"], "kickoff": fx["date"], "status": fx["status"]["short"],
                        "league_id": lg["id"], "league": lg["name"], "country": lg["country"], "season": lg["season"],
                        "home": teams["home"]["name"], "away": teams["away"]["name"],
                        "hg": goals["home"], "ag": goals["away"], "hthg": ht["home"], "htag": ht["away"]})
        except (KeyError, TypeError):
            continue
    return out


def fixtures(day: date) -> list[dict]:
    """Every game of one date (UTC): fresh cache, else a request within the quota, else the last saved list."""
    if not enabled():
        return []
    path = _cache_dir() / f"fixtures_{day.isoformat()}.json"
    hit = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if hit and (time.time() - hit["fetched_at"]) / 3600 < FIXTURES_TTL_H:
        return hit["games"]
    if not can_spend():
        state["skipped"] += 1
        return hit["games"] if hit else []
    try:
        games = normalize(_get("fixtures", date=day.isoformat())["response"])
        _spend()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        _spend()  # a refused call still counts towards the daily limit
        log.warning("API-Football: jogos de %s indisponíveis (%s)", day, e)
        return hit["games"] if hit else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "games": games}, ensure_ascii=False), encoding="utf-8")
    record(games)
    return games


def record(games: list[dict]) -> None:
    """Keep every finished game: the history that grows the models (after extra time: the 90-minute score is not
    published here, so those games are left out)."""
    done = [g for g in games if g["status"] == "FT" and g["hg"] is not None and g["ag"] is not None]
    if not done:
        return
    path = _history_dir() / "results.json"
    seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    seen.update({str(g["id"]): g for g in done})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")


def backfill(league_ids: list[int]) -> None:
    """Fetch the HISTORY_SEASON results of leagues that have none yet, a few per run and within the quota."""
    if not enabled():
        return
    todo = [lg for lg in league_ids if not (_history_dir() / f"season_{lg}_{HISTORY_SEASON}.json").exists()]
    for lg in todo[:HISTORY_PER_RUN]:
        if not can_spend(1 + FIXTURES_KEEP):  # the day's fixture refreshes come first
            return
        try:
            games = [g for g in normalize(_get("fixtures", league=lg, season=HISTORY_SEASON)["response"])
                     if g["status"] == "FT" and g["hg"] is not None]
        except ValueError as e:  # e.g. league not covered: remember it, do not ask again
            games = []
            log.info("API-Football: sem histórico %s para a liga %s (%s)", HISTORY_SEASON, lg, e)
        except httpx.HTTPError as e:
            _spend()
            log.warning("API-Football: histórico da liga %s indisponível (%s)", lg, e)
            continue
        _spend()
        path = _history_dir() / f"season_{lg}_{HISTORY_SEASON}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(games, ensure_ascii=False), encoding="utf-8")


def history(league_id: int) -> pd.DataFrame:
    """Finished games of a league (saved season + games recorded since), in the football-data column names."""
    games = []
    season = _history_dir() / f"season_{league_id}_{HISTORY_SEASON}.json"
    if season.exists():
        games += json.loads(season.read_text(encoding="utf-8"))
    results = _history_dir() / "results.json"
    if results.exists():
        games += [g for g in json.loads(results.read_text(encoding="utf-8")).values() if g["league_id"] == league_id]
    if not games:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"])
    df = pd.DataFrame(games).drop_duplicates("id")
    return pd.DataFrame({"Date": pd.to_datetime(df.kickoff, utc=True).dt.tz_localize(None).dt.normalize(),
                         "HomeTeam": df.home, "AwayTeam": df.away, "FTHG": df.hg.astype(int), "FTAG": df.ag.astype(int),
                         "HTHG": pd.to_numeric(df.hthg), "HTAG": pd.to_numeric(df.htag)}).sort_values("Date")


def upcoming(now: datetime | None = None) -> list[dict]:
    """Games from 2 h ago to the end of tomorrow (UTC) that have not finished; yesterday's list is fetched too, for
    its results."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    fixtures(today - timedelta(days=1))
    games = fixtures(today) + fixtures(today + timedelta(days=1))
    cutoff = now - timedelta(hours=2)
    return [g for g in games if g["status"] in ("NS", "TBD") and datetime.fromisoformat(g["kickoff"]) >= cutoff]
