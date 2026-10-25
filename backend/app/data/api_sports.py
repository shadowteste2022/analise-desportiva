"""Fixtures and results of the other sports from the API-Sports family (same key as API-Football, API_FOOTBALL_KEY).

Each sport is its own API with its own free quota (100 requests/day). As for football, the free plan serves games
by date only from yesterday to tomorrow, and whole seasons only up to 2024; so each league's history is its 2024
season, fetched once as the league shows up, plus every finished game seen from now on (DATA_DIR/api_sports).
"""
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from app.config import settings

log = logging.getLogger(__name__)
# sport key -> (API host, name shown on the site)
SPORTS = {"basketball": ("v1.basketball", "Basquetebol"), "hockey": ("v1.hockey", "Hóquei no gelo"),
          "american-football": ("v1.american-football", "Futebol americano"), "baseball": ("v1.baseball", "Basebol"),
          "handball": ("v1.handball", "Andebol"), "volleyball": ("v1.volleyball", "Voleibol"),
          "rugby": ("v1.rugby", "Râguebi")}
RESERVE = 10
GAMES_TTL_H = 6
HISTORY_PER_RUN = 6
GAMES_KEEP = 15  # requests left for the game lists (3 dates every GAMES_TTL_H hours)
FINISHED = {"FT", "AOT", "AET", "AP", "AW"}  # full time, after overtime / extra time / penalties or shoot-out
NOT_STARTED = {"NS", "TBD"}
state = {sport: {"used": None, "limit": None, "error": None} for sport in SPORTS}


def enabled() -> bool:
    return bool(settings.api_football_key)


def _get(sport: str, path: str, **params) -> dict:
    r = httpx.get(f"https://{SPORTS[sport][0]}.api-sports.io/{path}", params=params,
                  headers={"x-apisports-key": settings.api_football_key}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise ValueError(json.dumps(body["errors"], ensure_ascii=False))
    return body


def can_spend(sport: str, n: int = 1) -> bool:
    s = state[sport]
    if s["used"] is None and not s["error"]:
        try:  # /status costs no request
            req = _get(sport, "status")["response"]["requests"]
            s.update(used=int(req["current"]), limit=int(req["limit_day"]))
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            s["error"] = f"conta indisponível ({type(e).__name__})"
    return s["used"] is not None and s["used"] + n <= s["limit"] - RESERVE


def _total(score) -> int | None:
    """Scores are a number (hockey, handball...) or a breakdown with a total (basketball, baseball, NFL)."""
    if isinstance(score, dict):
        score = score.get("total")
    return int(score) if isinstance(score, (int, float)) else None


def normalize(raw: list[dict]) -> list[dict]:
    """One dict per game in the same shape for every sport (NFL nests id/date/status under "game")."""
    out = []
    for g in raw:
        try:
            core = g.get("game") or g
            ts = core.get("timestamp") or (core.get("date") or {}).get("timestamp")
            league = g["league"]
            country = (g.get("country") or league.get("country") or {}).get("name") or ""
            out.append({"id": core["id"], "kickoff": datetime.fromtimestamp(int(ts), timezone.utc).isoformat(),
                        "status": core["status"]["short"], "league_id": league["id"], "league": league["name"],
                        "country": country, "season": league.get("season"),
                        "home": g["teams"]["home"]["name"], "away": g["teams"]["away"]["name"],
                        "hs": _total(g["scores"]["home"]), "as": _total(g["scores"]["away"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _dirs(sport: str):
    return settings.data_dir / "cache" / "api_sports" / sport, settings.data_dir / "api_sports" / sport


def record(sport: str, games: list[dict]) -> None:
    done = [g for g in games if g["status"] in FINISHED and g["hs"] is not None and g["as"] is not None]
    if not done:
        return
    path = _dirs(sport)[1] / "results.json"
    seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    seen.update({str(g["id"]): g for g in done})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")


def games(sport: str, day: date) -> list[dict]:
    """Every game of one date: fresh cache, else a request within the quota, else the last saved list."""
    path = _dirs(sport)[0] / f"games_{day.isoformat()}.json"
    hit = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if hit and (time.time() - hit["fetched_at"]) / 3600 < GAMES_TTL_H:
        return hit["games"]
    if not can_spend(sport):
        return hit["games"] if hit else []
    state[sport]["used"] += 1  # a refused call counts too
    try:
        found = normalize(_get(sport, "games", date=day.isoformat())["response"])
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("API-Sports %s: jogos de %s indisponíveis (%s)", sport, day, e)
        return hit["games"] if hit else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "games": found}, ensure_ascii=False), encoding="utf-8")
    record(sport, found)
    return found


def history_season(season) -> str | int:
    """The free plan's latest season in the league's own format: "2024-2025" or 2024."""
    return "2024-2025" if isinstance(season, str) and "-" in season else 2024


def backfill(sport: str, leagues: list[tuple[int, object]]) -> None:
    """2024 results of leagues that have none yet, a few per run and within the quota. `leagues`: (id, season)."""
    hist_dir = _dirs(sport)[1]
    todo = [(lg, history_season(s)) for lg, s in leagues]
    todo = [(lg, s) for lg, s in todo if not (hist_dir / f"season_{lg}_{s}.json").exists()]
    for lg, season in todo[:HISTORY_PER_RUN]:
        if not can_spend(sport, 1 + GAMES_KEEP):
            return
        state[sport]["used"] += 1
        try:
            found = [g for g in normalize(_get(sport, "games", league=lg, season=season)["response"])
                     if g["status"] in FINISHED and g["hs"] is not None and g["as"] is not None]
        except ValueError as e:  # not covered: remember it
            found = []
            log.info("API-Sports %s: sem histórico %s para a liga %s (%s)", sport, season, lg, e)
        except httpx.HTTPError as e:
            log.warning("API-Sports %s: histórico da liga %s indisponível (%s)", sport, lg, e)
            continue
        hist_dir.mkdir(parents=True, exist_ok=True)
        (hist_dir / f"season_{lg}_{season}.json").write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")


def history(sport: str, league_id: int) -> pd.DataFrame:
    """Finished games of a league in the football-data column names (FTHG/FTAG = final points or goals)."""
    hist_dir = _dirs(sport)[1]
    found = []
    for path in hist_dir.glob(f"season_{league_id}_*.json"):
        found += json.loads(path.read_text(encoding="utf-8"))
    results = hist_dir / "results.json"
    if results.exists():
        found += [g for g in json.loads(results.read_text(encoding="utf-8")).values() if g["league_id"] == league_id]
    if not found:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = pd.DataFrame(found).drop_duplicates("id")
    return pd.DataFrame({"Date": pd.to_datetime(df.kickoff, utc=True).dt.tz_localize(None).dt.normalize(),
                         "HomeTeam": df.home, "AwayTeam": df.away, "FTHG": df.hs.astype(int),
                         "FTAG": df["as"].astype(int)}).sort_values("Date")


def upcoming(sport: str, now: datetime | None = None) -> list[dict]:
    """Not-started games from 2 h ago to the end of tomorrow (UTC); yesterday's list is fetched for its results."""
    now = now or datetime.now(timezone.utc)
    today = now.date()
    games(sport, today - timedelta(days=1))
    found = games(sport, today) + games(sport, today + timedelta(days=1))
    cutoff = now - timedelta(hours=2)
    return [g for g in found if g["status"] in NOT_STARTED and datetime.fromisoformat(g["kickoff"]) >= cutoff]
