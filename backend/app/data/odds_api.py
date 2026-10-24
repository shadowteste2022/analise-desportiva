"""Optional live odds from The Odds API (https://the-odds-api.com). Requires ODDS_API_KEY.

Each league call costs (markets x regions) = 2 credits. Only h2h (1X2) and totals are requested; BTTS needs
per-event calls and is left out. Spending is paced so the monthly budget lasts the whole billing period:
a call is made only if credits used so far plus this call fit within the share of the budget earned by now
(plus one full refresh of slack). Otherwise the last odds saved on disk are reused while not too old.
"""
import calendar
import difflib
import json
import logging
import time
from datetime import datetime, timezone

import httpx

from app.config import settings

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports"
URL = BASE + "/{sport}/odds"
COST_PER_CALL = 2  # markets (h2h, totals) x regions (eu)
SPORTS = {"E0": "soccer_epl", "SP1": "soccer_spain_la_liga", "I1": "soccer_italy_serie_a",
          "D1": "soccer_germany_bundesliga", "F1": "soccer_france_ligue_one",
          "P1": "soccer_portugal_primeira_liga", "N1": "soccer_netherlands_eredivisie",
          "E1": "soccer_efl_champ", "B1": "soccer_belgium_first_div", "SC0": "soccer_spl",
          "T1": "soccer_turkey_super_league", "G1": "soccer_greece_super_league"}
_cache: dict[str, tuple[float, list]] = {}
quota = {"remaining": None, "used": None, "budget": None, "allowed_now": None, "skipped": []}


def enabled() -> bool:
    return bool(settings.odds_api_key)


def _read_headers(r: httpx.Response) -> None:
    if "x-requests-used" in r.headers:
        quota["used"] = int(float(r.headers["x-requests-used"]))
        quota["remaining"] = int(float(r.headers.get("x-requests-remaining", 0)))


def _refresh_quota() -> None:
    """Current usage from /sports, which costs no credits. Done once per process."""
    if quota["used"] is not None:
        return
    try:
        r = httpx.get(BASE, params={"apiKey": settings.odds_api_key}, timeout=20)
        r.raise_for_status()
        _read_headers(r)
    except httpx.HTTPError as e:
        log.warning("The Odds API: não foi possível ler a quota (%s)", type(e).__name__)


def period_fraction(now: datetime, reset_day: int) -> float:
    """Share of the current billing period already elapsed (periods start on `reset_day` of each month)."""
    def at(year, month):
        return datetime(year, month, min(reset_day, calendar.monthrange(year, month)[1]), tzinfo=timezone.utc)
    start = at(now.year, now.month)
    if now < start:
        start = at(now.year - (now.month == 1), 12 if now.month == 1 else now.month - 1)
    end = at(start.year + (start.month == 12), 1 if start.month == 12 else start.month + 1)
    return (now - start) / (end - start)


def can_spend(cost: int, now: datetime | None = None) -> bool:
    """True if this call keeps spending on pace for the budget and above the safety reserve."""
    _refresh_quota()
    if quota["used"] is None:
        return False  # usage unknown: do not risk going over the plan
    budget = settings.odds_api_monthly_budget
    slack = COST_PER_CALL * len(settings.league_list)  # one full refresh
    allowed = budget * period_fraction(now or datetime.now(timezone.utc), settings.odds_api_reset_day) + slack
    quota.update(budget=budget, allowed_now=round(min(allowed, budget)))
    return (quota["used"] + cost <= min(allowed, budget)
            and quota["remaining"] - cost >= settings.odds_api_min_remaining)


def _disk(league: str):
    return settings.data_dir / "odds_api" / f"{league}.json"


def _load_disk(league: str) -> tuple[float, list] | None:
    try:
        d = json.loads(_disk(league).read_text(encoding="utf-8"))
        return d["fetched_at"], d["events"]
    except (OSError, ValueError, KeyError):
        return None


def fetch(league: str) -> list[dict]:
    """Raw events for a league: fresh cache, else a paced API call, else the last saved odds if not too old."""
    sport = SPORTS.get(league)
    if not enabled() or not sport:
        return []
    hit = _cache.get(league) or _load_disk(league)
    age_h = (time.time() - hit[0]) / 3600 if hit else None
    if hit and age_h < settings.fixtures_ttl_hours:
        return hit[1]
    stale = hit[1] if hit and age_h < settings.odds_max_age_hours else []
    if not can_spend(COST_PER_CALL):
        quota["skipped"].append(league)
        log.info("The Odds API: %s adiado para respeitar o orçamento (%s/%s créditos usados)",
                 league, quota["used"], quota["budget"])
        return stale
    try:
        r = httpx.get(URL.format(sport=sport), timeout=20, params={
            "apiKey": settings.odds_api_key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"})
        _read_headers(r)
        r.raise_for_status()
        events = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("The Odds API indisponível para %s: %s", league, type(e).__name__)  # never log the key
        return stale
    _cache[league] = (time.time(), events)
    _disk(league).parent.mkdir(parents=True, exist_ok=True)
    _disk(league).write_text(json.dumps({"fetched_at": time.time(), "events": events}), encoding="utf-8")
    return events


# football-data.co.uk abbreviations -> common names used by odds feeds
ALIASES = {"sp lisbon": "sporting", "sp braga": "braga", "man united": "manchester united",
           "man city": "manchester city", "nott'm forest": "nottingham forest", "sheffield weds": "sheffield wednesday",
           "ath madrid": "atletico madrid", "ath bilbao": "athletic bilbao", "m'gladbach": "monchengladbach",
           "ein frankfurt": "eintracht frankfurt", "fc koln": "koln", "paris sg": "paris saint germain",
           "st etienne": "saint etienne", "vitoria": "vitoria guimaraes", "guimaraes": "vitoria guimaraes"}
NOISE = {"fc", "cf", "sc", "cp", "ac", "afc", "sv", "ssc", "as", "club", "de", "fk"}


def _norm(name: str) -> str:
    n = name.lower().replace(".", "").replace("-", " ").strip()
    n = ALIASES.get(n, n)
    return " ".join(w for w in n.split() if w not in NOISE)


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    return 1.0 if a and b and (a in b or b in a) else difflib.SequenceMatcher(None, a, b).ratio()


def match_odds(events: list[dict], home: str, away: str) -> tuple[dict, str | None]:
    """Odds for one fixture in the same shape as football_data.row_odds, plus last_update.
    Team names differ between sources, so matching is fuzzy: both teams must pass and the best pair wins."""
    scored = [(min(_similarity(home, e["home_team"]), _similarity(away, e["away_team"])), k)
              for k, e in enumerate(events)]
    scored = [s for s in scored if s[0] >= 0.6]
    if scored:
        ev = events[max(scored)[1]]
        out: dict = {"1X2": {}}
        last = None
        for bk in ev.get("bookmakers", []):
            last = max(last or "", bk.get("last_update", ""))
            for mk in bk.get("markets", []):
                prices = {o["name"]: o["price"] for o in mk["outcomes"]}
                if mk["key"] == "h2h" and {ev["home_team"], ev["away_team"], "Draw"} <= prices.keys():
                    out["1X2"][bk["title"]] = {"1": prices[ev["home_team"]], "X": prices["Draw"],
                                               "2": prices[ev["away_team"]]}
                elif mk["key"] == "totals":
                    pts = {o.get("point") for o in mk["outcomes"]}
                    if len(pts) == 1 and {"Over", "Under"} <= prices.keys():
                        out.setdefault(f"OU{pts.pop()}", {})[bk["title"]] = {
                            "Mais": prices["Over"], "Menos": prices["Under"]}
        return out, last
    return {}, None
