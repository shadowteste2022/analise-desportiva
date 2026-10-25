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
          "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "E3": "soccer_england_league2",
          "SP2": "soccer_spain_segunda_division", "I2": "soccer_italy_serie_b", "D2": "soccer_germany_bundesliga2",
          "F2": "soccer_france_ligue_two", "B1": "soccer_belgium_first_div", "SC0": "soccer_spl",
          "T1": "soccer_turkey_super_league", "G1": "soccer_greece_super_league",
          "BRA": "soccer_brazil_campeonato", "ARG": "soccer_argentina_primera_division", "USA": "soccer_usa_mls",
          "MEX": "soccer_mexico_ligamx", "JPN": "soccer_japan_j_league", "CHN": "soccer_china_superleague",
          "AUT": "soccer_austria_bundesliga", "SWZ": "soccer_switzerland_superleague",
          "DNK": "soccer_denmark_superliga", "NOR": "soccer_norway_eliteserien", "SWE": "soccer_sweden_allsvenskan",
          "FIN": "soccer_finland_veikkausliiga", "POL": "soccer_poland_ekstraklasa", "IRL": "soccer_league_of_ireland",
          "RUS": "soccer_russia_premier_league"}
_cache: dict[str, tuple[float, list]] = {}
active: list[str] = []  # sport keys the API reports as in season, from /sports
catalog: dict[str, dict] = {}  # key -> {group, title, description, has_outrights} for the active sports
_events_cache: dict[str, tuple[float, list]] = {}
_events_free = [True]  # set to False if /events ever reports a cost; then it is never called again
quota = {"remaining": None, "used": None, "budget": None, "allowed_now": None, "skipped": [], "error": None}


def enabled() -> bool:
    return bool(settings.odds_api_key)


def _read_headers(r: httpx.Response) -> None:
    if "x-requests-used" in r.headers:
        quota["used"] = int(float(r.headers["x-requests-used"]))
        quota["remaining"] = int(float(r.headers.get("x-requests-remaining", 0)))


def _error(e: httpx.HTTPError) -> str:
    """Status and the API's own message (e.g. an invalid key), never the request URL, which carries the key."""
    if isinstance(e, httpx.HTTPStatusError):
        text = e.response.text[:200]
        if settings.odds_api_key:
            text = text.replace(settings.odds_api_key, "***")  # the message ends up on the public site
        if e.response.status_code == 401 and settings.odds_api_key:  # helps spot a wrong value without showing it
            text += f" [a chave recebida tem {len(settings.odds_api_key)} caracteres; as da The Odds API têm 32]"
        return f"HTTP {e.response.status_code}: {text}"
    return type(e).__name__


def _refresh_quota() -> None:
    """Current usage from /sports, which costs no credits. Tried once per process, even if it fails."""
    if quota.get("checked"):
        return
    quota["checked"] = True
    try:
        r = httpx.get(BASE, params={"apiKey": settings.odds_api_key}, timeout=20)
        r.raise_for_status()
        _read_headers(r)
        sports = [s for s in r.json() if s.get("active")]
        active[:] = [s["key"] for s in sports]  # in-season sports: nothing else is requested
        catalog.update({s["key"]: s for s in sports})
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        if not isinstance(e, httpx.HTTPError):
            quota["error"] = f"resposta inesperada de /sports ({type(e).__name__})"
            log.warning("The Odds API: %s", quota["error"])
            return
        quota["error"] = _error(e)
        log.warning("The Odds API: não foi possível ler a quota (%s)", quota["error"])


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
    slack = COST_PER_CALL * sum(lg in SPORTS for lg in settings.league_list)  # one full refresh
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
    """Raw events for a football-data league code (see fetch_sport)."""
    sport = SPORTS.get(league)
    return fetch_sport(sport) if sport else []


def events(sport: str) -> list[dict]:
    """Upcoming events of a sport, without odds. The API documents /events as free (no quota), so every sport's
    fixtures can be listed even when the odds budget only covers a few; cached like the odds."""
    if not enabled() or not _events_free[0]:
        return []
    hit = _events_cache.get(sport) or _load_disk(f"events_{sport}")
    if hit and (time.time() - hit[0]) / 3600 < settings.fixtures_ttl_hours:
        return hit[1]
    try:
        r = httpx.get(f"{BASE}/{sport}/events", params={"apiKey": settings.odds_api_key}, timeout=20)
        _read_headers(r)
        r.raise_for_status()
        evs = r.json()
    except httpx.HTTPError as e:
        log.warning("The Odds API: jogos de %s indisponíveis (%s)", sport, _error(e))
        return hit[1] if hit else []
    except ValueError:
        return hit[1] if hit else []
    if float(r.headers.get("x-requests-last", 0) or 0) > 0:  # documented as free; stop at once if that changes
        _events_free[0] = False
        log.warning("The Odds API: /events passou a gastar créditos; deixa de ser usado")
    _events_cache[sport] = (time.time(), evs)
    _disk(f"events_{sport}").parent.mkdir(parents=True, exist_ok=True)
    _disk(f"events_{sport}").write_text(json.dumps({"fetched_at": time.time(), "events": evs}), encoding="utf-8")
    return evs


def fetch_sport(sport: str, markets: str = "h2h,totals") -> list[dict]:
    """Raw events with odds for an Odds API sport key: fresh cache, else a paced API call, else the last saved odds
    if not too old. Cost = number of markets (one region)."""
    if not enabled():
        return []
    _refresh_quota()
    if catalog and sport not in catalog:  # out of season or unknown key: costs credits for nothing
        return []
    hit = _cache.get(sport) or _load_disk(sport)
    age_h = (time.time() - hit[0]) / 3600 if hit else None
    if hit and age_h < settings.odds_ttl_hours:
        return hit[1]
    stale = hit[1] if hit and age_h < settings.odds_max_age_hours else []
    if not can_spend(len(markets.split(","))):
        quota["skipped"].append(sport)
        log.info("The Odds API: %s adiado para respeitar o orçamento (%s/%s créditos usados)",
                 sport, quota["used"], quota["budget"])
        return stale
    try:
        r = httpx.get(URL.format(sport=sport), timeout=20, params={
            "apiKey": settings.odds_api_key, "regions": "eu", "markets": markets, "oddsFormat": "decimal"})
        _read_headers(r)
        r.raise_for_status()
        events = r.json()
    except httpx.HTTPError as e:
        log.warning("The Odds API indisponível para %s: %s", sport, _error(e))
        return stale
    except ValueError as e:
        log.warning("The Odds API: resposta inválida para %s (%s)", sport, type(e).__name__)
        return stale
    _cache[sport] = (time.time(), events)
    _disk(sport).parent.mkdir(parents=True, exist_ok=True)
    _disk(sport).write_text(json.dumps({"fetched_at": time.time(), "events": events}), encoding="utf-8")
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
                    half = len(pts) == 1 and isinstance(next(iter(pts)), (int, float)) and next(iter(pts)) % 1 == 0.5
                    if half and {"Over", "Under"} <= prices.keys():  # whole lines can push; not modelled
                        out.setdefault(f"OU{pts.pop()}", {})[bk["title"]] = {
                            "Mais": prices["Over"], "Menos": prices["Under"]}
        return out, last
    return {}, None
