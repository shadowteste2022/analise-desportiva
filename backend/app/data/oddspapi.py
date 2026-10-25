"""Odds from SRIJ-licensed bookmakers (Betano PT, Bwin PT) via OddsPapi (https://oddspapi.io). Requires ODDSPAPI_KEY.

Free plan: 250 requests per billing period. Verified with the API on 24/10/2026:
- /v4/account is free and reports `request_limit`, `request_count` and the subscription start (`valid_from`);
- /v4/odds-by-tournaments takes one bookmaker and at most 5 tournament ids, so tournaments go in priority batches;
- failed calls count too, and back-to-back calls can get HTTP 429; the free quota resets on the 1st of each month;
- /v4/participants maps participant ids to team names (one request, cached for a week).
Spending is paced over the billing period like The Odds API. Responses carry every market (~85 KB per game),
so only the markets the app uses are kept on disk.
"""
import calendar
import json
import logging
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from app.config import settings
from app.data.odds_api import _similarity

log = logging.getLogger(__name__)
BASE = "https://api.oddspapi.io/v4"
# SRIJ-licensed books in OddsPapi. Bwin PT ("bwin.pt") is listed but returned no odds even for a Premier League game
# (24/10/2026), so it is left out: every call costs a request.
BOOKS = {"betano.pt": "Betano PT"}
# market id -> (app market, {outcome id: selection}); ids from /v4/markets (sportId 10, full time)
MARKETS = {"101": ("1X2", {"101": "1", "102": "X", "103": "2"}), "104": ("BTTS", {"104": "Sim", "105": "Não"}),
           "106": ("OU0.5", {"106": "Mais", "107": "Menos"}), "108": ("OU1.5", {"108": "Mais", "109": "Menos"}),
           "1010": ("OU2.5", {"1010": "Mais", "1011": "Menos"}), "1012": ("OU3.5", {"1012": "Mais", "1013": "Menos"}),
           "1014": ("OU4.5", {"1014": "Mais", "1015": "Menos"})}
# tournament id -> competition name shown; ids from /v4/tournaments (sportId 10), checked 24/10/2026
TOURNAMENTS = {
    238: "Portugal — Liga Portugal", 239: "Portugal — Liga Portugal 2", 336: "Portugal — Taça de Portugal",
    327: "Portugal — Taça da Liga", 19274: "Portugal — Liga BPI (Feminino)", 53650: "Portugal — Taça de Portugal (Feminino)",
    7: "UEFA Liga dos Campeões", 679: "UEFA Liga Europa", 34480: "UEFA Liga Conferência",
    696: "UEFA Liga dos Campeões (Feminino)",
    17: "Inglaterra — Premier League", 18: "Inglaterra — Championship", 24: "Inglaterra — League One",
    25: "Inglaterra — League Two", 173: "Inglaterra — National League", 19: "Inglaterra — Taça de Inglaterra",
    21: "Inglaterra — Taça da Liga", 8: "Espanha — La Liga", 54: "Espanha — Segunda División",
    329: "Espanha — Taça do Rei", 1127: "Espanha — Liga F (Feminino)", 23: "Itália — Serie A", 53: "Itália — Serie B",
    328: "Itália — Taça de Itália", 20808: "Itália — Serie A (Feminino)", 35: "Alemanha — Bundesliga",
    44: "Alemanha — 2. Bundesliga", 217: "Alemanha — Taça da Alemanha", 34: "França — Ligue 1", 182: "França — Ligue 2",
    1139: "França — Première Ligue (Feminino)", 37: "Países Baixos — Eredivisie",
    2270: "Países Baixos — Eredivisie (Feminino)", 38: "Bélgica — Pro League", 52: "Turquia — Süper Lig",
    185: "Grécia — Super League", 36: "Escócia — Premiership", 206: "Escócia — Championship",
    20: "Noruega — Eliteserien", 40: "Suécia — Allsvenskan", 39: "Dinamarca — Superliga", 45: "Áustria — Bundesliga",
    215: "Suíça — Super League", 202: "Polónia — Ekstraklasa", 325: "Brasil — Série A", 390: "Brasil — Série B",
    155: "Argentina — Liga Profesional", 242: "EUA — MLS", 1690: "EUA — NWSL (Feminino)", 27464: "México — Liga MX",
    28953: "México — Liga MX (Feminino)", 196: "Japão — J1 League", 955: "Arábia Saudita — Pro League",
}
# The API accepts at most 5 tournament ids per request (INVALID_PARAMETER otherwise), so tournaments go in batches,
# most relevant to Portugal first. Each batch costs one request per bookmaker; the budget decides how far it goes.
BATCHES = [
    [238, 239, 7, 679, 34480],     # Liga Portugal, Liga Portugal 2, Champions, Europa, Conference League
    [17, 8, 23, 35, 34],           # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    [19274, 696, 1127, 325, 242],  # Liga BPI, Women's Champions League, Liga F, Brasileirão, MLS
    [18, 54, 53, 44, 182],         # second divisions of England, Spain, Italy, Germany, France
    [37, 38, 52, 36, 185],         # Eredivisie, Belgium, Turkey, Scotland, Greece
]
ODDS_TTL_H = 6  # a batch's odds are reused for 6 h: the free plan allows ~8 requests a day
PAUSE_S = 3  # between paid requests; back-to-back calls got HTTP 429, and failed calls still count
RESERVE = 10  # never spend the last requests of the plan
WEEK_H = 24 * 7

state = {"used": None, "limit": None, "reset_day": None, "error": None, "skipped": []}
_consumed: set[str] = set()  # fixtures already merged into another source's game this run


def enabled() -> bool:
    return bool(settings.oddspapi_key)


_last_call = [0.0]


def _get(path: str, paid: bool = True, **params):
    if paid:  # space out billable calls
        time.sleep(max(0.0, _last_call[0] + PAUSE_S - time.time()))
        _last_call[0] = time.time()
    r = httpx.get(f"{BASE}/{path}", params={"apiKey": settings.oddspapi_key, **params}, timeout=120)
    r.raise_for_status()
    return r.json()


def _path(name: str):
    return settings.data_dir / "oddspapi" / f"{name}.json"


def _load(name: str, max_age_h: float):
    try:
        d = json.loads(_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if (time.time() - d.get("fetched_at", 0)) / 3600 < max_age_h else None


def _save(name: str, payload) -> None:
    _path(name).parent.mkdir(parents=True, exist_ok=True)
    _path(name).write_text(json.dumps({"fetched_at": time.time(), "data": payload}, ensure_ascii=False),
                           encoding="utf-8")


def _refresh_account() -> None:
    """Usage from /v4/account, which costs no request. Once per process."""
    if state["used"] is not None or state["error"]:
        return
    try:
        sub = _get("account", paid=False)["subscriptions"][0]
        # the free quota resets on the 1st of each month (OddsPapi docs, "Requests & Quota")
        state.update(used=int(sub["request_count"]), limit=int(sub["request_limit"]),
                     reset_day=settings.oddspapi_reset_day)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
        state["error"] = f"conta indisponível ({type(e).__name__})"
        log.warning("OddsPapi: %s", state["error"])


def _period_fraction(now: datetime, day: int) -> float:
    def at(y, m):
        return datetime(y, m, min(day, calendar.monthrange(y, m)[1]), tzinfo=timezone.utc)
    start = at(now.year, now.month)
    if now < start:
        start = at(now.year - (now.month == 1), 12 if now.month == 1 else now.month - 1)
    end = at(start.year + (start.month == 12), 1 if start.month == 12 else start.month + 1)
    return (now - start) / (end - start)


def can_spend(cost: int = 1, now: datetime | None = None) -> bool:
    """Paced over the billing period (requests earned so far plus one full refresh), never into the reserve."""
    _refresh_account()
    if state["used"] is None:
        return False
    limit = min(state["limit"], settings.oddspapi_monthly_budget)
    earned = limit * _period_fraction(now or datetime.now(timezone.utc), state["reset_day"] or settings.oddspapi_reset_day)
    return state["used"] + cost <= min(limit, earned + len(BOOKS)) and state["limit"] - state["used"] - cost >= RESERVE


def _spend() -> None:
    state["used"] = (state["used"] or 0) + 1


def _participants() -> dict:
    hit = _load("participants", WEEK_H)
    if hit:
        return hit["data"]
    stale = _load("participants", 24 * 365)
    if not can_spend():
        return stale["data"] if stale else {}
    try:
        names = _get("participants", sportId=10)
        _spend()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("OddsPapi: participantes indisponíveis (%s)", type(e).__name__)
        return stale["data"] if stale else {}
    _save("participants", names)
    return names


def parse(raw: list[dict], book: str) -> list[dict]:
    """Keep only active markets the app uses, as {market: {bookmaker: {selection: odds}}}."""
    out = []
    for f in raw:
        b = (f.get("bookmakerOdds") or {}).get(book) or {}
        if b.get("suspended") or not b.get("bookmakerIsActive", True):
            continue
        odds, changed = {}, []
        for mid, (market, sels) in MARKETS.items():
            m = (b.get("markets") or {}).get(mid)
            if not m:  # marketActive is often false on priced, open markets: rely on fixture/outcome flags
                continue
            prices = {}
            for oid, sel in sels.items():
                p = (((m.get("outcomes") or {}).get(oid) or {}).get("players") or {}).get("0") or {}
                if p.get("active") and isinstance(p.get("price"), (int, float)) and p["price"] > 1:
                    prices[sel] = float(p["price"])
                    changed.append(p.get("changedAt") or "")
            if len(prices) == len(sels):
                odds[market] = {BOOKS[book]: prices}
        if odds:
            out.append({"id": f["fixtureId"], "tournament": f.get("tournamentId"), "start": f.get("startTime"),
                        "p1": f.get("participant1Id"), "p2": f.get("participant2Id"), "odds": odds,
                        # last price change; the fixture's updatedAt is about its details, not the odds
                        "updated": max(changed) or f.get("updatedAt")})
    return out


def _batch(book: str, i: int, ids: list[int]) -> list[dict]:
    """One bookmaker's fixtures for one batch of tournaments: fresh cache (ODDS_TTL_H), else a paced request,
    else the last saved odds if not too old."""
    name = f"odds_{book}_{i}"
    hit = _load(name, ODDS_TTL_H)
    if hit:
        return hit["data"]
    stale = _load(name, settings.odds_max_age_hours)
    if not can_spend():
        state["skipped"].append(f"{BOOKS[book]} grupo {i + 1}")
        return stale["data"] if stale else []
    try:
        raw = _get("odds-by-tournaments", bookmaker=book, tournamentIds=",".join(map(str, ids)))
    except (httpx.HTTPError, ValueError) as e:
        _spend()  # failed calls count too (OddsPapi docs)
        resp = getattr(e, "response", None)
        state["error"] = f"odds {BOOKS[book]} grupo {i + 1}: HTTP {resp.status_code}: {resp.text[:160]}" if resp is not None \
            else f"odds {BOOKS[book]} grupo {i + 1}: {type(e).__name__}"
        log.warning("OddsPapi: %s", state["error"])
        return stale["data"] if stale else []
    _spend()
    rows = parse(raw if isinstance(raw, list) else raw.get("data", []), book)
    _save(name, rows)
    return rows


_fixtures_cache: list[dict] | None = None


def fixtures() -> list[dict]:
    """Both books merged per fixture, with team names, kick-off and competition."""
    global _fixtures_cache
    if _fixtures_cache is not None or not enabled():
        return _fixtures_cache or []
    names = _participants()
    merged: dict[str, dict] = {}
    rows = [f for i, ids in enumerate(BATCHES) for book in BOOKS for f in _batch(book, i, ids)]  # priority order
    for f in rows:
        g = merged.setdefault(f["id"], {**f, "odds": {}})
        for market, books in f["odds"].items():
            g["odds"].setdefault(market, {}).update(books)
    out = []
    now = pd.Timestamp.now(tz="UTC")
    for g in merged.values():
        home, away = names.get(str(g["p1"])), names.get(str(g["p2"]))
        try:
            kickoff = pd.Timestamp(g["start"])
        except (TypeError, ValueError):
            continue
        # same window as the other sources: from 2 h after kick-off back to 7 days ahead
        if home and away and now - pd.Timedelta(hours=2) < kickoff < now + pd.Timedelta(days=7):
            out.append({**g, "home": home, "away": away, "kickoff": kickoff,
                        "competition": TOURNAMENTS.get(g["tournament"], "Futebol")})
    _fixtures_cache = out
    return out


def match_odds(home: str, away: str, when) -> tuple[dict, str | None]:
    """PT odds for a game another source already lists: same day (±1) and both team names similar."""
    day = pd.Timestamp(when).tz_localize(None).normalize() if pd.Timestamp(when).tzinfo is None \
        else pd.Timestamp(when).tz_convert(None).normalize()
    best = None
    for f in fixtures():
        if abs((f["kickoff"].tz_convert(None).normalize() - day).days) > 1:
            continue
        s = min(_similarity(home, f["home"]), _similarity(away, f["away"]))
        if s >= 0.6 and (best is None or s > best[0]):
            best = (s, f)
    if not best:
        return {}, None
    _consumed.add(best[1]["id"])
    return best[1]["odds"], best[1].get("updated")


def unmatched() -> list[dict]:
    """Fixtures no other source lists (e.g. Liga Portugal 2, Liga BPI): shown on their own."""
    return [f for f in fixtures() if f["id"] not in _consumed]
