"""Odds from Portuguese-licensed (SRIJ) bookmakers via odds-api.io. Requires ODDS_API_IO_KEY.

Why this source: of the 10 active SRIJ sports-betting licensees, only Betano PT, Betclic PT and Bwin PT are
available in an odds API we found (checked 24/10/2026); The Odds API only carries their FR/UK versions. The rest
(Placard, Solverde, ESC Online, Casino Portugal, LeBull, GoldenPark, Retabet) have no feed, and scraping their sites
would breach their terms, so they are left out.

Plans: paid from £49/month (Solo = 2 bookmakers, 5,000 req/h); existing free keys 100 req/h. Per refresh this uses
1 request for the event list plus 1 per 10 fixtures.
"""
import logging
import time

import httpx
import pandas as pd

from app.config import settings
from app.data.odds_api import _similarity

log = logging.getLogger(__name__)
BASE = "https://api.odds-api.io/v3"
MARKETS = "ML,Totals,Goals Over/Under,Both Teams To Score"
_events: tuple[float, list] | None = None
_odds: dict[str, tuple[float, dict]] = {}


def enabled() -> bool:
    return bool(settings.odds_api_io_key)


def bookmakers() -> list[str]:
    return [b.strip() for b in settings.odds_api_io_bookmakers.split(",") if b.strip()]


def _get(path: str, **params):
    r = httpx.get(f"{BASE}{path}", params={"apiKey": settings.odds_api_io_key, **params}, timeout=30)
    r.raise_for_status()
    return r.json()


def _fresh(ts: float) -> bool:
    return time.time() - ts < settings.fixtures_ttl_hours * 3600


def _event_list() -> list[dict]:
    global _events
    if _events and _fresh(_events[0]):
        return _events[1]
    try:
        ev = _get("/events", sport="football", limit=5000)
        ev = ev.get("data", ev) if isinstance(ev, dict) else ev
    except (httpx.HTTPError, ValueError) as e:
        log.warning("odds-api.io indisponível (eventos): %s", type(e).__name__)  # never log the key
        return _events[1] if _events else []
    _events = (time.time(), ev)
    return ev


def find_event(events: list[dict], home: str, away: str, when) -> dict | None:
    """Same day (±1 for time zones) and both team names similar; the best pair wins."""
    day = pd.Timestamp(when).normalize()
    best = None
    for e in events:
        try:
            d = pd.Timestamp(e["date"]).tz_localize(None) if pd.Timestamp(e["date"]).tzinfo is None \
                else pd.Timestamp(e["date"]).tz_convert(None)
        except (KeyError, ValueError, TypeError):
            continue
        if abs((d.normalize() - day).days) > 1:
            continue
        s = min(_similarity(home, str(e.get("home", ""))), _similarity(away, str(e.get("away", ""))))
        if s >= 0.6 and (best is None or s > best[0]):
            best = (s, e)
    return best[1] if best else None


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 1 else None


def parse(event_odds: dict) -> tuple[dict, str | None]:
    """odds-api.io event odds -> {market: {bookmaker: {selection: odds}}} in the app's naming, plus latest update."""
    out: dict = {}
    last = None
    for book, mkts in (event_odds.get("bookmakers") or {}).items():
        for m in mkts or []:
            last = max(last or "", m.get("updatedAt") or "")
            name = m.get("name")
            for o in m.get("odds") or []:
                if name == "ML":
                    sel = {"1": _num(o.get("home")), "X": _num(o.get("draw")), "2": _num(o.get("away"))}
                    key = "1X2"
                elif name in ("Totals", "Goals Over/Under"):
                    try:
                        line = float(o.get("hdp"))
                    except (TypeError, ValueError):
                        continue
                    if line % 1 != 0.5:  # whole/quarter lines are Asian totals with pushes, not our markets
                        continue
                    sel, key = {"Mais": _num(o.get("over")), "Menos": _num(o.get("under"))}, f"OU{line}"
                elif name == "Both Teams To Score":
                    sel, key = {"Sim": _num(o.get("yes")), "Não": _num(o.get("no"))}, "BTTS"
                else:
                    continue
                if all(sel.values()):
                    out.setdefault(key, {}).setdefault(book, sel)
    return out, last or None


def prefetch(rows: list) -> None:
    """Batch-load odds for many fixtures (10 events per request) so analysing each one is a cache hit."""
    if not enabled():
        return
    events = _event_list()
    ids = []
    for r in rows:
        e = find_event(events, r["HomeTeam"], r["AwayTeam"], r["Date"])
        if e and not (str(e["id"]) in _odds and _fresh(_odds[str(e["id"])][0])):
            ids.append(str(e["id"]))
    for k in range(0, len(ids), 10):
        try:
            res = _get("/odds/multi", eventIds=",".join(ids[k:k + 10]), bookmakers=",".join(bookmakers()),
                       markets=MARKETS)
        except (httpx.HTTPError, ValueError) as e:
            log.warning("odds-api.io indisponível (odds): %s", type(e).__name__)
            return
        for ev in res if isinstance(res, list) else res.get("data", []):
            _odds[str(ev.get("id"))] = (time.time(), ev)


def match_odds(home: str, away: str, when) -> tuple[dict, str | None]:
    if not enabled():
        return {}, None
    e = find_event(_event_list(), home, away, when)
    if not e:
        return {}, None
    hit = _odds.get(str(e["id"]))
    if not (hit and _fresh(hit[0])):
        try:
            ev = _get("/odds", eventId=e["id"], bookmakers=",".join(bookmakers()), markets=MARKETS)
        except (httpx.HTTPError, ValueError) as err:
            log.warning("odds-api.io indisponível (odds): %s", type(err).__name__)
            return ({}, None) if not hit else parse(hit[1])
        hit = _odds[str(e["id"])] = (time.time(), ev)
    return parse(hit[1])
