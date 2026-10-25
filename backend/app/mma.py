"""MMA forecasts from API-MMA (same key as API-Football): an Elo rating per fighter.

Why Elo: fights are one against one and each fighter has few fights, so a rating that moves with every result is the
standard, data-light choice. History = the 2022-2024 seasons (the free plan's range, 3 requests in all) plus every
finished fight seen from now on. Only fighters with MIN_FIGHTS in that history get a forecast.
"""
import json
import logging
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from app import analysis
from app.data import api_sports as aps
from app.data.football_data import LOCAL_TZ
from app.data.odds_api import _similarity

log = logging.getLogger(__name__)
K = 40  # rating change per fight; MMA ratings move fast, fighters fight 2-3 times a year
MIN_FIGHTS = 2
SEASONS = (2022, 2023, 2024)


def normalize(raw: list[dict]) -> list[dict]:
    out = []
    for f in raw:
        try:
            a, b = f["fighters"]["first"], f["fighters"]["second"]
            winner = "1" if a.get("winner") else "2" if b.get("winner") else None
            out.append({"id": f["id"], "kickoff": datetime.fromtimestamp(int(f["timestamp"]), timezone.utc).isoformat(),
                        "status": f["status"]["short"], "event": f.get("slug") or "", "category": f.get("category") or "",
                        "home": a["name"], "away": b["name"], "winner": winner})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _dirs():
    return aps.settings.data_dir / "cache" / "api_sports" / "mma", aps.settings.data_dir / "api_sports" / "mma"


def _record(fights: list[dict]) -> None:
    done = [f for f in fights if f["winner"] and f["status"] in aps.FINISHED]
    if not done:
        return
    path = _dirs()[1] / "results.json"
    seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    seen.update({str(f["id"]): f for f in done})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, ensure_ascii=False), encoding="utf-8")


def fights(day: date) -> list[dict]:
    path = _dirs()[0] / f"fights_{day.isoformat()}.json"
    hit = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if hit and (time.time() - hit["fetched_at"]) / 3600 < aps.GAMES_TTL_H:
        return hit["fights"]
    if not aps.can_spend("mma"):
        return hit["fights"] if hit else []
    aps.state["mma"]["used"] += 1
    try:
        found = normalize(aps._get("mma", "fights", date=day.isoformat())["response"])
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("API-MMA: combates de %s indisponíveis (%s)", day, e)
        return hit["fights"] if hit else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "fights": found}, ensure_ascii=False), encoding="utf-8")
    _record(found)
    return found


def backfill() -> None:
    for season in SEASONS:
        path = _dirs()[1] / f"season_{season}.json"
        if path.exists() or not aps.can_spend("mma", 1 + aps.GAMES_KEEP):
            continue
        aps.state["mma"]["used"] += 1
        try:
            found = [f for f in normalize(aps._get("mma", "fights", season=season)["response"]) if f["winner"]]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("API-MMA: época %s indisponível (%s)", season, e)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")


def history() -> list[dict]:
    found = []
    for path in _dirs()[1].glob("season_*.json"):
        found += json.loads(path.read_text(encoding="utf-8"))
    results = _dirs()[1] / "results.json"
    if results.exists():
        found += list(json.loads(results.read_text(encoding="utf-8")).values())
    return sorted({f["id"]: f for f in found if f.get("winner")}.values(), key=lambda f: f["kickoff"])


def expected(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def ratings(fights: list[dict]) -> tuple[dict, dict, dict]:
    """Elo after every fight in order; also fights and wins per fighter."""
    elo, n, wins = defaultdict(lambda: 1500.0), defaultdict(int), defaultdict(int)
    for f in fights:
        a, b = f["home"], f["away"]
        e = expected(elo[a], elo[b])
        s = 1.0 if f["winner"] == "1" else 0.0
        elo[a] += K * (s - e)
        elo[b] -= K * (s - e)
        n[a] += 1; n[b] += 1
        wins[a if s else b] += 1
    return dict(elo), dict(n), dict(wins)


def report(fight: dict, elo: dict, n: dict, wins: dict) -> dict:
    kick = pd.Timestamp(fight["kickoff"]).tz_convert(LOCAL_TZ)
    home, away = fight["home"], fight["away"]
    key = f"aps_mma|{kick.date().isoformat()}|{home}|{away}"
    comp = " — ".join(x for x in (fight["event"], fight["category"]) if x) or "MMA"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ok = n.get(home, 0) >= MIN_FIGHTS and n.get(away, 0) >= MIN_FIGHTS
    conf = ("baixa" if min(n.get(home, 0), n.get(away, 0)) >= 4 else "muito baixa") if ok else None
    out = {"key": key, "markets": [], "odds": {}, "alerts": [], "recommendation": None, "statistics": None,
           "identification": {"sport": "MMA", "league": "aps_mma", "competition": comp, "home": home, "away": away,
                              "date": kick.date().isoformat(), "time": kick.strftime("%H:%M"), "kickoff": kick.isoformat(),
                              "venue": "—"},
           "data_status": {"results": {"source": "API-MMA", "matches": sum(n.values()) // 2, "last_result": None,
                                       "fetched_at": now}, "odds": [], "confidence": conf, "analysed_at": now,
                           "team_matches": {home: n.get(home, 0), away: n.get(away, 0)}},
           "predictions": None}
    notes = ["Previsão por rating Elo: só conta quem ganhou cada combate (não o método nem as estatísticas), com "
             "histórico de 2022 a 2024 mais os combates registados desde 26/10/2026.",
             "Lesões, mudança de categoria de peso e tempo de paragem não entram na previsão."]
    out["conclusion"] = analysis._conclusion([], conf, notes)
    if not ok:
        return out
    p = expected(elo[home], elo[away])
    out["predictions"] = {"probabilities": {"Vencedor": {"1": p, "2": 1 - p}}, "model_probabilities": None, "blend": None,
                          "expected_goals": None, "matrix": None, "top_scores": [],
                          "method": f"Rating Elo por lutador (K={K}), {sum(n.values()) // 2} combates no histórico"}
    fav, dog = (home, away) if p >= 0.5 else (away, home)
    out["explanation"] = [
        f"Rating Elo: {home} {elo[home]:.0f} ({wins.get(home, 0)} vitórias em {n[home]} combates), "
        f"{away} {elo[away]:.0f} ({wins.get(away, 0)} em {n[away]}). A média é 1500.",
        f"A diferença de {abs(elo[home] - elo[away]):.0f} pontos dá {max(p, 1 - p):.0%} a {fav} e {min(p, 1 - p):.0%} a {dog}.",
        "Confiança baixa: os lutadores têm poucos combates no histórico, a previsão pode estar longe da realidade."]
    return out


def reports(known: list[dict]) -> list[dict]:
    if not aps.enabled():
        return []
    try:
        today = datetime.now(timezone.utc).date()
        fights(today - timedelta(days=1))
        upcoming = fights(today) + fights(today + timedelta(days=1))
        backfill()
    except Exception:
        log.exception("API-MMA: falhou")
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    upcoming = [f for f in upcoming if f["status"] in aps.NOT_STARTED and datetime.fromisoformat(f["kickoff"]) >= cutoff
                and not any(k["date"] == f["kickoff"][:10] and _similarity(f["home"], k["home"]) >= 0.75
                            and _similarity(f["away"], k["away"]) >= 0.75 for k in known)]
    elo, n, wins = ratings(history())
    return [report(f, elo, n, wins) for f in upcoming]
