"""Ingestion from football-data.co.uk (free, no key): historical results, stats, xG, odds and upcoming fixtures."""
import io
import logging
import time
from datetime import date, datetime, timezone

import httpx
import pandas as pd

from app.config import settings

log = logging.getLogger(__name__)
BASE = "https://www.football-data.co.uk"
SOURCE = "football-data.co.uk"

LEAGUE_NAMES = {
    "E0": "Inglaterra — Premier League", "E1": "Inglaterra — Championship", "SP1": "Espanha — La Liga",
    "I1": "Itália — Serie A", "D1": "Alemanha — Bundesliga", "F1": "França — Ligue 1",
    "P1": "Portugal — Primeira Liga", "N1": "Países Baixos — Eredivisie", "B1": "Bélgica — Pro League",
    "SC0": "Escócia — Premiership", "T1": "Turquia — Süper Lig", "G1": "Grécia — Super League",
}
# Named bookmakers in the CSVs (Max/Avg are market aggregates, BFE is an exchange before commission).
BOOKMAKERS = {"B365": "Bet365", "BFD": "Betfred", "BV": "BetVictor", "BW": "Betway",
              "PP": "Paddy Power", "SKB": "Sky Bet", "BFE": "Betfair Exchange"}
REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam"]
TEXT_COLS = ("Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee")
ODDS_PREFIXES = tuple(BOOKMAKERS) + ("Max", "Avg")


def season_codes(today: date, n: int) -> list[str]:
    """football-data season codes, e.g. 2627 for 2026/27, newest last."""
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(y % 100):02d}{((y + 1) % 100):02d}" for y in range(start - n + 1, start + 1)]


def _fetch(url: str, name: str, ttl_hours: float) -> tuple[pd.DataFrame, dict]:
    """Download with a disk cache. Falls back to a stale cache if the source is down."""
    path = settings.data_dir / "cache" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and time.time() - path.stat().st_mtime < ttl_hours * 3600
    stale = False
    if not fresh:
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            if not r.content.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
                raise ValueError("resposta não parece um CSV do football-data")
            path.write_bytes(r.content)
        except (httpx.HTTPError, ValueError) as e:
            if not path.exists():
                raise RuntimeError(f"Fonte indisponível e sem cache: {url} ({e})") from e
            log.warning("Usando cache desatualizada para %s: %s", url, e)
            stale = True
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colunas em falta {missing}")
    fetched = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return df, {"source": SOURCE, "url": url, "fetched_at": fetched.isoformat(), "stale": stale}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and type a raw CSV: parse dates, coerce numbers, drop impossible odds."""
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")  # old seasons: dd/mm/yy
    df = df.dropna(subset=["Date"])
    for c in df.columns:
        if c not in TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c.startswith(ODDS_PREFIXES):
                df.loc[~(df[c] > 1.0), c] = float("nan")  # decimal odds must be > 1
    return df


def load_results(league: str, today: date | None = None, seasons: int | None = None) -> tuple[pd.DataFrame, dict]:
    """All played matches for the last `seasons` (default `seasons_back`) seasons, sorted by date."""
    today = today or date.today()
    current = season_codes(today, 1)[0]
    frames, metas = [], []
    for code in season_codes(today, seasons or settings.seasons_back):
        ttl = settings.results_ttl_hours if code == current else 24 * 30
        try:
            df, meta = _fetch(f"{BASE}/mmz4281/{code}/{league}.csv", f"{league}_{code}.csv", ttl)
        except (RuntimeError, ValueError) as e:
            log.warning("%s", e)  # e.g. season not started yet
            continue
        frames.append(df)
        metas.append(meta)
    if not frames:
        raise RuntimeError(f"Sem resultados disponíveis para {league}")
    df = clean(pd.concat(frames, ignore_index=True))
    df = df.dropna(subset=["FTHG", "FTAG"])
    df = df[(df.FTHG >= 0) & (df.FTAG >= 0)].sort_values("Date").reset_index(drop=True)
    meta = {"source": SOURCE, "files": metas, "matches": len(df),
            "last_result": df.Date.max().date().isoformat(), "stale": any(m["stale"] for m in metas)}
    return df, meta


def load_fixtures() -> tuple[pd.DataFrame, dict]:
    """Upcoming matches (next days) for all covered leagues, with pre-match odds."""
    df, meta = _fetch(f"{BASE}/fixtures.csv", "fixtures.csv", settings.fixtures_ttl_hours)
    return clean(df).sort_values(["Date", "Time"]).reset_index(drop=True), meta


AGGREGATES = {"Max": "Máximo do mercado", "Avg": "Média do mercado"}


def row_odds(row, closing: bool = False) -> dict[str, dict[str, dict[str, float]]]:
    """{market: {bookmaker: {selection: odds}}} for the markets the CSVs carry (1X2, O/U 2.5)."""
    c = "C" if closing else ""
    out: dict = {"1X2": {}, "OU2.5": {}}
    for code, name in {**BOOKMAKERS, **AGGREGATES}.items():
        h, d, a = (row.get(f"{code}{c}{s}") for s in "HDA")
        if all(x is not None and x > 1 for x in (h, d, a)):  # NaN > 1 is False
            out["1X2"][name] = {"1": float(h), "X": float(d), "2": float(a)}
        o, u = row.get(f"{code}{c}>2.5"), row.get(f"{code}{c}<2.5")
        if all(x is not None and x > 1 for x in (o, u)):
            out["OU2.5"][name] = {"Mais": float(o), "Menos": float(u)}
    return out


def match_key(league: str, d, home: str, away: str) -> str:
    return f"{league}|{pd.Timestamp(d).date().isoformat()}|{home}|{away}"
