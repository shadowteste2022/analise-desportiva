"""SQLite persistence: prediction log (for honest after-the-fact evaluation) and odds snapshots (for movement alerts).

shortcut: SQLite single file, fine for one user/process; move to PostgreSQL when running several workers.
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, match_key TEXT NOT NULL, league TEXT, match_date TEXT,
  home TEXT, away TEXT, market TEXT NOT NULL, selection TEXT NOT NULL, model_prob REAL NOT NULL,
  odds REAL, ev REAL, category TEXT NOT NULL, confidence TEXT, model_version TEXT NOT NULL,
  UNIQUE (match_key, market, selection, model_version, odds)
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
  id INTEGER PRIMARY KEY, match_key TEXT NOT NULL, market TEXT NOT NULL, selection TEXT NOT NULL,
  bookmaker TEXT NOT NULL, odds REAL NOT NULL, fetched_at TEXT NOT NULL,
  UNIQUE (match_key, market, selection, bookmaker, fetched_at)
);
"""


def connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.data_dir / "app.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_predictions(rows: list[dict]) -> None:
    """Append-only; the same prediction at the same odds is stored once."""
    with closing(connect()) as con, con:
        con.executemany(
            """INSERT OR IGNORE INTO predictions (created_at, match_key, league, match_date, home, away, market,
               selection, model_prob, odds, ev, category, confidence, model_version)
               VALUES (:created_at, :match_key, :league, :match_date, :home, :away, :market, :selection,
                       :model_prob, :odds, :ev, :category, :confidence, :model_version)""",
            [{"created_at": now(), **r} for r in rows])


def snapshot_odds(match_key: str, odds: dict, fetched_at: str) -> None:
    """Store a price only when it differs from the last stored one: hourly refreshes would otherwise
    duplicate unchanged odds and bloat the database that the daily job commits."""
    with closing(connect()) as con, con:
        last = {(r["market"], r["selection"], r["bookmaker"]): r["odds"] for r in con.execute(
            """SELECT market, selection, bookmaker, odds FROM odds_snapshots s WHERE match_key = ? AND fetched_at =
               (SELECT MAX(fetched_at) FROM odds_snapshots t WHERE t.match_key = s.match_key AND t.market = s.market
                AND t.selection = s.selection AND t.bookmaker = s.bookmaker)""", (match_key,))}
        rows = [(match_key, mk, sel, bk, o, fetched_at)
                for mk, books in odds.items() for bk, sels in books.items() for sel, o in sels.items()
                if last.get((mk, sel, bk)) != o]
        con.executemany("INSERT OR IGNORE INTO odds_snapshots (match_key, market, selection, bookmaker, odds, "
                        "fetched_at) VALUES (?,?,?,?,?,?)", rows)


def prune_snapshots(days: int = 14) -> None:
    """Odds history only matters before kick-off; predictions are kept forever."""
    with closing(connect()) as con, con:
        # match_key = "LEAGUE|YYYY-MM-DD|home|away"
        con.execute("DELETE FROM odds_snapshots WHERE substr(match_key, instr(match_key, '|') + 1, 10) "
                    "< date('now', ?)", (f"-{days} day",))
    with closing(connect()) as con:
        con.execute("VACUUM")


def odds_moves(match_key: str | None = None) -> list[dict]:
    """First vs latest odds per (match, market, selection, bookmaker), keeping moves above the alert threshold."""
    sql = """
      WITH s AS (SELECT *, ROW_NUMBER() OVER w_asc AS a, ROW_NUMBER() OVER w_desc AS d FROM odds_snapshots
                 WHERE (:mk IS NULL OR match_key = :mk)
                 WINDOW w_asc AS (PARTITION BY match_key, market, selection, bookmaker ORDER BY fetched_at),
                        w_desc AS (PARTITION BY match_key, market, selection, bookmaker ORDER BY fetched_at DESC))
      SELECT f.match_key, f.market, f.selection, f.bookmaker, f.odds AS first_odds, l.odds AS last_odds,
             f.fetched_at AS first_at, l.fetched_at AS last_at
      FROM s f JOIN s l USING (match_key, market, selection, bookmaker) WHERE f.a = 1 AND l.d = 1"""
    with closing(connect()) as con:
        rows = [dict(r) for r in con.execute(sql, {"mk": match_key})]
    for r in rows:
        r["change"] = r["last_odds"] / r["first_odds"] - 1
    return [r for r in rows if abs(r["change"]) >= settings.odds_move_alert]


def history(limit: int = 500) -> list[dict]:
    with closing(connect()) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM predictions ORDER BY match_date DESC, id DESC LIMIT ?", (limit,))]
