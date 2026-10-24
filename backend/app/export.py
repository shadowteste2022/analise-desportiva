"""Static export for GitHub Pages: runs the same analysis as the API and writes the page plus JSON files.
Run hourly by .github/workflows/daily.yml. Locally: python -m app.export --out ../site
"""
import argparse
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app import analysis, backtest, db, sports
from app.config import ROOT, settings
from app.data import football_data as fd, odds_api, odds_api_io, oddspapi

FRONTEND = ROOT / "frontend"
log = logging.getLogger(__name__)


def slug(key: str) -> str:
    """File-safe name for a match key; the page computes the same one (frontend/index.html, staticPath)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would make the whole file unreadable in the browser, so fail the run instead
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")


def export(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    fx, meta = fd.load_fixtures()
    fx = fx[fx.Div.isin(settings.league_list)]
    odds_api_io.prefetch([r for _, r in fx.iterrows()])
    summaries = []
    # both log the prediction and odds snapshot, like the API. Other sports first: they get the scarce Odds API
    # credits before football-data leagues, which already carry odds from six bookmakers in their CSVs.
    try:
        others = sports.reports()
    except Exception:  # one failing source must not stop the whole site from being published
        log.exception("Falha ao analisar as outras modalidades")
        others = []
    football = []
    for _, row in fx.iterrows():
        try:
            football.append(analysis.analyze(row, meta))
        except Exception:  # same reason: one bad fixture is skipped
            log.exception("Falha ao analisar %s – %s (%s)", row.get("HomeTeam"), row.get("AwayTeam"), row.get("Div"))
    portuguese = []  # games only the Portuguese books list; after the others, which take the games they share
    for f in oddspapi.unmatched():
        try:
            portuguese.append(sports.from_oddspapi(f))
        except Exception:
            log.exception("Falha ao analisar %s – %s (OddsPapi)", f.get("home"), f.get("away"))
    for report in others + football + portuguese:
        report["alerts"] = analysis.public_alerts(db.odds_moves(report["key"]))
        _write(data / "match" / f"{slug(report['key'])}.json", report)
        summaries.append(analysis.summarize(report))
    db.prune_snapshots()
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    live = {"enabled": odds_api.enabled(),
            **{k: odds_api.quota.get(k) for k in ("used", "remaining", "budget", "allowed_now", "error")},
            "skipped": sorted(set(odds_api.quota["skipped"]))}
    legal = {"pt_only": settings.pt_only,
             "sources": [n for n, on in (("odds-api.io", odds_api_io.enabled()),
                                         ("OddsPapi (Betano PT, Bwin PT)", oddspapi.enabled())) if on],
             "oddspapi": {k: oddspapi.state[k] for k in ("used", "limit", "error", "skipped")}}
    _write(data / "fixtures.json", {"fixtures": summaries, "source": meta, "generated_at": generated, "odds_api": live,
                                    "legal": legal})
    _write(data / "history.json", analysis.settled_history(5000))
    _write(data / "alerts.json", analysis.public_alerts(db.odds_moves()))
    _write(data / "backtests.json", backtest.load_all())
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    page = page.replace("</head>", "<script>window.STATIC = true;</script>\n</head>", 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    shutil.copytree(FRONTEND / "static", out / "static")
    (out / ".nojekyll").touch()  # serve files as-is, no Jekyll processing
    return {"fixtures": len(summaries), "generated_at": generated}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    print(export(ap.parse_args().out))
