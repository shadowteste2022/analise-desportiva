"""Static export for GitHub Pages: runs the same analysis as the API and writes the page plus JSON files.
Run hourly by .github/workflows/daily.yml. Locally: python -m app.export --out ../site
"""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app import analysis, backtest, db, sports
from app.config import ROOT, settings
from app.data import football_data as fd, odds_api, odds_api_io

FRONTEND = ROOT / "frontend"


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
    # both log the prediction and odds snapshot, like the API; football first so it gets the Odds API budget first
    for report in [analysis.analyze(row, meta) for _, row in fx.iterrows()] + sports.reports():
        report["alerts"] = db.odds_moves(report["key"])
        _write(data / "match" / f"{slug(report['key'])}.json", report)
        summaries.append(analysis.summarize(report))
    db.prune_snapshots()
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    live = {"enabled": odds_api.enabled(),
            **{k: odds_api.quota.get(k) for k in ("used", "remaining", "budget", "allowed_now", "error")},
            "skipped": sorted(set(odds_api.quota["skipped"]))}
    _write(data / "fixtures.json", {"fixtures": summaries, "source": meta, "generated_at": generated, "odds_api": live})
    _write(data / "history.json", analysis.settled_history(5000))
    _write(data / "alerts.json", db.odds_moves())
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
