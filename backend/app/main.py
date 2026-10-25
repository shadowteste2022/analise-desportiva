"""HTTP API + static UI. Run: uvicorn app.main:app --reload (from backend/)."""
import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app import analysis, backtest, db, sports
from app.config import ROOT, settings
from app.data import odds_api, odds_api_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
FRONTEND = ROOT / "frontend"
LEAGUE_RE = r"^[A-Z]{1,3}[0-9]?$"


async def _refresh_loop():
    while True:
        try:
            await run_in_threadpool(analysis.list_fixtures)
            log.info("Jogos e odds atualizados")
        except Exception:  # keep the loop alive; the error is logged with traceback
            log.exception("Falha na atualização automática")
        await asyncio.sleep(settings.refresh_minutes * 60)


@asynccontextmanager
async def lifespan(_):
    task = asyncio.create_task(_refresh_loop()) if settings.refresh_minutes > 0 else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Análise Desportiva — MVP Futebol", lifespan=lifespan)


def require_admin(x_admin_token: str = Header(default="")):
    if not settings.admin_token:
        raise HTTPException(403, "Administração desativada: defina ADMIN_TOKEN no .env")
    if not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(401, "Token de administração inválido")


def check_league(league: str) -> str:
    if league not in settings.league_list:
        raise HTTPException(404, f"Liga não configurada: {league}")
    return league


def _unavailable(e: Exception):
    log.warning("Fonte indisponível: %s", e)
    return HTTPException(503, f"Fonte de dados indisponível: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok", "odds_api": odds_api.enabled(), "odds_api_quota": odds_api.quota,
            "odds_api_io_pt": odds_api_io.enabled(), "pt_bookmakers": odds_api_io.bookmakers(),
            "admin": bool(settings.admin_token)}


@app.get("/api/fixtures")
def fixtures(league: str | None = Query(None, pattern=LEAGUE_RE)):
    """Football from football-data plus the other sports from The Odds API (when ODDS_API_KEY is set)."""
    if league:
        check_league(league)
    try:
        items, meta = analysis.list_fixtures(league)
    except (RuntimeError, ValueError) as e:
        raise _unavailable(e)
    if not league:
        items += [analysis.summarize(r) for r in sports.reports()]
    return {"fixtures": items, "source": meta, "odds_api": odds_api.quota}


@app.get("/api/match")
def match(key: str = Query(..., max_length=200)):
    if not sports.is_other_sport(key):
        try:
            row, meta = analysis.find_fixture(key)
        except (RuntimeError, ValueError) as e:
            raise _unavailable(e)
        report = analysis.analyze(row, meta) if row is not None else None
    else:
        report = sports.find(key)
    if report is None:
        raise HTTPException(404, "Jogo não encontrado nos próximos jogos")
    report["alerts"] = analysis.public_alerts(db.odds_moves(key))
    report["odds_history"] = analysis.odds_timeline(key)
    return report


@app.get("/api/history")
def history(limit: int = Query(500, ge=1, le=5000)):
    return analysis.settled_history(limit)


@app.get("/api/performance")
def performance():
    return analysis.performance(analysis.settled_history(5000))


@app.get("/api/alerts")
def alerts():
    return analysis.public_alerts(db.odds_moves())


@app.get("/api/backtests")
def backtests():
    return backtest.load_all()


@app.post("/api/admin/backtest", dependencies=[Depends(require_admin)])
async def run_backtest(test_seasons: int = Query(backtest.TEST_SEASONS, ge=2, le=10)):
    """All configured leagues at once: the combination weights are fitted across leagues. Takes minutes."""
    try:
        reports = await run_in_threadpool(backtest.run_all, backtest.backtestable(), test_seasons)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(422, str(e))
    return [{k: r[k] for k in ("league", "period", "predictions", "accuracy", "warnings")} for r in reports]


@app.post("/api/admin/refresh", dependencies=[Depends(require_admin)])
async def refresh():
    """Force re-download on next access by expiring the cache files."""
    for p in (settings.data_dir / "cache").glob("*.csv"):
        os.utime(p, (0, 0))
    analysis._models.clear()
    odds_api._cache.clear()
    return {"status": "cache expirada"}




@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
