"""
Football Predictor — FastAPI backend
=====================================

Endpoints
---------
GET  /api/status                 model + API health
GET  /api/fixtures?date=YYYY-MM-DD&days=1   upcoming fixtures
GET  /api/predict/{fixture_id}   predict a specific fixture
POST /api/predict                predict from fixture dict payload
GET  /api/leagues                supported leagues
GET  /api/team/{name}            team insight (form, Elo, last match)
"""

import logging
import subprocess
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.api_football import APIFootballClient, LEAGUE_ID_MAP, SEASON_YEAR_MAP
from src.data.espn import fetch_fixtures as espn_fetch_fixtures, ESPN_LEAGUES
from src.data.football_data_org import FootballDataClient, LEAGUE_ID_TO_COMP as FDO_LEAGUE_IDS
from app.predictor import LivePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Alae's Pitch — Football Predictions",
    description="ML-powered football match outcome predictor and value bet detector by Alae-Eddine Dahane.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Shared resources (initialised on startup)
# ---------------------------------------------------------------------------

api_client: Optional[APIFootballClient] = None
fdo_client: Optional[FootballDataClient] = None
predictor: Optional[LivePredictor] = None

# ---------------------------------------------------------------------------
# Pipeline state (ingest + train background job)
# ---------------------------------------------------------------------------

_pipeline_state = {
    "status": "idle",        # idle | running | done | error
    "stage":  "",            # current stage label
    "log":    [],            # recent log lines (capped at 200)
    "error":  "",
}
_pipeline_lock = threading.Lock()


def _run_pipeline_bg():
    """Run ingest → train in a background thread, update _pipeline_state."""
    python = sys.executable
    cmd_base = [python, str(ROOT / "main.py")]

    def emit(line: str):
        with _pipeline_lock:
            _pipeline_state["log"].append(line)
            if len(_pipeline_state["log"]) > 200:
                _pipeline_state["log"].pop(0)

    try:
        with _pipeline_lock:
            _pipeline_state.update(status="running", stage="Ingesting data", log=[], error="")

        # ---- ingest ----
        proc = subprocess.Popen(
            cmd_base + ["ingest"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT),
        )
        for line in proc.stdout:
            emit(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ingest step failed")

        # ---- train ----
        with _pipeline_lock:
            _pipeline_state["stage"] = "Training model"

        proc = subprocess.Popen(
            cmd_base + ["train"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT),
        )
        for line in proc.stdout:
            emit(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("train step failed")

        # Reload predictor with the freshly trained model
        with _pipeline_lock:
            _pipeline_state["stage"] = "Loading model"
        global predictor
        predictor = LivePredictor.get().load()

        with _pipeline_lock:
            _pipeline_state.update(status="done", stage="Ready")

    except Exception as exc:
        with _pipeline_lock:
            _pipeline_state.update(status="error", stage="", error=str(exc))

SUPPORTED_LEAGUE_IDS = set(LEAGUE_ID_MAP.values())

# Free plan: seasons 2022-2024 only.
FREE_PLAN_MAX_SEASON = 2024

def _current_season_year() -> int:
    """
    Return the API-Football season year to use for 'current' fixtures.

    Tries each year from FREE_PLAN_MAX_SEASON downward until we find one
    that has fixtures near today's real date (used as fallback = max allowed).
    Season '2024-25' starts ~August 2024 and ends ~May 2025.
    """
    return FREE_PLAN_MAX_SEASON


def _date_to_season_year(d) -> int:
    """Given a calendar date, return the API-Football season year."""
    year = d.year
    # Jan-June → season started previous year
    if d.month < 7:
        year -= 1
    # Cap to free plan limit
    return min(year, FREE_PLAN_MAX_SEASON)

LEAGUE_DISPLAY = {
    39:  {"name": "Premier League",   "country": "England",  "code": "E0", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    40:  {"name": "Championship",     "country": "England",  "code": "E1", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
    140: {"name": "La Liga",          "country": "Spain",    "code": "SP1","flag": "🇪🇸"},
    78:  {"name": "Bundesliga",       "country": "Germany",  "code": "D1", "flag": "🇩🇪"},
    135: {"name": "Serie A",          "country": "Italy",    "code": "I1", "flag": "🇮🇹"},
    61:  {"name": "Ligue 1",          "country": "France",   "code": "F1", "flag": "🇫🇷"},
    88:  {"name": "Eredivisie",       "country": "Netherlands","code": "N1","flag": "🇳🇱"},
    94:  {"name": "Primeira Liga",    "country": "Portugal", "code": "P1", "flag": "🇵🇹"},
}


@app.on_event("startup")
async def startup():
    global api_client, fdo_client, predictor
    logger.info("Initialising API-Football client…")
    api_client = APIFootballClient()

    logger.info("Initialising football-data.org client…")
    try:
        fdo_client = FootballDataClient()
    except Exception as exc:
        logger.warning("football-data.org client unavailable: %s", exc)

    logger.info("Loading predictor (history + model)…")
    predictor = LivePredictor.get().load()
    logger.info("Server ready.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    fixture_id:  Optional[int] = None
    home_team:   str
    away_team:   str
    date:        str           # ISO: "2026-04-14"
    kickoff:     str = ""
    league:      str = ""
    league_code: str = "E0"
    season:      str = "2024-25"
    odds_home:   Optional[float] = None
    odds_draw:   Optional[float] = None
    odds_away:   Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def status():
    """Return model readiness + API-Football account info."""
    api_info = {}
    try:
        api_info = api_client.get_status()
    except Exception as e:
        api_info = {"error": str(e)}

    model_ready = predictor and predictor.model is not None
    history_size = len(predictor.history) if predictor and not predictor.history.empty else 0

    return {
        "model_ready":   model_ready,
        "history_size":  history_size,
        "teams_tracked": len(predictor.elo_state) if predictor else 0,
        "api_football":  api_info,
    }


@app.get("/api/leagues")
async def leagues():
    return {"leagues": LEAGUE_DISPLAY}


@app.get("/api/fixtures")
async def get_fixtures(
    date: Optional[str] = Query(None, description="YYYY-MM-DD, default today"),
    days: int = Query(1, ge=1, le=7, description="Days to look ahead"),
    league_id: Optional[int] = Query(None, description="Filter by league ID"),
):
    """
    Fetch fixtures using football-data.org (primary) + ESPN (fallback for unsupported leagues).

    football-data.org covers: Premier League, Championship, Bundesliga, Serie A, Ligue 1, Primeira Liga.
    ESPN covers the rest: La Liga, Eredivisie (and any other gap).
    """
    from_date = datetime.strptime(date, "%Y-%m-%d").date() if date else datetime.now().date()
    to_date   = from_date + timedelta(days=days - 1)

    flat_fixtures: List[Dict] = []
    seen_ids: set = set()

    # --- Primary: football-data.org ---
    fdo_league_ids = set(FDO_LEAGUE_IDS.keys())  # league_ids covered by FDO
    use_fdo = fdo_client and (league_id is None or league_id in fdo_league_ids)

    if use_fdo:
        fdo_fixtures = fdo_client.fetch_matches(from_date, to_date)
        for fx in fdo_fixtures:
            if league_id and fx["league_id"] != league_id:
                continue
            fid = fx.get("fixture_id")
            if fid not in seen_ids:
                seen_ids.add(fid)
                flat_fixtures.append(fx)

    # --- Fallback: ESPN for leagues not covered by FDO ---
    espn_only_ids = {140, 88}   # La Liga, Eredivisie
    need_espn = (
        league_id in espn_only_ids
        if league_id
        else True  # always fetch ESPN leagues when showing all
    )
    if need_espn:
        espn_lid = league_id if (league_id and league_id in espn_only_ids) else None
        espn_fixtures = espn_fetch_fixtures(from_date, to_date, league_id=espn_lid)
        for fx in espn_fixtures:
            if fx["league_id"] not in espn_only_ids:
                continue
            fid = fx.get("fixture_id")
            if fid not in seen_ids:
                seen_ids.add(fid)
                flat_fixtures.append(fx)

    # --- Group by league ---
    espn_league_by_id = {v["league_id"]: v for v in ESPN_LEAGUES.values()}
    grouped: Dict[int, Dict] = {}
    priority = [39, 140, 78, 135, 61, 40, 88, 94]

    for fx in flat_fixtures:
        lid = fx["league_id"]
        if lid not in grouped:
            linfo = LEAGUE_DISPLAY.get(lid) or espn_league_by_id.get(lid, {})
            grouped[lid] = {
                "league_id":   lid,
                "league_name": linfo.get("name", fx.get("league", "")),
                "country":     linfo.get("country", ""),
                "flag":        linfo.get("flag", ""),
                "fixtures":    [],
            }
        grouped[lid]["fixtures"].append(fx)

    # Sort fixtures within each league by kickoff time
    for lg in grouped.values():
        lg["fixtures"].sort(key=lambda f: f.get("kickoff", "") or "")

    result = sorted(
        grouped.values(),
        key=lambda x: priority.index(x["league_id"]) if x["league_id"] in priority else 99
    )

    return {
        "from_date": from_date.isoformat(),
        "to_date":   to_date.isoformat(),
        "total":     sum(len(lg["fixtures"]) for lg in result),
        "leagues":   result,
    }


@app.get("/api/predict/{fixture_id}")
async def predict_fixture(fixture_id: int):
    """Fetch fixture details then run prediction."""
    if not api_client:
        raise HTTPException(503, "API client not initialised")

    data = api_client._get("fixtures", {"id": fixture_id})
    if not data or not data.get("response"):
        raise HTTPException(404, f"Fixture {fixture_id} not found")

    f = data["response"][0]
    fix    = f.get("fixture", {})
    teams  = f.get("teams", {})
    goals  = f.get("goals", {})
    lg     = f.get("league", {})

    lid = lg.get("id")
    sy_raw = lg.get("season", FREE_PLAN_MAX_SEASON)
    season_str  = f"{sy_raw}-{str(sy_raw+1)[-2:]}"

    raw_dt = fix.get("date", "")
    try:
        dt = datetime.fromisoformat(raw_dt)
        kickoff = dt.strftime("%H:%M")
        match_date = dt.strftime("%Y-%m-%d")
    except Exception:
        kickoff = ""
        match_date = raw_dt[:10] if raw_dt else ""

    fixture_dict = {
        "fixture_id":  fixture_id,
        "date":        match_date,
        "kickoff":     kickoff,
        "home_team":   teams.get("home", {}).get("name", ""),
        "away_team":   teams.get("away", {}).get("name", ""),
        "league":      lg.get("name", ""),
        "league_code": LEAGUE_DISPLAY.get(lid, {}).get("code", "E0"),
        "season":      season_str,
    }

    prediction = predictor.predict(fixture_dict)
    home_insight = predictor.get_team_insight(fixture_dict["home_team"])
    away_insight = predictor.get_team_insight(fixture_dict["away_team"])

    return {
        "prediction": prediction.to_dict(),
        "home_insight": home_insight,
        "away_insight": away_insight,
        "fixture": fixture_dict,
    }


@app.post("/api/predict")
async def predict_custom(req: PredictRequest):
    """Predict from a custom payload (used by the UI when clicking a match card)."""
    if not predictor:
        raise HTTPException(503, "Predictor not initialised")

    fixture_dict = {
        "fixture_id":  req.fixture_id,
        "date":        req.date,
        "kickoff":     req.kickoff,
        "home_team":   req.home_team,
        "away_team":   req.away_team,
        "league":      req.league,
        "league_code": req.league_code,
        "season":      req.season,
    }

    bookmaker_odds = None
    if any([req.odds_home, req.odds_draw, req.odds_away]):
        bookmaker_odds = {
            "home": req.odds_home,
            "draw": req.odds_draw,
            "away": req.odds_away,
        }

    prediction = predictor.predict(fixture_dict, bookmaker_odds)
    home_insight = predictor.get_team_insight(req.home_team)
    away_insight = predictor.get_team_insight(req.away_team)

    return {
        "prediction": prediction.to_dict(),
        "home_insight": home_insight,
        "away_insight": away_insight,
    }


@app.post("/api/pipeline/run")
async def pipeline_run():
    """Start the ingest + train pipeline in the background."""
    with _pipeline_lock:
        if _pipeline_state["status"] == "running":
            return {"started": False, "message": "Pipeline already running"}
        _pipeline_state.update(status="queued", stage="Starting…", log=[], error="")

    t = threading.Thread(target=_run_pipeline_bg, daemon=True)
    t.start()
    return {"started": True}


@app.get("/api/pipeline/status")
async def pipeline_status():
    """Poll pipeline progress."""
    with _pipeline_lock:
        return {
            "status": _pipeline_state["status"],
            "stage":  _pipeline_state["stage"],
            "log":    _pipeline_state["log"][-50:],   # last 50 lines
            "error":  _pipeline_state["error"],
            "model_ready": predictor is not None and predictor.model is not None,
        }


@app.get("/api/team/{name}")
async def team_insight(name: str):
    if not predictor:
        raise HTTPException(503, "Predictor not initialised")
    return predictor.get_team_insight(name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT)],
        log_level="info",
    )
