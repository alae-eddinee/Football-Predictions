"""
ESPN Scoreboard API client
==========================
Unofficial ESPN API — no authentication required.
Returns real-time scheduled, live, and completed fixtures.

Base URL: https://site.api.espn.com/apis/site/v2/sports/soccer/{league_slug}/scoreboard
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("espn")

# ESPN league slug → (league_id, league_code, league_name, country, flag)
ESPN_LEAGUES: Dict[str, Dict] = {
    "eng.1": {
        "league_id":   39,
        "league_code": "E0",
        "name":        "Premier League",
        "country":     "England",
        "flag":        "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    },
    "eng.2": {
        "league_id":   40,
        "league_code": "E1",
        "name":        "Championship",
        "country":     "England",
        "flag":        "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    },
    "esp.1": {
        "league_id":   140,
        "league_code": "SP1",
        "name":        "La Liga",
        "country":     "Spain",
        "flag":        "🇪🇸",
    },
    "ger.1": {
        "league_id":   78,
        "league_code": "D1",
        "name":        "Bundesliga",
        "country":     "Germany",
        "flag":        "🇩🇪",
    },
    "ita.1": {
        "league_id":   135,
        "league_code": "I1",
        "name":        "Serie A",
        "country":     "Italy",
        "flag":        "🇮🇹",
    },
    "fra.1": {
        "league_id":   61,
        "league_code": "F1",
        "name":        "Ligue 1",
        "country":     "France",
        "flag":        "🇫🇷",
    },
    "ned.1": {
        "league_id":   88,
        "league_code": "N1",
        "name":        "Eredivisie",
        "country":     "Netherlands",
        "flag":        "🇳🇱",
    },
    "por.1": {
        "league_id":   94,
        "league_code": "P1",
        "name":        "Primeira Liga",
        "country":     "Portugal",
        "flag":        "🇵🇹",
    },
}

# Reverse lookup: league_id → ESPN slug
LEAGUE_ID_TO_SLUG: Dict[int, str] = {
    v["league_id"]: k for k, v in ESPN_LEAGUES.items()
}

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 FootballPredictor/1.0"})


def _fetch_scoreboard(slug: str, date_str: Optional[str] = None) -> List[Dict]:
    """
    Fetch raw ESPN scoreboard events for one league.

    Parameters
    ----------
    slug      : ESPN league slug e.g. "eng.1"
    date_str  : "YYYYMMDD" or None for today
    """
    url = BASE_URL.format(slug=slug)
    params = {}
    if date_str:
        params["dates"] = date_str

    try:
        resp = _SESSION.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception as exc:
        logger.warning("ESPN fetch failed for %s (date=%s): %s", slug, date_str, exc)
        return []


def _parse_event(event: Dict, league_info: Dict) -> Optional[Dict]:
    """Convert a raw ESPN event dict to the fixture dict format used by the server."""
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]

    # Teams
    home = away = None
    home_score = away_score = None
    for c in comp.get("competitors", []):
        team_name = c.get("team", {}).get("displayName", "")
        score_raw = c.get("score", None)
        score = int(score_raw) if score_raw is not None else None
        if c.get("homeAway") == "home":
            home = team_name
            home_score = score
        else:
            away = team_name
            away_score = score

    if not home or not away:
        return None

    # Date / kickoff
    raw_dt = comp.get("date") or event.get("date", "")
    try:
        dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
        match_date = dt.strftime("%Y-%m-%d")
        kickoff = dt.strftime("%H:%M")
    except Exception:
        match_date = raw_dt[:10] if len(raw_dt) >= 10 else ""
        kickoff = ""

    # Status
    status_obj = comp.get("status", {}).get("type", {})
    status_desc = status_obj.get("description", "Not Started")
    status_short = status_obj.get("shortDetail", "NS")
    completed = status_obj.get("completed", False)
    state = status_obj.get("state", "pre")   # pre / in / post

    # Map to API-Football-style short codes
    if state == "pre":
        status_short = "NS"
    elif state == "in":
        status_short = "1H" if comp.get("status", {}).get("period", 1) == 1 else "2H"
    elif completed:
        status_short = "FT"

    elapsed = None
    if state == "in":
        elapsed = int(comp.get("status", {}).get("clock", 0) / 60)

    # Season string
    season_year = event.get("season", {}).get("year", datetime.now().year)
    season_str = f"{season_year - 1}-{str(season_year)[-2:]}"

    return {
        "fixture_id":   int(event.get("id", 0)),
        "date":         match_date,
        "kickoff":      kickoff,
        "status_short": status_short,
        "status_long":  status_desc,
        "elapsed":      elapsed,
        "home_team":    home,
        "home_team_id": None,
        "away_team":    away,
        "away_team_id": None,
        "home_goals":   home_score if completed or state == "in" else None,
        "away_goals":   away_score if completed or state == "in" else None,
        "league_id":    league_info["league_id"],
        "league":       league_info["name"],
        "league_code":  league_info["league_code"],
        "season":       season_str,
        "round":        "",
    }


def fetch_fixtures(
    from_date: date,
    to_date: date,
    league_id: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch fixtures from ESPN for a date range.

    Returns a flat list of fixture dicts (same schema as the server uses).
    """
    slugs = (
        [LEAGUE_ID_TO_SLUG[league_id]]
        if league_id and league_id in LEAGUE_ID_TO_SLUG
        else list(ESPN_LEAGUES.keys())
    )

    # Build list of dates to query (ESPN scoreboard is per-day)
    days_range = (to_date - from_date).days + 1
    dates = [from_date + timedelta(days=i) for i in range(days_range)]
    date_strs = [d.strftime("%Y%m%d") for d in dates]

    all_fixtures: List[Dict] = []
    seen_ids: set = set()

    for slug in slugs:
        league_info = ESPN_LEAGUES[slug]
        for ds in date_strs:
            events = _fetch_scoreboard(slug, ds)
            for ev in events:
                fid = int(ev.get("id", 0))
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                parsed = _parse_event(ev, league_info)
                if parsed:
                    all_fixtures.append(parsed)

    return all_fixtures
