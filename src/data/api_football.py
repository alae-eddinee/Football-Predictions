"""
API-Football v3 client  (api-sports.io)
========================================

Base URL : https://v3.football.api-sports.io
Auth     : x-apisports-key header
Plan     : Free — 100 requests / day

Endpoints implemented
---------------------
  GET /status                        account + usage
  GET /leagues                       league metadata + coverage
  GET /fixtures                      full match list for a league/season
  GET /fixtures/statistics           per-fixture team stats (shots, xG…)
  GET /fixtures/lineups              starting XI, formation, subs
  GET /fixtures/headtohead           historical H2H between two teams
  GET /injuries                      injury/suspension list per fixture
  GET /standings                     league table with home/away splits + form
  GET /predictions                   Poisson-based H/D/A %s + comparison data
  GET /teams                         team metadata + venue

Usage budget awareness
----------------------
All fetch methods support a cache layer (Parquet files in data/external/).
When cache exists and force_refresh=False, no API call is made.
A RateLimitGuard tracks calls per run and warns at 80 requests consumed.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://v3.football.api-sports.io"

# football-data.co.uk league code  →  API-Football league ID
LEAGUE_ID_MAP: Dict[str, int] = {
    "E0": 39,    # Premier League
    "E1": 40,    # Championship
    "SP1": 140,  # La Liga
    "D1": 78,    # Bundesliga
    "I1": 135,   # Serie A
    "F1": 61,    # Ligue 1
    "N1": 88,    # Eredivisie
    "P1": 94,    # Primeira Liga
}

# Season string  →  API-Football season year
SEASON_YEAR_MAP: Dict[str, int] = {
    "2018-19": 2018,
    "2019-20": 2019,
    "2020-21": 2020,
    "2021-22": 2021,
    "2022-23": 2022,
    "2023-24": 2023,
    "2024-25": 2024,
}

FREE_PLAN_DAILY_LIMIT = 100


# ---------------------------------------------------------------------------
# Rate-limit guard
# ---------------------------------------------------------------------------

class _RateLimitGuard:
    def __init__(self, daily_limit: int = FREE_PLAN_DAILY_LIMIT):
        self.daily_limit = daily_limit
        self._calls_this_run = 0

    def tick(self) -> None:
        self._calls_this_run += 1
        if self._calls_this_run >= int(self.daily_limit * 0.8):
            logger.warning(
                "API-Football: %d/%d daily requests used in this run — "
                "approaching limit, consider using cache.",
                self._calls_this_run, self.daily_limit,
            )

    @property
    def calls(self) -> int:
        return self._calls_this_run


_guard = _RateLimitGuard()


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class APIFootballClient:
    """
    Full-featured API-Football v3 client.

    All public methods accept an optional `cache_dir` and `force_refresh`
    to avoid burning through the 100 req/day free quota unnecessarily.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_seconds: float = 0.4,
    ):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")
        if not self.api_key:
            logger.warning("No API_FOOTBALL_KEY found. Set it in .env")
        self._rate = rate_limit_seconds
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "x-apisports-key": self.api_key,
        })

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Rate-limited GET. Returns parsed JSON or None on error."""
        elapsed = time.time() - self._last
        if elapsed < self._rate:
            time.sleep(self._rate - elapsed)

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            self._last = time.time()
            _guard.tick()
            data = resp.json()
            if data.get("errors"):
                logger.error("API error on %s %s: %s", endpoint, params, data["errors"])
                return None
            return data
        except requests.RequestException as exc:
            logger.error("Request failed — %s %s: %s", endpoint, params, exc)
            return None

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return account info and today's request usage."""
        data = self._get("status", {})
        if not data:
            return {}
        info = data["response"]
        used = info["requests"]["current"]
        limit = info["requests"]["limit_day"]
        logger.info(
            "API-Football — Plan: %s  |  Requests today: %d/%d  |  Sub expires: %s",
            info["subscription"]["plan"],
            used, limit,
            info["subscription"]["end"][:10],
        )
        return info

    # ------------------------------------------------------------------
    # /leagues
    # ------------------------------------------------------------------

    def get_league_info(self, league_code: str, season: str) -> Optional[Dict]:
        """Return league metadata dict for one league/season."""
        league_id = LEAGUE_ID_MAP.get(league_code)
        season_year = SEASON_YEAR_MAP.get(season)
        if not league_id or not season_year:
            return None
        data = self._get("leagues", {"id": league_id, "season": season_year})
        if not data or not data["response"]:
            return None
        return data["response"][0]

    # ------------------------------------------------------------------
    # /fixtures
    # ------------------------------------------------------------------

    def get_fixtures(
        self,
        league_code: str,
        season: str,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch all fixtures for a league/season.

        Returns DataFrame with columns:
            fixture_id, date, status, referee, venue_name, venue_city,
            home_team_id, home_team, away_team_id, away_team,
            home_goals, away_goals, ht_home_goals, ht_away_goals,
            league_id, league_code, season, round
        """
        league_id = LEAGUE_ID_MAP.get(league_code)
        season_year = SEASON_YEAR_MAP.get(season)
        if not league_id or not season_year:
            logger.debug("No mapping for %s / %s", league_code, season)
            return pd.DataFrame()

        cache_path = (
            cache_dir / f"fixtures_{league_code}_{season_year}.parquet"
            if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            logger.debug("Cache hit: %s", cache_path)
            return pd.read_parquet(cache_path)

        data = self._get("fixtures", {"league": league_id, "season": season_year})
        if not data:
            return pd.DataFrame()

        records = []
        for f in data.get("response", []):
            fix = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            score = f.get("score", {})
            lg = f.get("league", {})
            records.append({
                "fixture_id":      fix.get("id"),
                "date":            pd.to_datetime(fix.get("date"), errors="coerce", utc=True).tz_localize(None) if fix.get("date") else pd.NaT,
                "status":          fix.get("status", {}).get("short"),
                "elapsed":         fix.get("status", {}).get("elapsed"),
                "referee":         fix.get("referee"),
                "venue_name":      fix.get("venue", {}).get("name"),
                "venue_city":      fix.get("venue", {}).get("city"),
                "round":           lg.get("round"),
                "home_team_id":    teams.get("home", {}).get("id"),
                "home_team":       teams.get("home", {}).get("name"),
                "away_team_id":    teams.get("away", {}).get("id"),
                "away_team":       teams.get("away", {}).get("name"),
                "home_goals":      goals.get("home"),
                "away_goals":      goals.get("away"),
                "ht_home_goals":   score.get("halftime", {}).get("home"),
                "ht_away_goals":   score.get("halftime", {}).get("away"),
                "et_home_goals":   score.get("extratime", {}).get("home"),
                "et_away_goals":   score.get("extratime", {}).get("away"),
                "league_id":       league_id,
                "league_code":     league_code,
                "season":          season,
            })

        df = pd.DataFrame(records)
        if not df.empty and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Cached %d fixtures → %s", len(df), cache_path)
        return df

    # ------------------------------------------------------------------
    # /fixtures/statistics
    # ------------------------------------------------------------------

    def get_fixture_stats(
        self,
        fixture_id: int,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Dict]]:
        """
        Return per-team stats for a fixture.

        Returns dict keyed by team_id:
            {
              team_id: {
                "team_name": ...,
                "shots_on_goal": ..., "shots_off_goal": ...,
                "total_shots": ..., "blocked_shots": ...,
                "shots_insidebox": ..., "shots_outsidebox": ...,
                "fouls": ..., "corner_kicks": ..., "offsides": ...,
                "ball_possession": ...,   # "65%" string → float 0.65
                "yellow_cards": ..., "red_cards": ...,
                "goalkeeper_saves": ...,
                "total_passes": ..., "passes_accurate": ...,
                "passes_pct": ...,
                "expected_goals": ...,   # xG if available
              }
            }
        """
        cache_path = (
            cache_dir / f"stats_{fixture_id}.json" if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            with open(cache_path) as f:
                return json.load(f)

        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return None

        result: Dict[str, Dict] = {}
        for team_data in data["response"]:
            team_id = str(team_data["team"]["id"])
            raw = {s["type"]: s["value"] for s in team_data.get("statistics", [])}
            result[team_id] = {
                "team_name":        team_data["team"]["name"],
                "shots_on_goal":    _int(raw.get("Shots on Goal")),
                "shots_off_goal":   _int(raw.get("Shots off Goal")),
                "total_shots":      _int(raw.get("Total Shots")),
                "blocked_shots":    _int(raw.get("Blocked Shots")),
                "shots_insidebox":  _int(raw.get("Shots insidebox")),
                "shots_outsidebox": _int(raw.get("Shots outsidebox")),
                "fouls":            _int(raw.get("Fouls")),
                "corner_kicks":     _int(raw.get("Corner Kicks")),
                "offsides":         _int(raw.get("Offsides")),
                "ball_possession":  _pct(raw.get("Ball Possession")),
                "yellow_cards":     _int(raw.get("Yellow Cards")),
                "red_cards":        _int(raw.get("Red Cards")),
                "goalkeeper_saves": _int(raw.get("Goalkeeper Saves")),
                "total_passes":     _int(raw.get("Total passes")),
                "passes_accurate":  _int(raw.get("Passes accurate")),
                "passes_pct":       _pct(raw.get("Passes %")),
                "expected_goals":   _float(raw.get("expected_goals") or raw.get("xG")),
            }

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(result, f)
        return result

    # ------------------------------------------------------------------
    # /fixtures/lineups
    # ------------------------------------------------------------------

    def get_fixture_lineups(
        self,
        fixture_id: int,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Dict]]:
        """
        Return lineup info keyed by team_id.

        Each value:
            {
              "team_name": ...,
              "formation": "4-3-3",
              "coach_name": ...,
              "start_xi": [{"player_id": .., "name": .., "number": .., "pos": ..}, ...],
              "substitutes": [...],
            }
        """
        cache_path = (
            cache_dir / f"lineups_{fixture_id}.json" if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            with open(cache_path) as f:
                return json.load(f)

        data = self._get("fixtures/lineups", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return None

        result: Dict[str, Dict] = {}
        for td in data["response"]:
            team_id = str(td["team"]["id"])
            result[team_id] = {
                "team_name":  td["team"]["name"],
                "formation":  td.get("formation"),
                "coach_name": td.get("coach", {}).get("name"),
                "start_xi": [
                    {
                        "player_id": p["player"]["id"],
                        "name":      p["player"]["name"],
                        "number":    p["player"]["number"],
                        "pos":       p["player"]["pos"],
                        "grid":      p["player"]["grid"],
                    }
                    for p in td.get("startXI", [])
                ],
                "substitutes": [
                    {
                        "player_id": p["player"]["id"],
                        "name":      p["player"]["name"],
                        "number":    p["player"]["number"],
                        "pos":       p["player"]["pos"],
                    }
                    for p in td.get("substitutes", [])
                ],
            }

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(result, f)
        return result

    # ------------------------------------------------------------------
    # /fixtures/headtohead
    # ------------------------------------------------------------------

    def get_h2h(
        self,
        team1_id: int,
        team2_id: int,
        last: int = 10,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Head-to-head history between two teams (last N fixtures).

        Returns DataFrame: fixture_id, date, home_team, home_team_id,
            away_team, away_team_id, home_goals, away_goals
        """
        cache_path = (
            cache_dir / f"h2h_{min(team1_id,team2_id)}_{max(team1_id,team2_id)}.parquet"
            if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        data = self._get(
            "fixtures/headtohead",
            {"h2h": f"{team1_id}-{team2_id}", "last": last},
        )
        if not data or not data.get("response"):
            return pd.DataFrame()

        records = []
        for f in data["response"]:
            fix = f.get("fixture", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            records.append({
                "fixture_id":   fix.get("id"),
                "date":         pd.to_datetime(fix.get("date"), errors="coerce", utc=True).tz_localize(None),
                "home_team":    teams.get("home", {}).get("name"),
                "home_team_id": teams.get("home", {}).get("id"),
                "away_team":    teams.get("away", {}).get("name"),
                "away_team_id": teams.get("away", {}).get("id"),
                "home_goals":   goals.get("home"),
                "away_goals":   goals.get("away"),
            })

        df = pd.DataFrame(records)
        if not df.empty and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df

    # ------------------------------------------------------------------
    # /injuries
    # ------------------------------------------------------------------

    def get_injuries(
        self,
        fixture_id: int,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Injury/suspension list for a fixture.

        Returns DataFrame: fixture_id, team_id, team_name,
            player_id, player_name, injury_type, injury_reason
        """
        cache_path = (
            cache_dir / f"injuries_{fixture_id}.parquet" if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        data = self._get("injuries", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return pd.DataFrame()

        records = []
        for item in data["response"]:
            p = item.get("player", {})
            t = item.get("team", {})
            records.append({
                "fixture_id":    fixture_id,
                "team_id":       t.get("id"),
                "team_name":     t.get("name"),
                "player_id":     p.get("id"),
                "player_name":   p.get("name"),
                "injury_type":   p.get("type"),
                "injury_reason": p.get("reason"),
            })

        df = pd.DataFrame(records)
        if not df.empty and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df

    # ------------------------------------------------------------------
    # /standings
    # ------------------------------------------------------------------

    def get_standings(
        self,
        league_code: str,
        season: str,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Full league standings table.

        Returns DataFrame per team with:
            rank, team_id, team_name, points, played, wins, draws, losses,
            goals_for, goals_against, goal_diff, form,
            home_played, home_wins, home_draws, home_losses, home_gf, home_ga,
            away_played, away_wins, away_draws, away_losses, away_gf, away_ga
        """
        league_id = LEAGUE_ID_MAP.get(league_code)
        season_year = SEASON_YEAR_MAP.get(season)
        if not league_id or not season_year:
            return pd.DataFrame()

        cache_path = (
            cache_dir / f"standings_{league_code}_{season_year}.parquet"
            if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        data = self._get("standings", {"league": league_id, "season": season_year})
        if not data or not data.get("response"):
            return pd.DataFrame()

        table = data["response"][0]["league"]["standings"][0]
        records = []
        for t in table:
            all_ = t.get("all", {})
            home_ = t.get("home", {})
            away_ = t.get("away", {})
            records.append({
                "league_code":  league_code,
                "season":       season,
                "rank":         t.get("rank"),
                "team_id":      t["team"]["id"],
                "team_name":    t["team"]["name"],
                "points":       t.get("points"),
                "played":       all_.get("played"),
                "wins":         all_.get("win"),
                "draws":        all_.get("draw"),
                "losses":       all_.get("lose"),
                "goals_for":    all_.get("goals", {}).get("for"),
                "goals_against":all_.get("goals", {}).get("against"),
                "goal_diff":    t.get("goalsDiff"),
                "form":         t.get("form", ""),       # e.g. "WWLDD"
                "home_played":  home_.get("played"),
                "home_wins":    home_.get("win"),
                "home_draws":   home_.get("draw"),
                "home_losses":  home_.get("lose"),
                "home_gf":      home_.get("goals", {}).get("for"),
                "home_ga":      home_.get("goals", {}).get("against"),
                "away_played":  away_.get("played"),
                "away_wins":    away_.get("win"),
                "away_draws":   away_.get("draw"),
                "away_losses":  away_.get("lose"),
                "away_gf":      away_.get("goals", {}).get("for"),
                "away_ga":      away_.get("goals", {}).get("against"),
                "update":       t.get("update"),
            })

        df = pd.DataFrame(records)
        if not df.empty and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df

    # ------------------------------------------------------------------
    # /predictions
    # ------------------------------------------------------------------

    def get_predictions(
        self,
        fixture_id: int,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict]:
        """
        API-Football's own Poisson-based predictions for a fixture.

        Returns dict:
            {
              "winner_id": ..., "winner_name": ..., "advice": ...,
              "percent_home": 45, "percent_draw": 25, "percent_away": 30,
              "comparison": {
                "form":   {"home": "80%", "away": "60%"},
                "att":    {"home": "65%", "away": "35%"},
                "def":    {"home": "45%", "away": "55%"},
                "poisson_distribution": {"home": "52%", "away": "33%"},
                "h2h":    {"home": "60%", "away": "40%"},
                "goals":  {"home": "55%", "away": "45%"},
                "total":  {"home": "54%", "away": "46%"},
              },
              "h2h": [...]   # last few H2H fixtures
            }
        """
        cache_path = (
            cache_dir / f"predictions_{fixture_id}.json" if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            with open(cache_path) as f:
                return json.load(f)

        data = self._get("predictions", {"fixture": fixture_id})
        if not data or not data.get("response"):
            return None

        r = data["response"][0]
        pred = r.get("predictions", {})
        comp = r.get("comparison", {})

        result = {
            "winner_id":    pred.get("winner", {}).get("id"),
            "winner_name":  pred.get("winner", {}).get("name"),
            "advice":       pred.get("advice"),
            "win_or_draw":  pred.get("win_or_draw"),
            "percent_home": _pct_str(pred.get("percent", {}).get("home")),
            "percent_draw": _pct_str(pred.get("percent", {}).get("draw")),
            "percent_away": _pct_str(pred.get("percent", {}).get("away")),
            "comparison":   comp,
            "h2h":          r.get("h2h", []),
        }

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(result, f, default=str)
        return result

    # ------------------------------------------------------------------
    # /teams
    # ------------------------------------------------------------------

    def get_teams(
        self,
        league_code: str,
        season: str,
        cache_dir: Optional[Path] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Team metadata + venue for a league/season.

        Returns DataFrame: team_id, team_name, team_code, founded,
            venue_name, venue_city, venue_capacity
        """
        league_id = LEAGUE_ID_MAP.get(league_code)
        season_year = SEASON_YEAR_MAP.get(season)
        if not league_id or not season_year:
            return pd.DataFrame()

        cache_path = (
            cache_dir / f"teams_{league_code}_{season_year}.parquet"
            if cache_dir else None
        )
        if cache_path and cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        data = self._get("teams", {"league": league_id, "season": season_year})
        if not data or not data.get("response"):
            return pd.DataFrame()

        records = []
        for item in data["response"]:
            t = item.get("team", {})
            v = item.get("venue", {})
            records.append({
                "team_id":        t.get("id"),
                "team_name":      t.get("name"),
                "team_code":      t.get("code"),
                "founded":        t.get("founded"),
                "national":       t.get("national"),
                "league_code":    league_code,
                "season":         season,
                "venue_name":     v.get("name"),
                "venue_city":     v.get("city"),
                "venue_capacity": v.get("capacity"),
                "venue_surface":  v.get("surface"),
            })

        df = pd.DataFrame(records)
        if not df.empty and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df

    # ------------------------------------------------------------------
    # Composite helpers
    # ------------------------------------------------------------------

    def fetch_full_season(
        self,
        league_code: str,
        season: str,
        cache_dir: Path,
        force_refresh: bool = False,
        include_stats: bool = False,      # costs 1 req/fixture — use carefully
        include_predictions: bool = False, # costs 1 req/fixture
    ) -> Dict[str, Any]:
        """
        Pull all data for one league/season in one call.

        Returns:
            {
              "fixtures":   pd.DataFrame,
              "standings":  pd.DataFrame,
              "teams":      pd.DataFrame,
            }

        If include_stats=True, also fetches per-fixture statistics.
        If include_predictions=True, also fetches Poisson predictions.
        Note: each fixture costs 1 extra API request.
        """
        result: Dict[str, Any] = {}
        result["fixtures"]  = self.get_fixtures(league_code, season, cache_dir, force_refresh)
        result["standings"] = self.get_standings(league_code, season, cache_dir, force_refresh)
        result["teams"]     = self.get_teams(league_code, season, cache_dir, force_refresh)

        if include_stats and not result["fixtures"].empty:
            logger.info("Fetching per-fixture stats (%d fixtures)…", len(result["fixtures"]))
            stats_list = []
            for fid in result["fixtures"]["fixture_id"].dropna():
                s = self.get_fixture_stats(int(fid), cache_dir, force_refresh)
                if s:
                    for team_id, st in s.items():
                        stats_list.append({"fixture_id": fid, "team_id": team_id, **st})
            result["fixture_stats"] = pd.DataFrame(stats_list)

        if include_predictions and not result["fixtures"].empty:
            logger.info("Fetching predictions (%d fixtures)…", len(result["fixtures"]))
            pred_list = []
            for fid in result["fixtures"]["fixture_id"].dropna():
                p = self.get_predictions(int(fid), cache_dir, force_refresh)
                if p:
                    pred_list.append({"fixture_id": fid, **p})
            result["predictions"] = pd.DataFrame(pred_list)

        return result


# ---------------------------------------------------------------------------
# Fatigue computation (no API calls required)
# ---------------------------------------------------------------------------

def compute_team_fatigue(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add days_since_last_home and days_since_last_away columns.

    Iterates over chronologically sorted matches and tracks each team's
    last appearance date.  Zero leakage — only past matches counted.
    """
    matches = matches.copy().sort_values("date").reset_index(drop=True)
    last_match: Dict[str, pd.Timestamp] = {}
    days_home_list = []
    days_away_list = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        days_home_list.append(
            int((date - last_match[home]).days) if home in last_match else None
        )
        days_away_list.append(
            int((date - last_match[away]).days) if away in last_match else None
        )
        last_match[home] = date
        last_match[away] = date

    matches["days_since_last_home"] = days_home_list
    matches["days_since_last_away"] = days_away_list
    return matches


def enrich_with_standings(
    matches: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge standings features (form string, home/away record, rank) into matches.

    Joins on team_name × season.  Adds columns:
        home_rank, home_points, home_form_str, home_home_win_rate_season,
        away_rank, away_points, away_form_str, away_away_win_rate_season
    """
    if standings_df.empty:
        return matches

    def _win_rate(played, wins):
        if played and played > 0:
            return round(wins / played, 4)
        return None

    standings_df = standings_df.copy()
    standings_df["home_win_rate_season"]  = standings_df.apply(
        lambda r: _win_rate(r["home_played"], r["home_wins"]), axis=1)
    standings_df["away_win_rate_season"]  = standings_df.apply(
        lambda r: _win_rate(r["away_played"], r["away_wins"]), axis=1)
    standings_df["form_wins_last5"] = standings_df["form"].apply(
        lambda f: f.count("W") if isinstance(f, str) else None)

    home_cols = {
        "team_name": "home_team",
        "rank":      "home_standing_rank",
        "points":    "home_standing_pts",
        "form":      "home_form_str",
        "home_win_rate_season": "home_win_rate_season",
        "form_wins_last5": "home_form_wins_last5",
    }
    away_cols = {
        "team_name": "away_team",
        "rank":      "away_standing_rank",
        "points":    "away_standing_pts",
        "form":      "away_form_str",
        "away_win_rate_season": "away_win_rate_season",
        "form_wins_last5": "away_form_wins_last5",
    }

    for col_map in [home_cols, away_cols]:
        left_on = col_map["team_name"]
        right_on = "team_name"
        merge_cols = list(col_map.keys())
        sub = standings_df[merge_cols + ["season"]].copy()
        sub = sub.rename(columns={k: v for k, v in col_map.items() if k != "team_name"})
        matches = matches.merge(
            sub,
            left_on=[left_on, "season"],
            right_on=["team_name", "season"],
            how="left",
            suffixes=("", "_drop"),
        )
        drop_cols = [c for c in matches.columns if c.endswith("_drop")]
        matches.drop(columns=drop_cols, inplace=True)

    return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int(val) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _pct(val) -> Optional[float]:
    """Convert '65%' or 65 → 0.65."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip().rstrip("%")
    try:
        return round(float(val) / 100, 4)
    except (TypeError, ValueError):
        return None


def _pct_str(val) -> Optional[float]:
    """Convert '45%' string to float 0.45."""
    return _pct(val)
