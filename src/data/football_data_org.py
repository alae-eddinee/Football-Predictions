"""
football-data.org API client (v4)
==================================
Auth: X-Auth-Token header
Rate limit: 10 requests/minute (free tier), signalled via x-requests-available-minute header
Docs: https://www.football-data.org/documentation/quickstart

Competitions available on free tier:
  PL  → Premier League      (id 2021)
  ELC → Championship        (id 2016)
  BL1 → Bundesliga          (id 2002)
  SA  → Serie A             (id 2019)
  FL1 → Ligue 1             (id 2015)
  PPL → Primeira Liga       (id 2017)
"""

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
logger = logging.getLogger("football_data_org")

BASE_URL = "https://api.football-data.org/v4"

# Competition code → (our league_id, our league_code, display info)
COMPETITIONS: Dict[str, Dict] = {
    "PL": {
        "league_id":   39,
        "league_code": "E0",
        "name":        "Premier League",
        "country":     "England",
        "flag":        "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    },
    "ELC": {
        "league_id":   40,
        "league_code": "E1",
        "name":        "Championship",
        "country":     "England",
        "flag":        "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    },
    "BL1": {
        "league_id":   78,
        "league_code": "D1",
        "name":        "Bundesliga",
        "country":     "Germany",
        "flag":        "🇩🇪",
    },
    "SA": {
        "league_id":   135,
        "league_code": "I1",
        "name":        "Serie A",
        "country":     "Italy",
        "flag":        "🇮🇹",
    },
    "FL1": {
        "league_id":   61,
        "league_code": "F1",
        "name":        "Ligue 1",
        "country":     "France",
        "flag":        "🇫🇷",
    },
    "PPL": {
        "league_id":   94,
        "league_code": "P1",
        "name":        "Primeira Liga",
        "country":     "Portugal",
        "flag":        "🇵🇹",
    },
}

# Reverse: our league_id → competition code
LEAGUE_ID_TO_COMP: Dict[int, str] = {
    v["league_id"]: k for k, v in COMPETITIONS.items()
}

# Status mapping: football-data.org → display short code
STATUS_MAP = {
    "SCHEDULED": "NS",
    "TIMED":     "NS",
    "IN_PLAY":   "LIVE",
    "PAUSED":    "HT",
    "FINISHED":  "FT",
    "POSTPONED": "PST",
    "CANCELLED": "CANC",
    "SUSPENDED": "SUSP",
}


class FootballDataClient:
    def __init__(self):
        self.key = os.getenv("FOOTBALL_DATA_KEY", "")
        if not self.key:
            raise RuntimeError("FOOTBALL_DATA_KEY not set in .env")
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Token": self.key,
            "User-Agent":   "FootballPredictor/1.0",
        })
        self._requests_remaining = 10  # conservative initial value

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{BASE_URL}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=15)

            # Respect rate limit header
            remaining = resp.headers.get("x-requests-available-minute")
            if remaining is not None:
                self._requests_remaining = int(remaining)
                if self._requests_remaining <= 1:
                    logger.warning("Rate limit nearly exhausted (%s remaining) — sleeping 10s", remaining)
                    time.sleep(10)

            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            logger.warning("HTTP %s for %s: %s", exc.response.status_code, url, exc.response.text[:200])
            return None
        except Exception as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    def fetch_matches(self, date_from: date, date_to: date) -> List[Dict]:
        """
        Fetch all matches across supported competitions for a date range.
        Returns flat list of fixture dicts in the server's standard schema.
        """
        # football-data.org uses exclusive dateTo — add 1 day to include the last requested day
        data = self._get("matches", {
            "dateFrom": date_from.isoformat(),
            "dateTo":   (date_to + timedelta(days=1)).isoformat(),
        })
        if not data:
            return []

        fixtures = []
        for m in data.get("matches", []):
            comp_code = m.get("competition", {}).get("code", "")
            if comp_code not in COMPETITIONS:
                continue
            parsed = self._parse_match(m, comp_code)
            if parsed:
                fixtures.append(parsed)
        return fixtures

    def _parse_match(self, m: Dict, comp_code: str) -> Optional[Dict]:
        info = COMPETITIONS[comp_code]

        home = m.get("homeTeam", {}).get("name", "") or m.get("homeTeam", {}).get("shortName", "")
        away = m.get("awayTeam", {}).get("name", "") or m.get("awayTeam", {}).get("shortName", "")
        if not home or not away:
            return None

        raw_dt = m.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            match_date = dt.strftime("%Y-%m-%d")
            kickoff = dt.strftime("%H:%M")
        except Exception:
            match_date = raw_dt[:10] if len(raw_dt) >= 10 else ""
            kickoff = ""

        fd_status = m.get("status", "SCHEDULED")
        status_short = STATUS_MAP.get(fd_status, "NS")

        score = m.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        season_obj = m.get("season", {})
        start_year_str = (season_obj.get("startDate") or "")[:4]
        try:
            start_year = int(start_year_str)
            season_str = f"{start_year}-{str(start_year + 1)[-2:]}"
        except Exception:
            season_str = ""

        return {
            "fixture_id":   m.get("id"),
            "date":         match_date,
            "kickoff":      kickoff,
            "status_short": status_short,
            "status_long":  fd_status.replace("_", " ").title(),
            "elapsed":      None,
            "home_team":    home,
            "home_team_id": m.get("homeTeam", {}).get("id"),
            "away_team":    away,
            "away_team_id": m.get("awayTeam", {}).get("id"),
            "home_goals":   home_goals,
            "away_goals":   away_goals,
            "league_id":    info["league_id"],
            "league":       info["name"],
            "league_code":  info["league_code"],
            "season":       season_str,
            "round":        f"Matchday {m.get('matchday', '')}",
            "home_crest":   m.get("homeTeam", {}).get("crest", ""),
            "away_crest":   m.get("awayTeam", {}).get("crest", ""),
        }
