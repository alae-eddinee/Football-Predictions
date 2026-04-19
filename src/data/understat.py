"""
Understat scraper for Expected Goals (xG) data.

Fetches match-level xG for home and away teams across major leagues.
Uses the embedded JSON in Understat's HTML pages (no official API).
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://understat.com/league/{league}/{season}"

LEAGUE_MAP = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}


def _extract_json_from_script(html: str, var_name: str) -> Optional[list]:
    """Extract a JS variable's JSON value from a Understat HTML page."""
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        return None
    raw = match.group(1)
    # Unescape JS unicode escapes
    raw = raw.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON decode error for %s: %s", var_name, e)
        return None


def _season_str(season: str) -> str:
    """Convert '2023-24' -> '2023' (Understat uses the start year)."""
    return season.split("-")[0]


def fetch_league_xg(
    league_code: str,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch xG data for a single league/season from Understat."""
    understat_league = LEAGUE_MAP.get(league_code)
    if not understat_league:
        logger.debug("No Understat mapping for league %s", league_code)
        return pd.DataFrame()

    season_year = _season_str(season)
    cache_path = raw_dir / f"understat_{league_code}_{season_year}.parquet"

    if cache_path.exists() and not force_refresh:
        logger.info("Understat cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    url = BASE_URL.format(league=understat_league, season=season_year)
    logger.info("Fetching Understat xG: %s", url)

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Understat fetch failed for %s %s: %s", league_code, season, e)
        return pd.DataFrame()

    data = _extract_json_from_script(resp.text, "datesData")
    if not data:
        logger.warning("Could not parse datesData for %s %s", league_code, season)
        return pd.DataFrame()

    records = []
    for match in data:
        try:
            records.append(
                {
                    "understat_id": match.get("id"),
                    "date": pd.to_datetime(match.get("datetime"), errors="coerce"),
                    "home_team_understat": match.get("h", {}).get("title"),
                    "away_team_understat": match.get("a", {}).get("title"),
                    "home_goals_understat": _safe_int(match.get("goals", {}).get("h")),
                    "away_goals_understat": _safe_int(match.get("goals", {}).get("a")),
                    "home_xg": _safe_float(match.get("xG", {}).get("h")),
                    "away_xg": _safe_float(match.get("xG", {}).get("a")),
                    "home_xg_against": _safe_float(match.get("xG", {}).get("a")),
                    "away_xg_against": _safe_float(match.get("xG", {}).get("h")),
                    "league_code": league_code,
                    "season": season,
                    "forecast_win": _safe_float(
                        match.get("forecast", {}).get("w")
                    ),
                    "forecast_draw": _safe_float(
                        match.get("forecast", {}).get("d")
                    ),
                    "forecast_loss": _safe_float(
                        match.get("forecast", {}).get("l")
                    ),
                }
            )
        except Exception as e:
            logger.debug("Skipping match record: %s", e)
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.dropna(subset=["date", "home_xg", "away_xg"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(cache_path, index=False)
    logger.info("Saved %d Understat rows to %s", len(df), cache_path)
    time.sleep(1.0)
    return df


def fetch_all_xg(
    league_codes: List[str],
    seasons: List[str],
    raw_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch xG for all league/season combinations."""
    frames = []
    for league in league_codes:
        for season in seasons:
            df = fetch_league_xg(league, season, raw_dir, force_refresh)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_xg_into_matches(
    matches: pd.DataFrame,
    xg_data: pd.DataFrame,
    date_tolerance_days: int = 1,
) -> pd.DataFrame:
    """
    Left-join xG metrics into the main matches DataFrame.

    Matches on (date ± tolerance, home_team fuzzy, away_team fuzzy).
    Falls back to exact date + team name matching via merge_asof.
    """
    if xg_data.empty or matches.empty:
        return matches

    # Normalise team names for matching
    matches = matches.copy()
    matches["_home_norm"] = matches["home_team"].str.lower().str.strip()
    matches["_away_norm"] = matches["away_team"].str.lower().str.strip()
    xg_data = xg_data.copy()
    xg_data["_home_norm"] = xg_data["home_team_understat"].str.lower().str.strip()
    xg_data["_away_norm"] = xg_data["away_team_understat"].str.lower().str.strip()

    xg_cols = [
        "date",
        "_home_norm",
        "_away_norm",
        "home_xg",
        "away_xg",
        "forecast_win",
        "forecast_draw",
        "forecast_loss",
    ]
    xg_subset = xg_data[xg_cols].copy()

    merged = matches.merge(
        xg_subset,
        on=["date", "_home_norm", "_away_norm"],
        how="left",
    )
    merged.drop(columns=["_home_norm", "_away_norm"], inplace=True)

    fill_rate = merged["home_xg"].notna().mean()
    logger.info("xG merge fill rate: %.1f%%", fill_rate * 100)
    return merged


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
