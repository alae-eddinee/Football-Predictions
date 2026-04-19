"""
football-data.co.uk ingestion module.

Downloads historical match CSVs (results + bookmaker odds) for multiple
leagues and seasons.  Normalises column names and saves to data/raw/.
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Canonical column mapping from football-data.co.uk raw headers
COLUMN_MAP = {
    "Div": "league",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    # Bookmaker odds - Home / Draw / Away
    "B365H": "b365_h",
    "B365D": "b365_d",
    "B365A": "b365_a",
    "BWH": "bw_h",
    "BWD": "bw_d",
    "BWA": "bw_a",
    "IWH": "iw_h",
    "IWD": "iw_d",
    "IWA": "iw_a",
    "PSH": "ps_h",
    "PSD": "ps_d",
    "PSA": "ps_a",
    "WHH": "wh_h",
    "WHD": "wh_d",
    "WHA": "wh_a",
    "VCH": "vc_h",
    "VCD": "vc_d",
    "VCA": "vc_a",
    "MaxH": "max_h",
    "MaxD": "max_d",
    "MaxA": "max_a",
    "AvgH": "avg_h",
    "AvgD": "avg_d",
    "AvgA": "avg_a",
}


def _season_to_code(season: str) -> str:
    """Convert '2023-24' -> '2324'."""
    parts = season.split("-")
    return parts[0][-2:] + parts[1][-2:]


def fetch_season(
    league_code: str,
    season: str,
    raw_dir: Path,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """Download one league/season CSV and return a cleaned DataFrame."""
    season_code = _season_to_code(season)
    url = BASE_URL.format(season=season_code, league=league_code)
    out_path = raw_dir / f"{league_code}_{season_code}.csv"

    if out_path.exists() and not force_refresh:
        logger.info("Cache hit: %s", out_path)
        return _load_and_clean(out_path, league_code, season)

    logger.info("Fetching %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    out_path.write_bytes(resp.content)
    time.sleep(0.5)  # polite rate limiting
    return _load_and_clean(out_path, league_code, season)


def _load_and_clean(path: Path, league_code: str, season: str) -> pd.DataFrame:
    """Load raw CSV, rename columns, parse dates, drop empty rows."""
    try:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        logger.error("Could not read %s: %s", path, e)
        return pd.DataFrame()

    # Drop completely empty rows
    df.dropna(how="all", inplace=True)

    # Rename only columns we know about
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df.dropna(subset=["date"], inplace=True)

    # Add metadata
    df["league_code"] = league_code
    df["season"] = season

    # Encode result: H=1 (home win), D=0 (draw), A=-1 (away win)
    if "result" in df.columns:
        df["outcome"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    # Cast goal columns to numeric
    for col in ["home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.debug("Loaded %d matches from %s", len(df), path.name)
    return df


def fetch_all(
    config_path: str = "configs/config.yaml",
    raw_dir: Optional[Path] = None,
    force_refresh: bool = False,
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Download all configured leagues & seasons and return a single DataFrame.

    Parameters
    ----------
    config_path : path to project config.yaml
    raw_dir     : override raw data directory
    force_refresh : re-download even if cached
    leagues     : override list of league codes (e.g. ['E0', 'SP1'])
    seasons     : override list of seasons (e.g. ['2022-23', '2023-24'])
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = raw_dir or Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    league_codes = leagues or [v["code"] for v in cfg["leagues"].values()]
    season_list = seasons or cfg["seasons"]

    frames: List[pd.DataFrame] = []
    for league_code in league_codes:
        for season in season_list:
            df = fetch_season(league_code, season, raw_dir, force_refresh)
            if df is not None and not df.empty:
                frames.append(df)

    if not frames:
        logger.warning("No data fetched.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("Total matches loaded: %d", len(combined))
    return combined


def get_bookmaker_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dict mapping bookmaker name to [home_col, draw_col, away_col]."""
    bm_map = {
        "bet365": ["b365_h", "b365_d", "b365_a"],
        "betwin": ["bw_h", "bw_d", "bw_a"],
        "interwetten": ["iw_h", "iw_d", "iw_a"],
        "pinnacle": ["ps_h", "ps_d", "ps_a"],
        "william_hill": ["wh_h", "wh_d", "wh_a"],
        "vcbet": ["vc_h", "vc_d", "vc_a"],
        "market_max": ["max_h", "max_d", "max_a"],
        "market_avg": ["avg_h", "avg_d", "avg_a"],
    }
    return {
        name: cols
        for name, cols in bm_map.items()
        if all(c in df.columns for c in cols)
    }
