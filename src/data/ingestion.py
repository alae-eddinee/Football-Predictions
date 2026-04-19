"""
Unified data ingestion pipeline.

Orchestrates:
1. football-data.co.uk  -> raw match results + bookmaker odds
2. Understat            -> xG metrics per match
3. API-Football         -> standings (form, home/away record, rank),
                           fatigue (days since last match)

Saves final merged dataset to data/processed/matches.parquet.

API-Football budget note
------------------------
Free plan = 100 req/day.  We only call the /standings endpoint
(1 req per league/season) + /teams (1 req per league/season).
For 6 leagues × 5 seasons that is 60 requests — within budget.
Per-fixture stats and predictions are opt-in via include_fixture_stats.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from .api_football import (
    APIFootballClient,
    compute_team_fatigue,
    enrich_with_standings,
)
from .football_data import fetch_all
from .understat import fetch_all_xg, merge_xg_into_matches

logger = logging.getLogger(__name__)


def run_ingestion(
    config_path: str = "configs/config.yaml",
    force_refresh: bool = False,
    include_xg: bool = True,
    include_standings: bool = True,
    leagues: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Full ingestion pipeline. Returns merged, enriched DataFrame.

    Parameters
    ----------
    config_path        : path to project config.yaml
    force_refresh      : ignore all caches and re-download
    include_xg         : merge Understat xG data
    include_standings  : merge API-Football standings (form, home/away splits)
    leagues            : subset of league codes (overrides config)
    seasons            : subset of seasons (overrides config)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    external_dir = Path(cfg["data"]["external_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    for d in [raw_dir, external_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "matches.parquet"

    if out_path.exists() and not force_refresh:
        logger.info("Loading cached matches from %s", out_path)
        return pd.read_parquet(out_path)

    league_codes = leagues or [v["code"] for v in cfg["leagues"].values()]
    season_list = seasons or cfg["seasons"]

    # --- Step 1: football-data.co.uk ---
    logger.info("=== Step 1: Fetching football-data.co.uk ===")
    matches = fetch_all(
        config_path=config_path,
        raw_dir=raw_dir,
        force_refresh=force_refresh,
        leagues=league_codes,
        seasons=season_list,
    )

    if matches.empty:
        logger.error("No match data fetched. Aborting.")
        return pd.DataFrame()

    logger.info("Matches loaded: %d", len(matches))

    # --- Step 2: Understat xG ---
    if include_xg:
        logger.info("=== Step 2: Fetching Understat xG ===")
        xg_data = fetch_all_xg(
            league_codes=league_codes,
            seasons=season_list,
            raw_dir=raw_dir,
            force_refresh=force_refresh,
        )
        if not xg_data.empty:
            matches = merge_xg_into_matches(matches, xg_data)
        else:
            logger.warning("No xG data fetched; continuing without it.")

    # --- Step 3: API-Football standings enrichment ---
    if include_standings:
        logger.info("=== Step 3: Fetching API-Football standings ===")
        matches = _enrich_with_api_football(
            matches, league_codes, season_list, external_dir, force_refresh
        )

    # --- Step 4: Fatigue features (no API calls) ---
    logger.info("=== Step 4: Computing fatigue features ===")
    matches = compute_team_fatigue(matches)

    # --- Step 5: Bookmaker implied probabilities ---
    logger.info("=== Step 5: Computing implied probabilities ===")
    matches = add_implied_probabilities(matches)

    # --- Step 6: Final cleanup ---
    matches = _final_cleanup(matches)

    matches.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(matches), out_path)
    return matches


def _enrich_with_api_football(
    matches: pd.DataFrame,
    league_codes: List[str],
    season_list: List[str],
    external_dir: Path,
    force_refresh: bool,
) -> pd.DataFrame:
    """
    Pull standings (form, home/away record, rank) from API-Football and
    left-join them into the match DataFrame.

    Budget: 1 request per (league × season) for /standings.
    """
    client = APIFootballClient()
    status = client.get_status()
    used = status.get("requests", {}).get("current", 0)
    limit = status.get("requests", {}).get("limit_day", 100)
    remaining = limit - used
    budget_needed = len(league_codes) * len(season_list)  # standings calls

    if remaining < budget_needed:
        logger.warning(
            "API-Football budget too low (%d remaining, need %d). "
            "Skipping standings enrichment.",
            remaining, budget_needed,
        )
        return matches

    all_standings = []
    for league_code in league_codes:
        for season in season_list:
            standings = client.get_standings(
                league_code=league_code,
                season=season,
                cache_dir=external_dir,
                force_refresh=force_refresh,
            )
            if not standings.empty:
                all_standings.append(standings)
                logger.debug(
                    "Standings: %s %s — %d teams", league_code, season, len(standings)
                )

    if not all_standings:
        logger.warning("No standings data retrieved.")
        return matches

    standings_df = pd.concat(all_standings, ignore_index=True)
    logger.info(
        "Standings loaded: %d rows across %d league/seasons",
        len(standings_df),
        len(all_standings),
    )

    matches = enrich_with_standings(matches, standings_df)

    # Also cache full standings for feature engineering
    standings_path = external_dir / "standings_all.parquet"
    standings_df.to_parquet(standings_path, index=False)
    logger.info("Standings saved to %s", standings_path)

    return matches


def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute implied probabilities from bookmaker odds.

    Uses the market-average odds (avg_h/d/a) as the primary source,
    falls back to Bet365 if market avg is unavailable.
    For each bookmaker trio, applies the standard overround normalisation:
        p_i = (1/odds_i) / sum(1/odds_j for j in {H,D,A})
    """
    df = df.copy()

    bm_sets = {
        "avg": ("avg_h", "avg_d", "avg_a"),
        "b365": ("b365_h", "b365_d", "b365_a"),
        "pinnacle": ("ps_h", "ps_d", "ps_a"),
        "max": ("max_h", "max_d", "max_a"),
    }

    for bm, (h_col, d_col, a_col) in bm_sets.items():
        if not all(c in df.columns for c in (h_col, d_col, a_col)):
            continue

        raw_h = pd.to_numeric(df[h_col], errors="coerce")
        raw_d = pd.to_numeric(df[d_col], errors="coerce")
        raw_a = pd.to_numeric(df[a_col], errors="coerce")

        # Implied (raw)
        imp_h = 1.0 / raw_h
        imp_d = 1.0 / raw_d
        imp_a = 1.0 / raw_a
        total = imp_h + imp_d + imp_a

        # Overround-normalised
        df[f"{bm}_prob_h"] = imp_h / total
        df[f"{bm}_prob_d"] = imp_d / total
        df[f"{bm}_prob_a"] = imp_a / total
        df[f"{bm}_overround"] = total - 1.0

    # Primary implied prob columns (use avg, fallback to b365)
    for outcome, avg_col, b365_col in [
        ("home", "avg_prob_h", "b365_prob_h"),
        ("draw", "avg_prob_d", "b365_prob_d"),
        ("away", "avg_prob_a", "b365_prob_a"),
    ]:
        if avg_col in df.columns:
            df[f"implied_prob_{outcome}"] = df[avg_col]
        elif b365_col in df.columns:
            df[f"implied_prob_{outcome}"] = df[b365_col]

    return df


def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove matches missing critical columns."""
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    if before != after:
        logger.info("Dropped %d rows missing required fields", before - after)
    df = df.sort_values("date").reset_index(drop=True)
    return df
