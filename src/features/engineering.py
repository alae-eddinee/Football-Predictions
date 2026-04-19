"""
Feature engineering pipeline.

Builds the full feature matrix from raw match data.

Features generated per match (from the HOME team's perspective):
  - Rolling form (last N matches): points, goals scored/conceded, shots, xG
  - Home / Away split form
  - Head-to-head record (last M meetings)
  - Rest / fatigue: days since last match for each side
  - Season context: match_number, week of season
  - League-specific encoding
  - xG-based features (when available)
  - Elo ratings (updated iteratively)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORM_WINDOW = 5
H2H_WINDOW = 10
ELO_K = 32
ELO_BASE = 1500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    matches: pd.DataFrame,
    form_window: int = FORM_WINDOW,
    h2h_window: int = H2H_WINDOW,
) -> pd.DataFrame:
    """
    Given the raw/cleaned match DataFrame, return a feature matrix.

    Input DataFrame must have at minimum:
        date, home_team, away_team, home_goals, away_goals, outcome (1/0/-1)

    Returns the same rows with ~60+ feature columns appended.
    Call after ingestion.
    """
    df = matches.copy().sort_values("date").reset_index(drop=True)

    logger.info("Building features for %d matches...", len(df))

    # Build per-team rolling stats first (iterative, order-sensitive)
    df = _add_rolling_form(df, form_window)
    df = _add_home_away_form(df, form_window)
    df = _add_h2h_features(df, h2h_window)
    df = _add_elo_ratings(df)
    df = _add_rest_features(df)
    df = _add_season_context(df)
    df = _add_xg_features(df, form_window)
    df = _add_goal_diff_features(df, form_window)
    df = _add_standings_features(df)
    df = _encode_leagues(df)

    n_features = len(_get_feature_columns(df))
    logger.info("Feature matrix built: %d rows × %d features", len(df), n_features)
    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for model training."""
    feature_cols = _get_feature_columns(df)
    y = _encode_target(df)
    X = df[feature_cols].copy()
    return X, y


def _encode_target(df: pd.DataFrame) -> pd.Series:
    """Encode outcome as 3-class: 0=away win, 1=draw, 2=home win."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return df["outcome"].map(mapping).astype(int)


# ---------------------------------------------------------------------------
# Rolling form helpers
# ---------------------------------------------------------------------------

def _points_from_outcome(outcome: int, is_home: bool) -> int:
    if outcome == 1:
        return 3 if is_home else 0
    if outcome == 0:
        return 1
    return 0 if is_home else 3


def _add_rolling_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each match, compute rolling stats over the last `window` matches
    for both home and away teams (combined home+away appearances).
    """
    # Build a team-indexed event list
    events = _build_team_events(df)

    # Compute rolling aggregates per team
    team_stats = {}
    for team, group in events.groupby("team"):
        group = group.sort_values("date").reset_index(drop=True)
        rolled = _rolling_team_stats(group, window)
        team_stats[team] = rolled.set_index("match_idx")

    # Map back to main df
    cols = [
        "form_points",
        "form_goals_scored",
        "form_goals_conceded",
        "form_goal_diff",
        "form_wins",
        "form_draws",
        "form_losses",
        "form_shots",
        "form_shots_target",
        "form_win_rate",
    ]

    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        for col in cols:
            df[f"{prefix}_{col}"] = df.apply(
                lambda row: _lookup(team_stats, row[team_col], row.name, col),
                axis=1,
            )

    return df


def _build_team_events(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten matches into per-team event rows."""
    home_events = df[
        ["date", "home_team", "away_team", "home_goals", "away_goals", "outcome"]
    ].copy()
    home_events.columns = [
        "date",
        "team",
        "opponent",
        "goals_scored",
        "goals_conceded",
        "match_outcome",
    ]
    home_events["is_home"] = True
    home_events["match_idx"] = df.index

    away_events = df[
        ["date", "away_team", "home_team", "away_goals", "home_goals", "outcome"]
    ].copy()
    away_events.columns = [
        "date",
        "team",
        "opponent",
        "goals_scored",
        "goals_conceded",
        "match_outcome",
    ]
    away_events["is_home"] = False
    # For away team: outcome is inverted
    away_events["match_outcome"] = away_events["match_outcome"].map(
        {1: -1, 0: 0, -1: 1}
    )
    away_events["match_idx"] = df.index

    events = pd.concat([home_events, away_events], ignore_index=True)
    events.sort_values(["team", "date", "match_idx"], inplace=True)

    # Add shot columns if available
    if "home_shots" in df.columns:
        shots_home = df[["home_shots", "home_shots_target"]].copy()
        shots_away = df[["away_shots", "away_shots_target"]].copy()
        shots_home.columns = ["shots", "shots_target"]
        shots_away.columns = ["shots", "shots_target"]
        shots_home.index = df.index
        shots_away.index = df.index

        home_events_s = shots_home
        away_events_s = shots_away

        home_events_s.index = range(len(home_events_s))
        away_events_s.index = range(len(away_events_s))

        home_mask = events[events["is_home"]].index
        away_mask = events[~events["is_home"]].index

        events.loc[home_mask, "shots"] = home_events_s["shots"].values
        events.loc[home_mask, "shots_target"] = home_events_s["shots_target"].values
        events.loc[away_mask, "shots"] = away_events_s["shots"].values
        events.loc[away_mask, "shots_target"] = away_events_s["shots_target"].values
    else:
        events["shots"] = np.nan
        events["shots_target"] = np.nan

    return events


def _rolling_team_stats(group: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute expanding/rolling stats for one team's event history."""
    records = []
    for i, row in group.iterrows():
        # Look back `window` rows BEFORE the current match
        hist = group.iloc[max(0, group.index.get_loc(i) - window): group.index.get_loc(i)]
        if len(hist) == 0:
            records.append(_empty_stats(row["match_idx"]))
            continue

        outcome_col = hist["match_outcome"]
        wins = (outcome_col == 1).sum()
        draws = (outcome_col == 0).sum()
        losses = (outcome_col == -1).sum()
        points = wins * 3 + draws

        records.append(
            {
                "match_idx": row["match_idx"],
                "form_points": points,
                "form_goals_scored": hist["goals_scored"].sum(),
                "form_goals_conceded": hist["goals_conceded"].sum(),
                "form_goal_diff": (
                    hist["goals_scored"].sum() - hist["goals_conceded"].sum()
                ),
                "form_wins": wins,
                "form_draws": draws,
                "form_losses": losses,
                "form_shots": hist["shots"].sum() if "shots" in hist else np.nan,
                "form_shots_target": (
                    hist["shots_target"].sum() if "shots_target" in hist else np.nan
                ),
                "form_win_rate": wins / len(hist) if len(hist) > 0 else np.nan,
            }
        )

    return pd.DataFrame(records)


def _empty_stats(match_idx: int) -> dict:
    return {
        "match_idx": match_idx,
        "form_points": np.nan,
        "form_goals_scored": np.nan,
        "form_goals_conceded": np.nan,
        "form_goal_diff": np.nan,
        "form_wins": np.nan,
        "form_draws": np.nan,
        "form_losses": np.nan,
        "form_shots": np.nan,
        "form_shots_target": np.nan,
        "form_win_rate": np.nan,
    }


def _lookup(team_stats: dict, team: str, idx: int, col: str) -> float:
    if team not in team_stats:
        return np.nan
    ts = team_stats[team]
    if idx in ts.index:
        return ts.loc[idx, col]
    return np.nan


# ---------------------------------------------------------------------------
# Home / Away split form
# ---------------------------------------------------------------------------

def _add_home_away_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling stats computed only from home fixtures / away fixtures."""
    home_records: dict[str, list] = {}
    away_records: dict[str, list] = {}

    home_form = {col: [] for col in ["h_pts", "h_gf", "h_ga", "h_wr"]}
    away_form = {col: [] for col in ["a_pts", "a_gf", "a_ga", "a_wr"]}

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        # Home team home form
        h_hist = home_records.get(ht, [])[-window:]
        away_hist = away_records.get(at, [])[-window:]

        home_form["h_pts"].append(_calc_points(h_hist))
        home_form["h_gf"].append(_calc_goals_for(h_hist))
        home_form["h_ga"].append(_calc_goals_against(h_hist))
        home_form["h_wr"].append(_calc_win_rate(h_hist))
        away_form["a_pts"].append(_calc_points(away_hist))
        away_form["a_gf"].append(_calc_goals_for(away_hist))
        away_form["a_ga"].append(_calc_goals_against(away_hist))
        away_form["a_wr"].append(_calc_win_rate(away_hist))

        # Update history AFTER reading (avoid data leakage)
        home_records.setdefault(ht, []).append(
            {
                "outcome": 1 if row["outcome"] == 1 else (0 if row["outcome"] == 0 else -1),
                "gf": row["home_goals"],
                "ga": row["away_goals"],
            }
        )
        away_records.setdefault(at, []).append(
            {
                "outcome": -1 if row["outcome"] == 1 else (0 if row["outcome"] == 0 else 1),
                "gf": row["away_goals"],
                "ga": row["home_goals"],
            }
        )

    df["home_home_pts"] = home_form["h_pts"]
    df["home_home_gf"] = home_form["h_gf"]
    df["home_home_ga"] = home_form["h_ga"]
    df["home_home_wr"] = home_form["h_wr"]
    df["away_away_pts"] = away_form["a_pts"]
    df["away_away_gf"] = away_form["a_gf"]
    df["away_away_ga"] = away_form["a_ga"]
    df["away_away_wr"] = away_form["a_wr"]
    return df


def _calc_points(hist: list) -> float:
    if not hist:
        return np.nan
    pts = sum(3 if m["outcome"] == 1 else (1 if m["outcome"] == 0 else 0) for m in hist)
    return pts


def _calc_goals_for(hist: list) -> float:
    if not hist:
        return np.nan
    return sum(m["gf"] for m in hist)


def _calc_goals_against(hist: list) -> float:
    if not hist:
        return np.nan
    return sum(m["ga"] for m in hist)


def _calc_win_rate(hist: list) -> float:
    if not hist:
        return np.nan
    return sum(1 for m in hist if m["outcome"] == 1) / len(hist)


# ---------------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------------

def _add_h2h_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute H2H win rates and goal averages for the last `window` meetings."""
    h2h_history: dict[tuple, list] = {}
    h2h_home_wins = []
    h2h_draws = []
    h2h_away_wins = []
    h2h_home_goals_avg = []
    h2h_away_goals_avg = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        key = tuple(sorted([ht, at]))
        hist = h2h_history.get(key, [])[-window:]
        n = len(hist)

        if n == 0:
            h2h_home_wins.append(np.nan)
            h2h_draws.append(np.nan)
            h2h_away_wins.append(np.nan)
            h2h_home_goals_avg.append(np.nan)
            h2h_away_goals_avg.append(np.nan)
        else:
            # Normalise H2H outcomes relative to the current home team
            home_wins = sum(
                1 for m in hist
                if (m["home"] == ht and m["outcome"] == 1)
                or (m["away"] == ht and m["outcome"] == -1)
            )
            draws = sum(1 for m in hist if m["outcome"] == 0)
            away_wins = n - home_wins - draws
            h2h_home_wins.append(home_wins / n)
            h2h_draws.append(draws / n)
            h2h_away_wins.append(away_wins / n)
            h2h_home_goals_avg.append(
                np.mean([
                    m["home_goals"] if m["home"] == ht else m["away_goals"]
                    for m in hist
                ])
            )
            h2h_away_goals_avg.append(
                np.mean([
                    m["away_goals"] if m["away"] == at else m["home_goals"]
                    for m in hist
                ])
            )

        h2h_history.setdefault(key, []).append(
            {
                "home": ht,
                "away": at,
                "outcome": row["outcome"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
            }
        )

    df["h2h_home_win_rate"] = h2h_home_wins
    df["h2h_draw_rate"] = h2h_draws
    df["h2h_away_win_rate"] = h2h_away_wins
    df["h2h_home_goals_avg"] = h2h_home_goals_avg
    df["h2h_away_goals_avg"] = h2h_away_goals_avg
    return df


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------

def _add_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively compute Elo ratings; add pre-match Elo for each side."""
    elo: dict[str, float] = {}

    pre_home_elo = []
    pre_away_elo = []
    elo_diff = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        elo_h = elo.get(ht, ELO_BASE)
        elo_a = elo.get(at, ELO_BASE)

        pre_home_elo.append(elo_h)
        pre_away_elo.append(elo_a)
        elo_diff.append(elo_h - elo_a)

        # Expected scores
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        exp_a = 1 - exp_h

        # Actual scores
        outcome = row["outcome"]
        if outcome == 1:
            act_h, act_a = 1.0, 0.0
        elif outcome == 0:
            act_h, act_a = 0.5, 0.5
        else:
            act_h, act_a = 0.0, 1.0

        elo[ht] = elo_h + ELO_K * (act_h - exp_h)
        elo[at] = elo_a + ELO_K * (act_a - exp_a)

    df["home_elo"] = pre_home_elo
    df["away_elo"] = pre_away_elo
    df["elo_diff"] = elo_diff
    return df


# ---------------------------------------------------------------------------
# Rest / fatigue features
# ---------------------------------------------------------------------------

def _add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode rest/fatigue from days_since_last_home/away columns."""
    for side in ("home", "away"):
        col = f"days_since_last_{side}"
        if col not in df.columns:
            df[col] = np.nan
        df[f"{side}_fatigue"] = (df[col] < 4).astype(float)
        df[f"{side}_rested"] = (df[col] >= 7).astype(float)
        df[f"{side}_rest_days"] = df[col].clip(0, 30)
    return df


# ---------------------------------------------------------------------------
# Season context
# ---------------------------------------------------------------------------

def _add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add match number within the season and calendar features."""
    df = df.copy()
    df["match_number"] = df.groupby(["league_code", "season"]).cumcount() + 1
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # Season stage: early (0-0.33), mid (0.33-0.66), late (0.66-1)
    season_lengths = df.groupby(["league_code", "season"])["match_number"].transform("max")
    df["season_progress"] = df["match_number"] / season_lengths
    return df


# ---------------------------------------------------------------------------
# xG features
# ---------------------------------------------------------------------------

def _add_xg_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling xG averages when Understat data is available."""
    if "home_xg" not in df.columns:
        return df

    xg_history: dict[str, list] = {}

    home_xg_avg = []
    away_xg_avg = []
    home_xga_avg = []
    away_xga_avg = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        ht_xg_hist = xg_history.get(ht, [])[-window:]
        at_xg_hist = xg_history.get(at, [])[-window:]

        home_xg_avg.append(
            np.mean([x["xg"] for x in ht_xg_hist]) if ht_xg_hist else np.nan
        )
        away_xg_avg.append(
            np.mean([x["xg"] for x in at_xg_hist]) if at_xg_hist else np.nan
        )
        home_xga_avg.append(
            np.mean([x["xga"] for x in ht_xg_hist]) if ht_xg_hist else np.nan
        )
        away_xga_avg.append(
            np.mean([x["xga"] for x in at_xg_hist]) if at_xg_hist else np.nan
        )

        # Update after reading (no leakage)
        if pd.notna(row.get("home_xg")):
            xg_history.setdefault(ht, []).append(
                {"xg": row["home_xg"], "xga": row.get("away_xg", np.nan)}
            )
            xg_history.setdefault(at, []).append(
                {"xg": row.get("away_xg", np.nan), "xga": row["home_xg"]}
            )

    df["home_xg_rolling"] = home_xg_avg
    df["away_xg_rolling"] = away_xg_avg
    df["home_xga_rolling"] = home_xga_avg
    df["away_xga_rolling"] = away_xga_avg
    df["xg_diff_rolling"] = (
        df["home_xg_rolling"].fillna(0) - df["away_xg_rolling"].fillna(0)
    )
    return df


# ---------------------------------------------------------------------------
# Goal difference rolling features
# ---------------------------------------------------------------------------

def _add_goal_diff_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling average goals scored/conceded per match."""
    for side, gf_col, ga_col in [
        ("home", "home_goals", "away_goals"),
        ("away", "away_goals", "home_goals"),
    ]:
        form_gf_key = f"{side}_form_goals_scored"
        form_ga_key = f"{side}_form_goals_conceded"
        if form_gf_key in df.columns:
            df[f"{side}_avg_goals_scored"] = df[form_gf_key] / window
            df[f"{side}_avg_goals_conceded"] = df[form_ga_key] / window

    return df


# ---------------------------------------------------------------------------
# Standings features (from API-Football enrichment)
# ---------------------------------------------------------------------------

def _add_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive numeric features from the standings columns added by
    enrich_with_standings() in api_football.py.

    Columns expected (if present):
        home_standing_rank, home_standing_pts, home_form_str,
        home_win_rate_season, home_form_wins_last5,
        away_standing_rank, away_standing_pts, away_form_str,
        away_win_rate_season, away_form_wins_last5
    """
    # Rank differential (positive = home team is ranked higher/better)
    if "home_standing_rank" in df.columns and "away_standing_rank" in df.columns:
        df["rank_diff"] = (
            df["away_standing_rank"].astype(float)
            - df["home_standing_rank"].astype(float)
        )

    # Points differential
    if "home_standing_pts" in df.columns and "away_standing_pts" in df.columns:
        df["pts_diff"] = (
            df["home_standing_pts"].astype(float)
            - df["away_standing_pts"].astype(float)
        )

    # Win rate differential (home season record vs away season record)
    if "home_win_rate_season" in df.columns and "away_win_rate_season" in df.columns:
        df["win_rate_diff_season"] = (
            df["home_win_rate_season"].astype(float)
            - df["away_win_rate_season"].astype(float)
        )

    # Recent form wins differential (last 5 from standings form string)
    if "home_form_wins_last5" in df.columns and "away_form_wins_last5" in df.columns:
        df["form_wins_diff_last5"] = (
            df["home_form_wins_last5"].astype(float)
            - df["away_form_wins_last5"].astype(float)
        )

    return df


# ---------------------------------------------------------------------------
# League encoding
# ---------------------------------------------------------------------------

def _encode_leagues(df: pd.DataFrame) -> pd.DataFrame:
    if "league_code" in df.columns:
        league_dummies = pd.get_dummies(df["league_code"], prefix="league")
        df = pd.concat([df, league_dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Feature column selector
# ---------------------------------------------------------------------------

FEATURE_PREFIXES = (
    # Rolling form (all matches)
    "home_form_",
    "away_form_",
    # Home/away split form
    "home_home_",
    "away_away_",
    # Head-to-head
    "h2h_",
    # Elo
    "home_elo",
    "away_elo",
    "elo_diff",
    # Fatigue / rest
    "home_fatigue",
    "away_fatigue",
    "home_rested",
    "away_rested",
    "home_rest_days",
    "away_rest_days",
    # Season context
    "match_number",
    "day_of_week",
    "month",
    "is_weekend",
    "season_progress",
    # xG (Understat)
    "home_xg_rolling",
    "away_xg_rolling",
    "home_xga_rolling",
    "away_xga_rolling",
    "xg_diff_rolling",
    # Goal averages
    "home_avg_goals_",
    "away_avg_goals_",
    # Standings (API-Football)
    "rank_diff",
    "pts_diff",
    "win_rate_diff_season",
    "form_wins_diff_last5",
    "home_win_rate_season",
    "away_win_rate_season",
    "home_form_wins_last5",
    "away_form_wins_last5",
    # League dummies
    "league_",
)


def _get_feature_columns(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if any(c.startswith(p) or c == p for p in FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
