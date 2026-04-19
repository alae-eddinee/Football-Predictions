"""
Betting market inefficiency detector.

Core logic:
  - Convert bookmaker decimal odds to implied probabilities (with overround removal)
  - Compare model probabilities vs. implied market probabilities
  - Flag "value bets" where model edge > threshold
  - Apply fractional Kelly criterion for position sizing
  - Produce a ranked value bet report per match
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Outcome index mapping
HOME_WIN = 2
DRAW = 1
AWAY_WIN = 0
OUTCOME_LABELS = {HOME_WIN: "H", DRAW: "D", AWAY_WIN: "A"}
OUTCOME_NAMES = {HOME_WIN: "Home Win", DRAW: "Draw", AWAY_WIN: "Away Win"}


@dataclass
class ValueBet:
    match_idx: int
    date: pd.Timestamp
    home_team: str
    away_team: str
    league: str
    outcome: str          # "H", "D", or "A"
    outcome_name: str
    model_prob: float     # model's P(outcome)
    implied_prob: float   # bookmaker's normalised P(outcome)
    best_odds: float      # best available decimal odds
    edge: float           # model_prob - implied_prob
    kelly_fraction: float # recommended bet size (fraction of bankroll)
    expected_value: float # EV = model_prob * (odds - 1) - (1 - model_prob)
    bookmaker: str        # which bookmaker offers best odds


@dataclass
class MatchValueReport:
    match_idx: int
    date: pd.Timestamp
    home_team: str
    away_team: str
    league: str
    model_probs: Dict[str, float]   # {"H": p, "D": p, "A": p}
    implied_probs: Dict[str, float] # normalised market probs
    best_odds: Dict[str, float]     # best available for each outcome
    value_bets: List[ValueBet] = field(default_factory=list)

    @property
    def has_value(self) -> bool:
        return len(self.value_bets) > 0

    @property
    def best_value_bet(self) -> Optional[ValueBet]:
        if not self.value_bets:
            return None
        return max(self.value_bets, key=lambda b: b.edge)


class ValueBetDetector:
    """
    Detects value bets by comparing model output to bookmaker odds.

    Parameters
    ----------
    min_edge          : minimum (model_prob - implied_prob) to flag a bet
    kelly_fraction    : fractional Kelly multiplier (0.25 = quarter-Kelly)
    max_bet_fraction  : hard cap on bet size as fraction of bankroll
    min_odds          : ignore bets with odds below this
    max_odds          : ignore bets with odds above this
    primary_bookmaker : which odds column set to use for EV calculation
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.05,
        min_odds: float = 1.5,
        max_odds: float = 10.0,
        primary_bookmaker: str = "avg",
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.primary_bookmaker = primary_bookmaker

    def scan(
        self,
        matches: pd.DataFrame,
        model_probas: np.ndarray,
    ) -> pd.DataFrame:
        """
        Scan all matches and return a DataFrame of value bet opportunities.

        Parameters
        ----------
        matches      : match DataFrame (must contain odds columns)
        model_probas : (n, 3) array — columns [P(away), P(draw), P(home)]

        Returns
        -------
        DataFrame of ValueBet records (sorted by edge descending)
        """
        records = []
        bookmaker_cols = self._get_bookmaker_cols(matches)

        for i, (_, row) in enumerate(matches.iterrows()):
            if i >= len(model_probas):
                break

            model_prob_row = model_probas[i]
            match_bets = self._scan_match(
                idx=i,
                row=row,
                model_probs=model_prob_row,
                bookmaker_cols=bookmaker_cols,
            )
            records.extend([b.__dict__ for b in match_bets])

        if not records:
            logger.info("No value bets found with edge > %.2f", self.min_edge)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.sort_values("edge", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(
            "Found %d value bets across %d unique matches",
            len(df),
            df["match_idx"].nunique(),
        )
        return df

    def _scan_match(
        self,
        idx: int,
        row: pd.Series,
        model_probs: np.ndarray,
        bookmaker_cols: Dict[str, Tuple[str, str, str]],
    ) -> List[ValueBet]:
        """Find value bets in a single match row."""
        bets = []

        # Best odds for each outcome across all bookmakers
        best_odds_h, best_bm_h = self._best_odds_for(row, bookmaker_cols, "h")
        best_odds_d, best_bm_d = self._best_odds_for(row, bookmaker_cols, "d")
        best_odds_a, best_bm_a = self._best_odds_for(row, bookmaker_cols, "a")

        # Primary bookmaker implied probs (for edge calculation)
        primary = bookmaker_cols.get(
            self.primary_bookmaker, bookmaker_cols.get("avg", None)
        )
        if primary is None:
            return bets

        h_col, d_col, a_col = primary
        odds_h = _safe_odds(row.get(h_col))
        odds_d = _safe_odds(row.get(d_col))
        odds_a = _safe_odds(row.get(a_col))

        if any(o is None for o in [odds_h, odds_d, odds_a]):
            return bets

        # Normalised implied probabilities
        imp_h, imp_d, imp_a = _normalise_implied(odds_h, odds_d, odds_a)

        for outcome_idx, (model_p, imp_p, best_o, bm) in enumerate([
            (model_probs[HOME_WIN], imp_h, best_odds_h, best_bm_h),
            (model_probs[DRAW], imp_d, best_odds_d, best_bm_d),
            (model_probs[AWAY_WIN], imp_a, best_odds_a, best_bm_a),
        ]):
            outcome_code = OUTCOME_LABELS[outcome_idx if outcome_idx < 2 else 2]
            if outcome_idx == 0:
                outcome_code = "H"
            elif outcome_idx == 1:
                outcome_code = "D"
            else:
                outcome_code = "A"

            if best_o is None:
                continue
            if not (self.min_odds <= best_o <= self.max_odds):
                continue

            edge = model_p - imp_p
            if edge < self.min_edge:
                continue

            ev = model_p * (best_o - 1) - (1 - model_p)
            kelly = self._kelly(model_p, best_o)

            bets.append(
                ValueBet(
                    match_idx=idx,
                    date=row.get("date", pd.NaT),
                    home_team=row.get("home_team", ""),
                    away_team=row.get("away_team", ""),
                    league=row.get("league_code", ""),
                    outcome=outcome_code,
                    outcome_name=OUTCOME_NAMES[
                        HOME_WIN if outcome_code == "H"
                        else (DRAW if outcome_code == "D" else AWAY_WIN)
                    ],
                    model_prob=float(model_p),
                    implied_prob=float(imp_p),
                    best_odds=float(best_o),
                    edge=float(edge),
                    kelly_fraction=float(kelly),
                    expected_value=float(ev),
                    bookmaker=bm or "",
                )
            )

        return bets

    def _kelly(self, prob: float, odds: float) -> float:
        """Fractional Kelly criterion: f = kelly_fraction * (p*b - q) / b."""
        b = odds - 1  # net odds
        q = 1 - prob
        full_kelly = (prob * b - q) / b if b > 0 else 0.0
        frac = max(0.0, full_kelly * self.kelly_fraction)
        return min(frac, self.max_bet_fraction)

    def _best_odds_for(
        self,
        row: pd.Series,
        bookmaker_cols: Dict[str, Tuple[str, str, str]],
        side: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Return (best_odds, bookmaker_name) for a given outcome side."""
        best = None
        best_bm = None
        for bm, (h, d, a) in bookmaker_cols.items():
            col = {"h": h, "d": d, "a": a}[side]
            odds = _safe_odds(row.get(col))
            if odds is not None and (best is None or odds > best):
                best = odds
                best_bm = bm
        return best, best_bm

    @staticmethod
    def _get_bookmaker_cols(
        df: pd.DataFrame,
    ) -> Dict[str, Tuple[str, str, str]]:
        """Discover available bookmaker odds columns from the DataFrame."""
        bm_map = {
            "b365": ("b365_h", "b365_d", "b365_a"),
            "bw": ("bw_h", "bw_d", "bw_a"),
            "iw": ("iw_h", "iw_d", "iw_a"),
            "ps": ("ps_h", "ps_d", "ps_a"),
            "wh": ("wh_h", "wh_d", "wh_a"),
            "vc": ("vc_h", "vc_d", "vc_a"),
            "max": ("max_h", "max_d", "max_a"),
            "avg": ("avg_h", "avg_d", "avg_a"),
        }
        return {
            name: cols
            for name, cols in bm_map.items()
            if all(c in df.columns for c in cols)
        }


def compute_market_efficiency(
    matches: pd.DataFrame,
    model_probas: np.ndarray,
    bookmaker: str = "avg",
) -> pd.DataFrame:
    """
    For each match, add model vs. implied probability columns and edge values.

    Returns the matches DataFrame with added columns:
        model_prob_h, model_prob_d, model_prob_a
        implied_prob_h, implied_prob_d, implied_prob_a
        edge_h, edge_d, edge_a
        overround
    """
    df = matches.copy()
    n = min(len(df), len(model_probas))

    h_col = f"{bookmaker}_h"
    d_col = f"{bookmaker}_d"
    a_col = f"{bookmaker}_a"

    if not all(c in df.columns for c in [h_col, d_col, a_col]):
        logger.warning("Bookmaker columns for '%s' not found.", bookmaker)
        return df

    df["model_prob_h"] = np.nan
    df["model_prob_d"] = np.nan
    df["model_prob_a"] = np.nan
    df["implied_prob_h"] = np.nan
    df["implied_prob_d"] = np.nan
    df["implied_prob_a"] = np.nan
    df["edge_h"] = np.nan
    df["edge_d"] = np.nan
    df["edge_a"] = np.nan
    df["overround"] = np.nan

    for i in range(n):
        row = df.iloc[i]
        odds_h = _safe_odds(row.get(h_col))
        odds_d = _safe_odds(row.get(d_col))
        odds_a = _safe_odds(row.get(a_col))

        if any(o is None for o in [odds_h, odds_d, odds_a]):
            continue

        imp_h, imp_d, imp_a = _normalise_implied(odds_h, odds_d, odds_a)
        overround = (1 / odds_h + 1 / odds_d + 1 / odds_a) - 1.0

        mp_h = float(model_probas[i][HOME_WIN])
        mp_d = float(model_probas[i][DRAW])
        mp_a = float(model_probas[i][AWAY_WIN])

        df.iloc[i, df.columns.get_loc("model_prob_h")] = mp_h
        df.iloc[i, df.columns.get_loc("model_prob_d")] = mp_d
        df.iloc[i, df.columns.get_loc("model_prob_a")] = mp_a
        df.iloc[i, df.columns.get_loc("implied_prob_h")] = imp_h
        df.iloc[i, df.columns.get_loc("implied_prob_d")] = imp_d
        df.iloc[i, df.columns.get_loc("implied_prob_a")] = imp_a
        df.iloc[i, df.columns.get_loc("edge_h")] = mp_h - imp_h
        df.iloc[i, df.columns.get_loc("edge_d")] = mp_d - imp_d
        df.iloc[i, df.columns.get_loc("edge_a")] = mp_a - imp_a
        df.iloc[i, df.columns.get_loc("overround")] = overround

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_odds(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f > 1.0 else None
    except (TypeError, ValueError):
        return None


def _normalise_implied(
    odds_h: float, odds_d: float, odds_a: float
) -> Tuple[float, float, float]:
    """Remove overround and return normalised implied probabilities."""
    raw_h = 1.0 / odds_h
    raw_d = 1.0 / odds_d
    raw_a = 1.0 / odds_a
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def summarise_value_bets(value_bets_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate value bet statistics by league and outcome type."""
    if value_bets_df.empty:
        return pd.DataFrame()
    summary = (
        value_bets_df.groupby(["league", "outcome"])
        .agg(
            count=("edge", "count"),
            mean_edge=("edge", "mean"),
            mean_ev=("expected_value", "mean"),
            mean_odds=("best_odds", "mean"),
            mean_kelly=("kelly_fraction", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return summary.sort_values("mean_edge", ascending=False)
