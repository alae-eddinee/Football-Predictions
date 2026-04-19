"""
Backtesting engine for simulated betting strategy.

Simulates flat-stake and Kelly-criterion betting strategies on value bets
detected by the ValueBetDetector.  Produces:
  - Match-by-match P&L log
  - Running bankroll curve
  - ROI, Sharpe, max drawdown, win rate
  - Profit by league, outcome type, edge bucket, odds range
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    date: pd.Timestamp
    home_team: str
    away_team: str
    league: str
    outcome: str          # "H", "D", "A"
    model_prob: float
    implied_prob: float
    best_odds: float
    edge: float
    kelly_fraction: float
    stake: float
    bankroll_before: float
    result: str           # "WIN" or "LOSS"
    profit: float
    bankroll_after: float
    bookmaker: str


@dataclass
class BacktestResult:
    bets: List[BetRecord] = field(default_factory=list)
    initial_bankroll: float = 1000.0
    final_bankroll: float = 0.0
    total_staked: float = 0.0
    total_profit: float = 0.0
    n_bets: int = 0
    n_wins: int = 0
    roi: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_odds: float = 0.0
    avg_edge: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([b.__dict__ for b in self.bets])

    def summary(self) -> Dict:
        return {
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": round(self.final_bankroll, 2),
            "total_staked": round(self.total_staked, 2),
            "total_profit": round(self.total_profit, 2),
            "n_bets": self.n_bets,
            "n_wins": self.n_wins,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "roi_pct": round(self.roi * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "avg_odds": round(self.avg_odds, 3),
            "avg_edge": round(self.avg_edge, 4),
        }


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Simulate a betting strategy over historical value bets.

    Strategies:
      - "flat"  : fixed stake per bet (as fraction of initial bankroll)
      - "kelly" : variable stake = bankroll * kelly_fraction

    Parameters
    ----------
    initial_bankroll : starting capital
    strategy         : "flat" or "kelly"
    flat_stake_pct   : fraction of initial bankroll for flat betting
    max_bets_per_day : max concurrent bets placed on the same day
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        strategy: str = "kelly",
        flat_stake_pct: float = 0.02,
        max_bets_per_day: int = 5,
    ):
        self.initial_bankroll = initial_bankroll
        self.strategy = strategy
        self.flat_stake_pct = flat_stake_pct
        self.max_bets_per_day = max_bets_per_day

    def run(
        self,
        value_bets_df: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        value_bets_df : output from ValueBetDetector.scan()
        matches       : original match DataFrame with actual results
        """
        if value_bets_df.empty:
            logger.warning("No value bets to backtest.")
            return BacktestResult(initial_bankroll=self.initial_bankroll)

        vb = value_bets_df.copy()
        vb["date"] = pd.to_datetime(vb["date"], errors="coerce")
        vb.sort_values("date", inplace=True)

        # Merge actual result from matches
        match_results = self._extract_results(matches)
        vb = vb.merge(
            match_results[["match_idx", "actual_result"]],
            on="match_idx",
            how="left",
        )

        bankroll = self.initial_bankroll
        records: List[BetRecord] = []

        for date, day_bets in vb.groupby("date"):
            # Limit bets per day, prioritised by edge
            day_bets = day_bets.nlargest(self.max_bets_per_day, "edge")

            for _, bet_row in day_bets.iterrows():
                if bankroll <= 0:
                    break

                stake = self._compute_stake(bankroll, bet_row["kelly_fraction"])
                stake = min(stake, bankroll)  # can't bet more than we have

                won = (bet_row.get("actual_result") == bet_row["outcome"])
                if won:
                    profit = stake * (bet_row["best_odds"] - 1)
                    result_str = "WIN"
                else:
                    profit = -stake
                    result_str = "LOSS"

                new_bankroll = bankroll + profit

                records.append(
                    BetRecord(
                        date=date,
                        home_team=bet_row["home_team"],
                        away_team=bet_row["away_team"],
                        league=bet_row["league"],
                        outcome=bet_row["outcome"],
                        model_prob=float(bet_row["model_prob"]),
                        implied_prob=float(bet_row["implied_prob"]),
                        best_odds=float(bet_row["best_odds"]),
                        edge=float(bet_row["edge"]),
                        kelly_fraction=float(bet_row["kelly_fraction"]),
                        stake=float(stake),
                        bankroll_before=float(bankroll),
                        result=result_str,
                        profit=float(profit),
                        bankroll_after=float(new_bankroll),
                        bookmaker=str(bet_row.get("bookmaker", "")),
                    )
                )

                bankroll = new_bankroll

        return self._compute_stats(records)

    def _compute_stake(self, bankroll: float, kelly: float) -> float:
        if self.strategy == "kelly":
            return bankroll * kelly
        return self.initial_bankroll * self.flat_stake_pct

    @staticmethod
    def _extract_results(matches: pd.DataFrame) -> pd.DataFrame:
        """Map match index to actual FTR result ('H', 'D', 'A')."""
        results = matches[["result"]].copy() if "result" in matches.columns else pd.DataFrame()
        if results.empty and "outcome" in matches.columns:
            results = pd.DataFrame(
                {
                    "actual_result": matches["outcome"].map(
                        {1: "H", 0: "D", -1: "A"}
                    )
                }
            )
        elif not results.empty:
            results.rename(columns={"result": "actual_result"}, inplace=True)
        else:
            results = pd.DataFrame({"actual_result": [""] * len(matches)})
        results["match_idx"] = range(len(results))
        return results

    def _compute_stats(self, records: List[BetRecord]) -> BacktestResult:
        if not records:
            return BacktestResult(initial_bankroll=self.initial_bankroll)

        bets_df = pd.DataFrame([r.__dict__ for r in records])

        n_bets = len(records)
        n_wins = sum(1 for r in records if r.result == "WIN")
        total_staked = bets_df["stake"].sum()
        total_profit = bets_df["profit"].sum()
        final_bankroll = records[-1].bankroll_after
        roi = total_profit / total_staked if total_staked > 0 else 0.0
        win_rate = n_wins / n_bets if n_bets > 0 else 0.0

        # Max drawdown
        bankroll_curve = np.array([self.initial_bankroll] + [r.bankroll_after for r in records])
        peak = np.maximum.accumulate(bankroll_curve)
        drawdown = (peak - bankroll_curve) / peak
        max_dd = float(drawdown.max())

        # Sharpe (annualised, using daily returns proxy)
        bets_df["date"] = pd.to_datetime(bets_df["date"])
        daily_pnl = bets_df.groupby("date")["profit"].sum()
        if len(daily_pnl) > 1:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        result = BacktestResult(
            bets=records,
            initial_bankroll=self.initial_bankroll,
            final_bankroll=final_bankroll,
            total_staked=float(total_staked),
            total_profit=float(total_profit),
            n_bets=n_bets,
            n_wins=n_wins,
            roi=roi,
            win_rate=win_rate,
            max_drawdown=max_dd,
            sharpe_ratio=float(sharpe),
            avg_odds=float(bets_df["best_odds"].mean()),
            avg_edge=float(bets_df["edge"].mean()),
        )

        self._log_summary(result)
        return result

    @staticmethod
    def _log_summary(result: BacktestResult) -> None:
        s = result.summary()
        logger.info("=" * 55)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 55)
        logger.info("  Bets placed    : %d  (wins: %d)", s["n_bets"], s["n_wins"])
        logger.info("  Win rate       : %.1f%%", s["win_rate_pct"])
        logger.info("  Total staked   : £%.2f", s["total_staked"])
        logger.info("  Total profit   : £%.2f", s["total_profit"])
        logger.info("  ROI            : %.2f%%", s["roi_pct"])
        logger.info("  Final bankroll : £%.2f  (started £%.2f)",
                    s["final_bankroll"], s["initial_bankroll"])
        logger.info("  Max drawdown   : %.1f%%", s["max_drawdown_pct"])
        logger.info("  Sharpe ratio   : %.3f", s["sharpe_ratio"])
        logger.info("  Avg odds       : %.3f", s["avg_odds"])
        logger.info("  Avg edge       : %.4f", s["avg_edge"])
        logger.info("=" * 55)


# ---------------------------------------------------------------------------
# Breakdown analytics
# ---------------------------------------------------------------------------

def breakdown_by_league(result: BacktestResult) -> pd.DataFrame:
    """P&L breakdown by league."""
    df = result.to_dataframe()
    return (
        df.groupby("league")
        .agg(
            bets=("profit", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            profit=("profit", "sum"),
            staked=("stake", "sum"),
        )
        .assign(
            win_rate=lambda x: x["wins"] / x["bets"],
            roi=lambda x: x["profit"] / x["staked"],
        )
        .round(4)
        .sort_values("roi", ascending=False)
    )


def breakdown_by_outcome(result: BacktestResult) -> pd.DataFrame:
    """P&L breakdown by outcome type (H/D/A)."""
    df = result.to_dataframe()
    label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    df["outcome_name"] = df["outcome"].map(label_map)
    return (
        df.groupby("outcome_name")
        .agg(
            bets=("profit", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            profit=("profit", "sum"),
            staked=("stake", "sum"),
            avg_odds=("best_odds", "mean"),
            avg_edge=("edge", "mean"),
        )
        .assign(
            win_rate=lambda x: x["wins"] / x["bets"],
            roi=lambda x: x["profit"] / x["staked"],
        )
        .round(4)
        .sort_values("roi", ascending=False)
    )


def breakdown_by_edge_bucket(result: BacktestResult) -> pd.DataFrame:
    """P&L breakdown by edge bucket (0-5%, 5-10%, 10-15%, 15%+)."""
    df = result.to_dataframe()
    bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    df["edge_bucket"] = pd.cut(df["edge"], bins=bins, labels=labels)
    return (
        df.groupby("edge_bucket", observed=True)
        .agg(
            bets=("profit", "count"),
            wins=("result", lambda x: (x == "WIN").sum()),
            profit=("profit", "sum"),
            staked=("stake", "sum"),
            avg_odds=("best_odds", "mean"),
        )
        .assign(
            win_rate=lambda x: x["wins"] / x["bets"],
            roi=lambda x: x["profit"] / x["staked"],
        )
        .round(4)
    )


def bankroll_curve(result: BacktestResult) -> pd.Series:
    """Return bankroll over time as a Series indexed by date."""
    df = result.to_dataframe()
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    curve = df.set_index("date")["bankroll_after"]
    # Prepend initial bankroll
    start = pd.Series(
        [result.initial_bankroll],
        index=[curve.index[0] - pd.Timedelta(days=1)],
    )
    return pd.concat([start, curve]).sort_index()
