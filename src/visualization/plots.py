"""
Visualization module — all plots for the ML pipeline report.

Produces:
  - Bankroll curve over time
  - ROI breakdown by league, outcome, edge bucket
  - Feature importance (XGBoost + LightGBM)
  - Calibration curves (model vs. bookmaker)
  - Edge distribution histogram
  - Confusion matrix heatmap
  - Prediction probability distributions
  - Model vs. bookmaker RPS comparison
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = {
    "home": "#2ecc71",
    "draw": "#f39c12",
    "away": "#e74c3c",
    "bankroll": "#3498db",
    "negative": "#e74c3c",
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
}

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#2d3142",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#2d3142",
        "grid.alpha": 0.5,
        "font.family": "monospace",
    }
)


def save_or_show(fig: plt.Figure, path: Optional[Path] = None) -> None:
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Saved plot: %s", path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Bankroll curve
# ---------------------------------------------------------------------------

def plot_bankroll_curve(
    bankroll_series: pd.Series,
    initial_bankroll: float,
    title: str = "Bankroll Over Time",
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        bankroll_series.index,
        bankroll_series.values,
        color=PALETTE["bankroll"],
        linewidth=2,
        label="Bankroll",
    )
    ax.axhline(initial_bankroll, color=PALETTE["neutral"], linestyle="--", linewidth=1, label="Initial bankroll")

    # Shade profit / loss regions
    ax.fill_between(
        bankroll_series.index,
        bankroll_series.values,
        initial_bankroll,
        where=bankroll_series.values >= initial_bankroll,
        alpha=0.2,
        color=PALETTE["positive"],
        label="Profit",
    )
    ax.fill_between(
        bankroll_series.index,
        bankroll_series.values,
        initial_bankroll,
        where=bankroll_series.values < initial_bankroll,
        alpha=0.2,
        color=PALETTE["negative"],
        label="Loss",
    )

    final = bankroll_series.iloc[-1]
    roi = (final - initial_bankroll) / initial_bankroll * 100
    ax.set_title(f"{title}  |  Final: £{final:,.0f}  |  ROI: {roi:+.1f}%", fontsize=13, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll (£)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 25,
    save_path: Optional[Path] = None,
) -> None:
    df = importance_df.head(top_n).sort_values("mean_importance")
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))

    bars = ax.barh(df.index, df["mean_importance"], color=PALETTE["bankroll"], alpha=0.85)

    # Add XGB vs LGBM if both present
    if "xgb" in df.columns and "lgbm" in df.columns:
        ax.barh(
            df.index, df["xgb"], alpha=0.4, color=PALETTE["home"], label="XGBoost"
        )
        ax.barh(
            df.index, df["lgbm"], alpha=0.4, color=PALETTE["away"], label="LightGBM", left=0
        )
        ax.legend(facecolor="#1a1d27")

    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, pad=10)
    ax.set_xlabel("Importance (mean)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Calibration curves
# ---------------------------------------------------------------------------

def plot_calibration(
    calibration_data: List[pd.DataFrame],
    save_path: Optional[Path] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = [PALETTE["home"], PALETTE["draw"], PALETTE["away"]]

    for ax, cal_df, color in zip(axes, calibration_data, colors):
        if cal_df.empty:
            continue
        ax.plot(
            cal_df["mean_predicted_prob"],
            cal_df["fraction_of_positives"],
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label="Model",
        )
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], linewidth=1, label="Perfect")
        ax.set_title(cal_df["class"].iloc[0] if "class" in cal_df else "")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(facecolor="#1a1d27", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Curves by Outcome", fontsize=13, y=1.02)
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Edge distribution
# ---------------------------------------------------------------------------

def plot_edge_distribution(
    value_bets_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    if value_bets_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Edge histogram
    ax = axes[0]
    for outcome, color in [("H", PALETTE["home"]), ("D", PALETTE["draw"]), ("A", PALETTE["away"])]:
        subset = value_bets_df[value_bets_df["outcome"] == outcome]["edge"]
        if not subset.empty:
            ax.hist(subset, bins=30, alpha=0.6, color=color, label={"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome])
    ax.set_title("Value Bet Edge Distribution", fontsize=12)
    ax.set_xlabel("Edge (model prob − implied prob)")
    ax.set_ylabel("Count")
    ax.legend(facecolor="#1a1d27")
    ax.grid(True, alpha=0.3)

    # Odds vs edge scatter
    ax = axes[1]
    color_map = {"H": PALETTE["home"], "D": PALETTE["draw"], "A": PALETTE["away"]}
    for outcome in ["H", "D", "A"]:
        subset = value_bets_df[value_bets_df["outcome"] == outcome]
        if not subset.empty:
            ax.scatter(
                subset["best_odds"],
                subset["edge"],
                alpha=0.5,
                s=20,
                c=color_map[outcome],
                label={"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome],
            )
    ax.axhline(0, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.set_title("Odds vs. Edge", fontsize=12)
    ax.set_xlabel("Best Available Odds")
    ax.set_ylabel("Edge")
    ax.legend(facecolor="#1a1d27")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        linecolor="#2d3142",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Confusion Matrix", fontsize=13, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 6. ROI breakdown charts
# ---------------------------------------------------------------------------

def plot_roi_by_league(
    breakdown_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    _plot_roi_bar(breakdown_df, index_col="league", title="ROI by League", save_path=save_path)


def plot_roi_by_outcome(
    breakdown_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    _plot_roi_bar(breakdown_df, index_col="outcome_name", title="ROI by Outcome Type", save_path=save_path)


def plot_roi_by_edge_bucket(
    breakdown_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    _plot_roi_bar(breakdown_df.reset_index(), index_col="edge_bucket", title="ROI by Edge Bucket", save_path=save_path)


def _plot_roi_bar(
    df: pd.DataFrame,
    index_col: str,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    if df.empty or "roi" not in df.columns:
        return
    df = df.set_index(index_col) if index_col in df.columns else df
    roi_vals = df["roi"] * 100

    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in roi_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(roi_vals.index.astype(str), roi_vals.values, color=colors, edgecolor="#2d3142")
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)

    for bar, val in zip(bars, roi_vals.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.5 if val >= 0 else -1.5),
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    if "bets" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            df.index.astype(str),
            df["bets"].values,
            "o--",
            color=PALETTE["bankroll"],
            linewidth=1.5,
            markersize=5,
            label="# Bets",
        )
        ax2.set_ylabel("# Bets", color=PALETTE["bankroll"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["bankroll"])
        ax2.legend(facecolor="#1a1d27", loc="upper right")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("ROI (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Model vs. bookmaker probability comparison
# ---------------------------------------------------------------------------

def plot_model_vs_bookmaker_probs(
    matches: pd.DataFrame,
    outcome: str = "h",
    save_path: Optional[Path] = None,
) -> None:
    model_col = f"model_prob_{outcome}"
    implied_col = f"implied_prob_{outcome}"
    if model_col not in matches.columns or implied_col not in matches.columns:
        logger.warning("Columns %s or %s not found.", model_col, implied_col)
        return

    df = matches[[model_col, implied_col]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[implied_col], df[model_col], alpha=0.3, s=10, c=PALETTE["bankroll"])
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], linewidth=1, label="Parity")
    ax.set_title(f"Model vs. Implied Probability — {outcome.upper()}", fontsize=12)
    ax.set_xlabel("Bookmaker implied prob")
    ax.set_ylabel("Model prob")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(facecolor="#1a1d27")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Full report generator
# ---------------------------------------------------------------------------

def generate_full_report(
    bankroll_series: pd.Series,
    initial_bankroll: float,
    feature_importance: pd.DataFrame,
    calibration_data: List[pd.DataFrame],
    value_bets_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    league_breakdown: pd.DataFrame,
    outcome_breakdown: pd.DataFrame,
    edge_breakdown: pd.DataFrame,
    matches_with_probs: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate and save all plots to output_dir/plots/."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating full report in %s ...", plots_dir)

    plot_bankroll_curve(bankroll_series, initial_bankroll, save_path=plots_dir / "bankroll_curve.png")
    plot_feature_importance(feature_importance, save_path=plots_dir / "feature_importance.png")
    plot_calibration(calibration_data, save_path=plots_dir / "calibration.png")
    plot_edge_distribution(value_bets_df, save_path=plots_dir / "edge_distribution.png")
    plot_confusion_matrix(cm_df, save_path=plots_dir / "confusion_matrix.png")
    plot_roi_by_league(league_breakdown, save_path=plots_dir / "roi_by_league.png")
    plot_roi_by_outcome(outcome_breakdown, save_path=plots_dir / "roi_by_outcome.png")
    plot_roi_by_edge_bucket(edge_breakdown, save_path=plots_dir / "roi_by_edge.png")
    plot_model_vs_bookmaker_probs(matches_with_probs, "h", save_path=plots_dir / "model_vs_bm_home.png")
    plot_model_vs_bookmaker_probs(matches_with_probs, "d", save_path=plots_dir / "model_vs_bm_draw.png")
    plot_model_vs_bookmaker_probs(matches_with_probs, "a", save_path=plots_dir / "model_vs_bm_away.png")

    logger.info("Report complete. %d plots saved to %s", 11, plots_dir)
