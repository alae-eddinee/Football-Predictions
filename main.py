"""
Football Match Outcome Predictor + Betting Market Inefficiency Detector
========================================================================

Full ML pipeline entry point.

Usage examples
--------------
# Full run (fetch data, train, backtest, report):
    python main.py run

# Only fetch & process data:
    python main.py ingest

# Train models (assumes data already fetched):
    python main.py train

# Backtest with pre-trained model:
    python main.py backtest

# Force-refresh data from sources:
    python main.py run --refresh

# Quick smoke test on a single league:
    python main.py run --leagues E0 --seasons 2022-23 2023-24
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")
console = Console()

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.betting.backtesting import (
    Backtester,
    BacktestResult,
    bankroll_curve,
    breakdown_by_edge_bucket,
    breakdown_by_league,
    breakdown_by_outcome,
)
from src.betting.value_detection import (
    ValueBetDetector,
    compute_market_efficiency,
    summarise_value_bets,
)
from src.data.ingestion import run_ingestion
from src.features.engineering import build_features, get_feature_matrix
from src.models.ensemble import EnsemblePredictor, load_config
from src.models.evaluation import (
    calibration_data,
    compare_to_bookmaker,
    evaluate_predictions,
    get_confusion_matrix,
)
from src.visualization.plots import generate_full_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Football Match Outcome Predictor + Betting Market Inefficiency Detector."""
    pass


@cli.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
@click.option("--refresh", is_flag=True, help="Force re-download all data")
@click.option("--leagues", multiple=True, help="League codes, e.g. E0 SP1")
@click.option("--seasons", multiple=True, help="Seasons, e.g. 2022-23 2023-24")
@click.option("--no-xg", is_flag=True, help="Skip Understat xG enrichment")
def ingest(config, refresh, leagues, seasons, no_xg):
    """Step 1: Fetch & process raw data."""
    console.print(Panel("[bold cyan]Step 1 — Data Ingestion[/bold cyan]"))
    matches = run_ingestion(
        config_path=config,
        force_refresh=refresh,
        include_xg=not no_xg,
        leagues=list(leagues) or None,
        seasons=list(seasons) or None,
    )
    console.print(f"[green]✓ {len(matches):,} matches loaded[/green]")
    _print_data_summary(matches)


@cli.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
@click.option("--tune", is_flag=True, help="Run Optuna hyperparameter tuning")
@click.option("--tune-trials", default=50, show_default=True)
@click.option("--model-out", default="models/ensemble.pkl", show_default=True)
def train(config, tune, tune_trials, model_out):
    """Step 2: Train ensemble model on processed data."""
    console.print(Panel("[bold cyan]Step 2 — Model Training[/bold cyan]"))
    cfg = load_config(config)

    matches = _load_processed(cfg)
    if matches.empty:
        console.print("[red]No processed data found. Run `ingest` first.[/red]")
        raise SystemExit(1)

    console.print(f"Building features for {len(matches):,} matches...")
    df_feat = build_features(matches)
    X, y = get_feature_matrix(df_feat)
    console.print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    console.print(f"Class distribution: {dict(y.value_counts().sort_index())}")

    # Temporal split: train on first 80%, test on last 20%
    split = int(len(X) * (1 - cfg["models"]["test_size"]))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = EnsemblePredictor(cfg)
    model.fit(X_train, y_train, tune=tune, tune_trials=tune_trials)

    # Evaluate on test set
    proba_test = model.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, proba_test)
    _print_metrics_table(metrics)

    # Save model
    model_path = Path(model_out)
    model.save(model_path)
    console.print(f"[green]✓ Model saved to {model_path}[/green]")

    # Save feature-engineered data for backtesting step
    out_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    df_feat.to_parquet(out_path, index=False)
    console.print(f"[green]✓ Feature data saved to {out_path}[/green]")

    # Return split index for downstream use
    return split


@cli.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
@click.option("--model-path", default="models/ensemble.pkl", show_default=True)
@click.option("--strategy", default="kelly", type=click.Choice(["kelly", "flat"]), show_default=True)
@click.option("--bankroll", default=1000.0, show_default=True, help="Starting bankroll (£)")
@click.option("--min-edge", default=0.05, show_default=True)
@click.option("--output-dir", default="output", show_default=True)
def backtest(config, model_path, strategy, bankroll, min_edge, output_dir):
    """Step 3: Detect value bets and backtest a simulated strategy."""
    console.print(Panel("[bold cyan]Step 3 — Value Bet Detection & Backtest[/bold cyan]"))
    cfg = load_config(config)
    output_dir = Path(output_dir)

    # Load feature data (or raw + rebuild)
    feat_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    if feat_path.exists():
        df_feat = pd.read_parquet(feat_path)
        console.print(f"Loaded features from {feat_path}")
    else:
        console.print("[yellow]Features not found, rebuilding from processed data...[/yellow]")
        matches = _load_processed(cfg)
        df_feat = build_features(matches)

    # Load model
    model = EnsemblePredictor.load(model_path)
    from src.features.engineering import get_feature_matrix, _get_feature_columns
    X, y = get_feature_matrix(df_feat)

    # Temporal test split (same split as training)
    split = int(len(X) * (1 - cfg["models"]["test_size"]))
    X_test = X.iloc[split:]
    df_test = df_feat.iloc[split:].copy().reset_index(drop=True)
    y_test = y.iloc[split:].reset_index(drop=True)

    console.print(f"Scoring {len(X_test):,} test matches...")
    proba_test = model.predict_proba(X_test)

    # Compare to bookmaker
    bm_proba = _extract_bookmaker_proba(df_test)
    if bm_proba is not None:
        compare_to_bookmaker(y_test, proba_test, bm_proba)

    # Add model probabilities to test df
    df_test_probs = compute_market_efficiency(df_test, proba_test)

    # Value bet detection
    detector = ValueBetDetector(
        min_edge=min_edge,
        kelly_fraction=cfg["betting"]["kelly_fraction"],
        max_bet_fraction=cfg["betting"]["max_bet_fraction"],
        min_odds=cfg["betting"]["min_odds"],
        max_odds=cfg["betting"]["max_odds"],
    )
    vb_df = detector.scan(df_test, proba_test)

    if vb_df.empty:
        console.print("[yellow]No value bets found. Try lowering --min-edge.[/yellow]")
    else:
        console.print(f"[green]✓ {len(vb_df)} value bet opportunities found[/green]")
        _print_value_bets_summary(vb_df)

    # Backtest
    backtester = Backtester(
        initial_bankroll=bankroll,
        strategy=strategy,
        flat_stake_pct=0.02,
    )
    result = backtester.run(vb_df, df_test)
    _print_backtest_summary(result)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    if not vb_df.empty:
        vb_df.to_csv(output_dir / "value_bets.csv", index=False)
    bets_df = result.to_dataframe()
    if not bets_df.empty:
        bets_df.to_csv(output_dir / "backtest_bets.csv", index=False)

    # Generate report
    _generate_report(
        result=result,
        model=model,
        vb_df=vb_df,
        df_test_probs=df_test_probs,
        y_test=y_test,
        proba_test=proba_test,
        output_dir=output_dir,
    )

    console.print(f"\n[bold green]✓ All outputs saved to {output_dir}/[/bold green]")


@cli.command()
@click.option("--config", default="configs/config.yaml", show_default=True)
@click.option("--refresh", is_flag=True)
@click.option("--leagues", multiple=True)
@click.option("--seasons", multiple=True)
@click.option("--tune", is_flag=True)
@click.option("--strategy", default="kelly", type=click.Choice(["kelly", "flat"]))
@click.option("--bankroll", default=1000.0)
@click.option("--min-edge", default=0.05)
@click.option("--output-dir", default="output")
@click.pass_context
def run(ctx, config, refresh, leagues, seasons, tune, strategy, bankroll, min_edge, output_dir):
    """Run the full pipeline: ingest → train → backtest."""
    console.print(
        Panel(
            "[bold white]Football Match Outcome Predictor[/bold white]\n"
            "[dim]+ Betting Market Inefficiency Detector[/dim]",
            style="bold cyan",
        )
    )

    ctx.invoke(ingest, config=config, refresh=refresh, leagues=leagues,
               seasons=seasons, no_xg=False)
    ctx.invoke(train, config=config, tune=tune, tune_trials=50,
               model_out="models/ensemble.pkl")
    ctx.invoke(backtest, config=config, model_path="models/ensemble.pkl",
               strategy=strategy, bankroll=bankroll, min_edge=min_edge,
               output_dir=output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_processed(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data"]["processed_dir"]) / "matches.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _extract_bookmaker_proba(df: pd.DataFrame) -> np.ndarray:
    """Build (n,3) bookmaker probability matrix from implied_prob columns."""
    cols = ["implied_prob_away", "implied_prob_draw", "implied_prob_home"]
    fallback = ["avg_prob_a", "avg_prob_d", "avg_prob_h"]
    for col_set in [cols, fallback]:
        if all(c in df.columns for c in col_set):
            return df[col_set].fillna(1 / 3).values
    return None


def _generate_report(
    result: BacktestResult,
    model: EnsemblePredictor,
    vb_df: pd.DataFrame,
    df_test_probs: pd.DataFrame,
    y_test: pd.Series,
    proba_test: np.ndarray,
    output_dir: Path,
) -> None:
    console.print("\n[cyan]Generating visualisations...[/cyan]")
    try:
        curve = bankroll_curve(result) if result.n_bets > 0 else pd.Series(dtype=float)
        fi = model.feature_importance()
        cal = calibration_data(y_test, proba_test)
        cm = get_confusion_matrix(y_test, proba_test)
        league_bd = breakdown_by_league(result) if result.n_bets > 0 else pd.DataFrame()
        outcome_bd = breakdown_by_outcome(result) if result.n_bets > 0 else pd.DataFrame()
        edge_bd = breakdown_by_edge_bucket(result) if result.n_bets > 0 else pd.DataFrame()

        generate_full_report(
            bankroll_series=curve,
            initial_bankroll=result.initial_bankroll,
            feature_importance=fi,
            calibration_data=cal,
            value_bets_df=vb_df,
            cm_df=cm,
            league_breakdown=league_bd,
            outcome_breakdown=outcome_bd,
            edge_breakdown=edge_bd,
            matches_with_probs=df_test_probs,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.warning("Report generation error: %s", e)


def _print_data_summary(df: pd.DataFrame) -> None:
    table = Table(title="Data Summary", style="cyan")
    table.add_column("League", style="white")
    table.add_column("Matches", justify="right")
    table.add_column("Seasons", justify="right")
    table.add_column("Date Range")

    if "league_code" in df.columns:
        for league, grp in df.groupby("league_code"):
            table.add_row(
                league,
                str(len(grp)),
                str(grp["season"].nunique()) if "season" in grp else "—",
                f"{grp['date'].min().date()} → {grp['date'].max().date()}",
            )
    console.print(table)


def _print_metrics_table(metrics: dict) -> None:
    table = Table(title="Test Set Metrics", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right")
    table.add_row("Log-loss", f"{metrics['logloss']:.4f}")
    table.add_row("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    table.add_row("Brier Score", f"{metrics['brier']:.4f}")
    table.add_row("RPS", f"{metrics['rps']:.4f}")
    table.add_row("RPS Skill Score", f"{metrics['rps_skill']:+.4f}")
    console.print(table)


def _print_value_bets_summary(vb_df: pd.DataFrame) -> None:
    summary = summarise_value_bets(vb_df)
    if summary.empty:
        return
    table = Table(title="Value Bets by League & Outcome", style="green")
    for col in summary.columns:
        table.add_column(str(col), justify="right" if col not in ["league", "outcome"] else "left")
    for _, row in summary.head(15).iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)


def _print_backtest_summary(result: BacktestResult) -> None:
    s = result.summary()
    roi_color = "green" if s["roi_pct"] >= 0 else "red"
    table = Table(title="Backtest Summary", style="bold")
    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right")
    table.add_row("Total Bets", str(s["n_bets"]))
    table.add_row("Wins", str(s["n_wins"]))
    table.add_row("Win Rate", f"{s['win_rate_pct']:.1f}%")
    table.add_row("Total Staked", f"£{s['total_staked']:,.2f}")
    table.add_row("Total Profit", f"£{s['total_profit']:+,.2f}")
    table.add_row("ROI", f"[{roi_color}]{s['roi_pct']:+.2f}%[/{roi_color}]")
    table.add_row("Final Bankroll", f"£{s['final_bankroll']:,.2f}")
    table.add_row("Max Drawdown", f"{s['max_drawdown_pct']:.1f}%")
    table.add_row("Sharpe Ratio", f"{s['sharpe_ratio']:.3f}")
    table.add_row("Avg Odds", f"{s['avg_odds']:.3f}")
    table.add_row("Avg Edge", f"{s['avg_edge']:.4f}")
    console.print(table)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli()
