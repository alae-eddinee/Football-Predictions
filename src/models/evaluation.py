"""
Model evaluation metrics for probabilistic 3-class predictions.

Metrics:
  - Log-loss (cross-entropy)
  - Accuracy (argmax prediction)
  - Brier Score (multi-class)
  - Ranked Probability Score (RPS) — the standard in sports forecasting
  - Calibration curves
  - Confusion matrix
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

logger = logging.getLogger(__name__)

# Outcome labels: 0=away win, 1=draw, 2=home win
OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true  : true class labels (0, 1, or 2)
    y_proba : (n, 3) probability matrix
    verbose : print results to logger

    Returns
    -------
    dict with keys: logloss, accuracy, brier, rps, rps_skill
    """
    y_pred = np.argmax(y_proba, axis=1)

    ll = log_loss(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_multi(y_true, y_proba)
    rps_val = mean_rps(y_true, y_proba)
    rps_skill_val = rps_skill_score(y_true, y_proba)

    metrics = {
        "logloss": ll,
        "accuracy": acc,
        "brier": brier,
        "rps": rps_val,
        "rps_skill": rps_skill_val,
    }

    if verbose:
        logger.info("=" * 50)
        logger.info("Model Evaluation Metrics")
        logger.info("=" * 50)
        logger.info("  Log-loss      : %.4f", ll)
        logger.info("  Accuracy      : %.4f (%.1f%%)", acc, acc * 100)
        logger.info("  Brier Score   : %.4f", brier)
        logger.info("  RPS           : %.4f", rps_val)
        logger.info("  RPS Skill     : %.4f", rps_skill_val)
        logger.info("\nClassification Report:\n%s", classification_report(
            y_true, y_pred, target_names=OUTCOME_LABELS
        ))

    return metrics


def brier_score_multi(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error of probability vectors."""
    n_classes = y_proba.shape[1]
    y_onehot = np.zeros_like(y_proba)
    for i, cls in enumerate(y_true):
        y_onehot[i, int(cls)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def rps_single(y_true_class: int, y_proba_row: np.ndarray) -> float:
    """
    Ranked Probability Score for a single prediction.

    RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_forecast[k] - CDF_obs[k])^2
    where K = number of outcome classes.
    """
    k = len(y_proba_row)
    cum_prob = np.cumsum(y_proba_row)
    cum_obs = np.zeros(k)
    cum_obs[y_true_class:] = 1.0
    return float(np.sum((cum_prob - cum_obs) ** 2)) / (k - 1)


def mean_rps(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Mean RPS over all predictions."""
    scores = [
        rps_single(int(yt), yp) for yt, yp in zip(y_true, y_proba)
    ]
    return float(np.mean(scores))


def rps_skill_score(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    RPS Skill Score vs. climatological baseline.
    RPSS = 1 - RPS_model / RPS_climatology
    Climatology = historical class frequencies as constant prediction.
    """
    counts = np.bincount(y_true.astype(int), minlength=3)
    clim = counts / counts.sum()
    clim_proba = np.tile(clim, (len(y_true), 1))
    rps_clim = mean_rps(y_true, clim_proba)
    rps_model = mean_rps(y_true, y_proba)
    if rps_clim == 0:
        return 0.0
    return float(1.0 - rps_model / rps_clim)


def get_confusion_matrix(
    y_true: pd.Series,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Return confusion matrix as a labelled DataFrame."""
    y_pred = np.argmax(y_proba, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=OUTCOME_LABELS, columns=OUTCOME_LABELS)


def calibration_data(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> List[pd.DataFrame]:
    """
    Compute calibration data for each outcome class.

    Returns a list of DataFrames (one per class) with columns:
    mean_predicted_prob, fraction_of_positives, count.
    """
    results = []
    bins = np.linspace(0, 1, n_bins + 1)

    for cls_idx, cls_name in enumerate(OUTCOME_LABELS):
        probs = y_proba[:, cls_idx]
        actuals = (y_true == cls_idx).astype(int)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        records = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            records.append(
                {
                    "class": cls_name,
                    "mean_predicted_prob": probs[mask].mean(),
                    "fraction_of_positives": actuals[mask].mean(),
                    "count": mask.sum(),
                }
            )
        results.append(pd.DataFrame(records))

    return results


def compare_to_bookmaker(
    y_true: pd.Series,
    model_proba: np.ndarray,
    bookmaker_proba: np.ndarray,
) -> Dict[str, float]:
    """Compare model RPS vs. bookmaker RPS."""
    rps_model = mean_rps(y_true, model_proba)
    rps_bm = mean_rps(y_true, bookmaker_proba)
    improvement = rps_bm - rps_model

    logger.info("Model RPS:     %.4f", rps_model)
    logger.info("Bookmaker RPS: %.4f", rps_bm)
    logger.info("Improvement:   %.4f (positive = model better)", improvement)

    return {
        "model_rps": rps_model,
        "bookmaker_rps": rps_bm,
        "rps_improvement": improvement,
    }
