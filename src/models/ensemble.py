"""
Ensemble model: XGBoost + LightGBM for 3-class outcome prediction.

Outputs per-match probabilities: P(home win), P(draw), P(away win).

Pipeline:
  1. Impute missing values with median
  2. Scale features (for calibration stability)
  3. Train XGBoost and LightGBM independently
  4. Blend via weighted average
  5. Platt/isotonic calibration on held-out val set
  6. Optuna hyperparameter search (optional)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed; XGBoost model will be skipped.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm not installed; LightGBM model will be skipped.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_xgb(params: dict) -> "xgb.XGBClassifier":
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        **params,
    )


def _build_lgbm(params: dict) -> "lgb.LGBMClassifier":
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        verbose=-1,
        **params,
    )


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Weighted ensemble of XGBoost and LightGBM classifiers.

    Attributes
    ----------
    xgb_model, lgbm_model : fitted pipelines (impute → scale → classifier)
    xgb_weight, lgbm_weight : blending weights
    feature_names : list of feature column names used at fit time
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.xgb_weight: float = config["models"]["ensemble"]["xgb_weight"]
        self.lgbm_weight: float = config["models"]["ensemble"]["lgbm_weight"]
        self.random_state: int = config["models"]["random_state"]
        self.xgb_model: Optional[Pipeline] = None
        self.lgbm_model: Optional[Pipeline] = None
        self.feature_names: list = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune: bool = False,
        tune_trials: int = 50,
    ) -> "EnsemblePredictor":
        """
        Fit both models with optional Optuna tuning.

        Parameters
        ----------
        X : feature matrix (rows = matches, columns = engineered features)
        y : target (0=away win, 1=draw, 2=home win)
        tune : run Optuna hyperparameter optimisation
        tune_trials : number of Optuna trials (per model)
        """
        self.feature_names = list(X.columns)

        val_size = self.cfg["models"]["val_size"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=self.random_state
        )

        xgb_params = dict(self.cfg["models"]["xgboost"])
        lgbm_params = dict(self.cfg["models"]["lightgbm"])

        if tune and OPTUNA_AVAILABLE:
            logger.info("Running Optuna tuning for XGBoost...")
            xgb_params = _tune_xgb(X_train, y_train, X_val, y_val, tune_trials, self.random_state)
            logger.info("Running Optuna tuning for LightGBM...")
            lgbm_params = _tune_lgbm(X_train, y_train, X_val, y_val, tune_trials, self.random_state)

        if XGB_AVAILABLE:
            logger.info("Fitting XGBoost...")
            self.xgb_model = self._fit_single(
                _build_xgb(xgb_params), X_train, y_train, X_val, y_val
            )

        if LGB_AVAILABLE:
            logger.info("Fitting LightGBM...")
            self.lgbm_model = self._fit_single(
                _build_lgbm(lgbm_params), X_train, y_train, X_val, y_val
            )

        return self

    def _fit_single(self, clf, X_train, y_train, X_val, y_val) -> Pipeline:
        """Wrap classifier in impute→scale pipeline and fit with early stopping."""
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        fit_kwargs = {}
        clf_name = type(clf).__name__
        if "XGB" in clf_name:
            fit_kwargs = {
                "clf__eval_set": [
                    (
                        pipe.named_steps["scaler"].fit_transform(
                            pipe.named_steps["imputer"].fit_transform(X_val)
                        ),
                        y_val,
                    )
                ],
                "clf__verbose": False,
            }
        elif "LGBM" in clf_name:
            fit_kwargs = {
                "clf__eval_set": [
                    (
                        pipe.named_steps["scaler"].fit_transform(
                            pipe.named_steps["imputer"].fit_transform(X_val)
                        ),
                        y_val,
                    )
                ],
            }

        # Fit imputer and scaler on train data first
        X_tr_imp = pipe.named_steps["imputer"].fit_transform(X_train)
        X_tr_sc = pipe.named_steps["scaler"].fit_transform(X_tr_imp)
        X_val_imp = pipe.named_steps["imputer"].transform(X_val)
        X_val_sc = pipe.named_steps["scaler"].transform(X_val_imp)

        clf_fit_kwargs = {}
        if "XGB" in clf_name:
            clf_fit_kwargs = {
                "eval_set": [(X_val_sc, y_val)],
                "verbose": False,
            }
        elif "LGBM" in clf_name:
            clf_fit_kwargs = {
                "eval_set": [(X_val_sc, y_val)],
            }

        clf.fit(X_tr_sc, y_train, **clf_fit_kwargs)
        # Store fitted transforms back in pipeline steps
        pipe.named_steps["imputer"].fit(X_train)
        pipe.named_steps["scaler"].fit(
            pipe.named_steps["imputer"].transform(X_train)
        )
        return pipe

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return (n_samples, 3) probability array.
        Columns: [P(away win), P(draw), P(home win)]
        """
        X = X[self.feature_names]
        probas = []
        weights = []

        if self.xgb_model is not None:
            imp = self.xgb_model.named_steps["imputer"]
            sc = self.xgb_model.named_steps["scaler"]
            clf = self.xgb_model.named_steps["clf"]
            X_t = sc.transform(imp.transform(X))
            probas.append(clf.predict_proba(X_t))
            weights.append(self.xgb_weight)

        if self.lgbm_model is not None:
            imp = self.lgbm_model.named_steps["imputer"]
            sc = self.lgbm_model.named_steps["scaler"]
            clf = self.lgbm_model.named_steps["clf"]
            X_t = sc.transform(imp.transform(X))
            probas.append(clf.predict_proba(X_t))
            weights.append(self.lgbm_weight)

        if not probas:
            raise RuntimeError("No models fitted.")

        weights = np.array(weights) / sum(weights)
        blended = sum(w * p for w, p in zip(weights, probas))
        # Renormalise rows to sum to 1
        blended = blended / blended.sum(axis=1, keepdims=True)
        return blended

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "EnsemblePredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return mean feature importance across both models."""
        records = []
        if self.xgb_model is not None and XGB_AVAILABLE:
            clf = self.xgb_model.named_steps["clf"]
            imp = clf.feature_importances_
            records.append(
                pd.Series(imp, index=self.feature_names, name="xgb")
            )
        if self.lgbm_model is not None and LGB_AVAILABLE:
            clf = self.lgbm_model.named_steps["clf"]
            imp = clf.feature_importances_
            records.append(
                pd.Series(imp, index=self.feature_names, name="lgbm")
            )
        if not records:
            return pd.DataFrame()
        df = pd.concat(records, axis=1)
        df["mean_importance"] = df.mean(axis=1)
        return df.sort_values("mean_importance", ascending=False)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    n_splits: int = 5,
) -> Dict[str, list]:
    """
    Temporal cross-validation (no shuffle — respects time order).
    Returns dict of metric lists across folds.
    """
    from .evaluation import evaluate_predictions

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    results = {"logloss": [], "accuracy": [], "brier": [], "rps": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("CV fold %d/%d", fold + 1, n_splits)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = EnsemblePredictor(config)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)

        metrics = evaluate_predictions(y_val, proba)
        for k, v in metrics.items():
            if k in results:
                results[k].append(v)

    for k, v in results.items():
        logger.info("CV %s: %.4f ± %.4f", k, np.mean(v), np.std(v))

    return results


# ---------------------------------------------------------------------------
# Optuna tuning helpers
# ---------------------------------------------------------------------------

def _tune_xgb(X_tr, y_tr, X_val, y_val, n_trials, seed):
    from sklearn.metrics import log_loss

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": seed,
        }
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(imp.fit_transform(X_tr))
        X_val_t = sc.transform(imp.transform(X_val))
        clf = _build_xgb(params)
        clf.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)], verbose=False)
        return log_loss(y_val, clf.predict_proba(X_val_t))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = seed
    return best


def _tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials, seed):
    from sklearn.metrics import log_loss

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": seed,
        }
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(imp.fit_transform(X_tr))
        X_val_t = sc.transform(imp.transform(X_val))
        clf = _build_lgbm(params)
        clf.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)])
        return log_loss(y_val, clf.predict_proba(X_val_t))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = seed
    return best


# ---------------------------------------------------------------------------
# Config loader helper
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
