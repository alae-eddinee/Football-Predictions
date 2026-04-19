"""
Live prediction engine for upcoming matches.

Builds a full feature vector for any (home_team, away_team, date, league)
tuple using only historical data that would be known BEFORE the match.

Steps:
  1. Load processed historical matches + trained model
  2. Replay Elo ratings up to the match date
  3. Compute rolling form (last N) for each team up to the match date
  4. Compute H2H record
  5. Compute rest days (days since each team's last match)
  6. Merge standings features (API-Football)
  7. Run EnsemblePredictor.predict_proba → {home, draw, away} probabilities
"""

import difflib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
MODEL_PATH = ROOT / "models" / "ensemble.pkl"
PROCESSED_PATH = ROOT / "data" / "processed" / "matches.parquet"
EXTERNAL_DIR = ROOT / "data" / "external"

ELO_K = 32
ELO_BASE = 1500
FORM_WINDOW = 5
H2H_WINDOW = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    fixture_id: Optional[int]
    home_team: str
    away_team: str
    date: str
    league: str
    league_code: str
    kickoff: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str          # "Home Win" / "Draw" / "Away Win"
    confidence: float               # max probability
    value_bets: List[Dict] = field(default_factory=list)
    model_available: bool = True
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "fixture_id":       self.fixture_id,
            "home_team":        self.home_team,
            "away_team":        self.away_team,
            "date":             self.date,
            "league":           self.league,
            "league_code":      self.league_code,
            "kickoff":          self.kickoff,
            "prob_home":        round(self.prob_home, 4),
            "prob_draw":        round(self.prob_draw, 4),
            "prob_away":        round(self.prob_away, 4),
            "predicted_outcome": self.predicted_outcome,
            "confidence":       round(self.confidence, 4),
            "value_bets":       self.value_bets,
            "model_available":  self.model_available,
            "note":             self.note,
        }


# ---------------------------------------------------------------------------
# LivePredictor
# ---------------------------------------------------------------------------

class LivePredictor:
    """
    Singleton-style predictor that pre-computes team states from history
    once on first load, then answers per-match prediction requests instantly.
    """

    _instance: Optional["LivePredictor"] = None

    def __init__(self):
        self._loaded = False
        self.history: pd.DataFrame = pd.DataFrame()
        self.elo_state: Dict[str, float] = {}
        self.form_state: Dict[str, List[Dict]] = {}
        self.h2h_state: Dict[Tuple, List[Dict]] = {}
        self.last_match_date: Dict[str, pd.Timestamp] = {}
        self.standings: pd.DataFrame = pd.DataFrame()
        self.model = None
        self.feature_names: List[str] = []
        self.cfg: Dict = {}
        self.team_name_index: Dict[str, str] = {}   # api_name → training_name

    @classmethod
    def get(cls) -> "LivePredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, force: bool = False) -> "LivePredictor":
        if self._loaded and not force:
            return self

        with open(CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

        # Load model
        if MODEL_PATH.exists():
            try:
                from src.models.ensemble import EnsemblePredictor
                self.model = EnsemblePredictor.load(MODEL_PATH)
                self.feature_names = self.model.feature_names
                logger.info("Model loaded: %d features", len(self.feature_names))
            except Exception as e:
                logger.warning("Could not load model: %s", e)
                self.model = None
        else:
            logger.warning("No trained model found at %s", MODEL_PATH)

        # Load historical matches
        if PROCESSED_PATH.exists():
            self.history = pd.read_parquet(PROCESSED_PATH)
            self.history["date"] = pd.to_datetime(self.history["date"])
            self.history.sort_values("date", inplace=True)
            self.history.reset_index(drop=True, inplace=True)
            logger.info("History loaded: %d matches", len(self.history))
            self._build_team_states()
            self._build_team_name_index()
        else:
            logger.warning("No processed history at %s — predictions will lack form features", PROCESSED_PATH)

        # Load standings
        standings_path = EXTERNAL_DIR / "standings_all.parquet"
        if standings_path.exists():
            self.standings = pd.read_parquet(standings_path)
            logger.info("Standings loaded: %d rows", len(self.standings))

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # State building from history
    # ------------------------------------------------------------------

    def _build_team_states(self) -> None:
        """Replay all history to get current Elo, form, H2H, last match date."""
        logger.info("Building team states from history…")

        for _, row in self.history.iterrows():
            ht = row["home_team"]
            at = row["away_team"]
            date_ = row["date"]
            outcome = row.get("outcome", None)
            hg = row.get("home_goals", 0) or 0
            ag = row.get("away_goals", 0) or 0

            if pd.isna(outcome):
                continue

            # --- Elo ---
            elo_h = self.elo_state.get(ht, ELO_BASE)
            elo_a = self.elo_state.get(at, ELO_BASE)
            exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
            act_h = 1.0 if outcome == 1 else (0.5 if outcome == 0 else 0.0)
            self.elo_state[ht] = elo_h + ELO_K * (act_h - exp_h)
            self.elo_state[at] = elo_a + ELO_K * ((1 - act_h) - (1 - exp_h))

            # --- Rolling form ---
            self.form_state.setdefault(ht, []).append({
                "outcome": 1 if outcome == 1 else (0 if outcome == 0 else -1),
                "gf": hg, "ga": ag, "is_home": True, "date": date_,
            })
            self.form_state.setdefault(at, []).append({
                "outcome": -1 if outcome == 1 else (0 if outcome == 0 else 1),
                "gf": ag, "ga": hg, "is_home": False, "date": date_,
            })

            # --- H2H ---
            key = tuple(sorted([ht, at]))
            self.h2h_state.setdefault(key, []).append({
                "home": ht, "away": at, "outcome": outcome,
                "home_goals": hg, "away_goals": ag, "date": date_,
            })

            # --- Last match date ---
            self.last_match_date[ht] = date_
            self.last_match_date[at] = date_

        logger.info(
            "Team states built: %d teams, Elo range [%.0f–%.0f]",
            len(self.elo_state),
            min(self.elo_state.values(), default=ELO_BASE),
            max(self.elo_state.values(), default=ELO_BASE),
        )

    def _build_team_name_index(self) -> None:
        """Build index of lowercase → canonical team names for fuzzy matching."""
        teams = set(self.history["home_team"].tolist() + self.history["away_team"].tolist())
        self.team_name_index = {t.lower().strip(): t for t in teams}

    # ------------------------------------------------------------------
    # Team name resolution
    # ------------------------------------------------------------------

    def resolve_team_name(self, api_name: str) -> str:
        """
        Map an API-Football team name to the canonical name used in training.

        Tries exact match → normalised exact → fuzzy match with cutoff 0.7.
        Falls back to the original name if no match found.
        """
        if not self.team_name_index:
            return api_name

        norm = api_name.lower().strip()

        # Exact
        if norm in self.team_name_index:
            return self.team_name_index[norm]

        # Fuzzy
        candidates = list(self.team_name_index.keys())
        matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.65)
        if matches:
            resolved = self.team_name_index[matches[0]]
            if resolved.lower() != api_name.lower():
                logger.debug("Name resolved: '%s' → '%s'", api_name, resolved)
            return resolved

        logger.debug("No name match for '%s', using as-is", api_name)
        return api_name

    # ------------------------------------------------------------------
    # Feature vector construction
    # ------------------------------------------------------------------

    def build_feature_row(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        league_code: str,
        season: str,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for an upcoming match."""
        feats: Dict[str, Any] = {}

        # --- Elo ---
        elo_h = self.elo_state.get(home_team, ELO_BASE)
        elo_a = self.elo_state.get(away_team, ELO_BASE)
        feats["home_elo"] = elo_h
        feats["away_elo"] = elo_a
        feats["elo_diff"] = elo_h - elo_a

        # --- Rolling form (all matches) ---
        form_h = self.form_state.get(home_team, [])[-FORM_WINDOW:]
        form_a = self.form_state.get(away_team, [])[-FORM_WINDOW:]

        feats.update(self._form_features(form_h, "home"))
        feats.update(self._form_features(form_a, "away"))

        # --- Home/Away split form ---
        home_at_home = [m for m in self.form_state.get(home_team, []) if m["is_home"]][-FORM_WINDOW:]
        away_at_away = [m for m in self.form_state.get(away_team, []) if not m["is_home"]][-FORM_WINDOW:]

        feats.update(self._split_form_features(home_at_home, "home"))
        feats.update(self._split_form_features(away_at_away, "away"))

        # --- H2H ---
        key = tuple(sorted([home_team, away_team]))
        h2h = self.h2h_state.get(key, [])[-H2H_WINDOW:]
        feats.update(self._h2h_features(h2h, home_team, away_team))

        # --- Rest / fatigue ---
        last_h = self.last_match_date.get(home_team)
        last_a = self.last_match_date.get(away_team)
        feats.update(self._rest_features(last_h, match_date, "home"))
        feats.update(self._rest_features(last_a, match_date, "away"))

        # --- Season context ---
        feats["day_of_week"] = match_date.weekday()
        feats["month"] = match_date.month
        feats["is_weekend"] = int(match_date.weekday() >= 5)
        # Approximate match_number from history in same league/season
        n = len(self.history[
            (self.history["league_code"] == league_code) &
            (self.history["season"] == season)
        ])
        feats["match_number"] = n + 1
        feats["season_progress"] = min(1.0, (n + 1) / 380)  # ~380 per league/season

        # --- Standings (API-Football) ---
        feats.update(self._standings_features(home_team, away_team, season))

        # --- xG rolling (NaN — pre-match, handled by imputer) ---
        feats["home_xg_rolling"] = np.nan
        feats["away_xg_rolling"] = np.nan
        feats["home_xga_rolling"] = np.nan
        feats["away_xga_rolling"] = np.nan
        feats["xg_diff_rolling"] = np.nan

        # --- League dummies ---
        for lc in ["E0", "E1", "SP1", "D1", "I1", "F1"]:
            feats[f"league_{lc}"] = 1 if league_code == lc else 0

        # Build row aligned to model's feature names
        row = pd.DataFrame([feats])

        if self.feature_names:
            for col in self.feature_names:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[self.feature_names]

        return row

    # ------------------------------------------------------------------
    # Feature sub-builders
    # ------------------------------------------------------------------

    def _form_features(self, hist: List[Dict], prefix: str) -> Dict:
        if not hist:
            return {
                f"{prefix}_form_points": np.nan,
                f"{prefix}_form_goals_scored": np.nan,
                f"{prefix}_form_goals_conceded": np.nan,
                f"{prefix}_form_goal_diff": np.nan,
                f"{prefix}_form_wins": np.nan,
                f"{prefix}_form_draws": np.nan,
                f"{prefix}_form_losses": np.nan,
                f"{prefix}_form_win_rate": np.nan,
                f"{prefix}_form_shots": np.nan,
                f"{prefix}_form_shots_target": np.nan,
                f"{prefix}_avg_goals_scored": np.nan,
                f"{prefix}_avg_goals_conceded": np.nan,
            }
        wins   = sum(1 for m in hist if m["outcome"] == 1)
        draws  = sum(1 for m in hist if m["outcome"] == 0)
        losses = sum(1 for m in hist if m["outcome"] == -1)
        gf     = sum(m["gf"] for m in hist)
        ga     = sum(m["ga"] for m in hist)
        n      = len(hist)
        return {
            f"{prefix}_form_points":         wins * 3 + draws,
            f"{prefix}_form_goals_scored":   gf,
            f"{prefix}_form_goals_conceded": ga,
            f"{prefix}_form_goal_diff":      gf - ga,
            f"{prefix}_form_wins":           wins,
            f"{prefix}_form_draws":          draws,
            f"{prefix}_form_losses":         losses,
            f"{prefix}_form_win_rate":       wins / n,
            f"{prefix}_form_shots":          np.nan,
            f"{prefix}_form_shots_target":   np.nan,
            f"{prefix}_avg_goals_scored":    gf / n,
            f"{prefix}_avg_goals_conceded":  ga / n,
        }

    def _split_form_features(self, hist: List[Dict], prefix: str) -> Dict:
        if not hist:
            return {
                f"{prefix}_{prefix}_pts": np.nan,
                f"{prefix}_{prefix}_gf": np.nan,
                f"{prefix}_{prefix}_ga": np.nan,
                f"{prefix}_{prefix}_wr": np.nan,
            }
        wins = sum(1 for m in hist if m["outcome"] == 1)
        n    = len(hist)
        return {
            f"{prefix}_{prefix}_pts": wins * 3 + sum(1 for m in hist if m["outcome"] == 0),
            f"{prefix}_{prefix}_gf":  sum(m["gf"] for m in hist),
            f"{prefix}_{prefix}_ga":  sum(m["ga"] for m in hist),
            f"{prefix}_{prefix}_wr":  wins / n,
        }

    def _h2h_features(
        self, hist: List[Dict], home_team: str, away_team: str
    ) -> Dict:
        if not hist:
            return {
                "h2h_home_win_rate": np.nan,
                "h2h_draw_rate": np.nan,
                "h2h_away_win_rate": np.nan,
                "h2h_home_goals_avg": np.nan,
                "h2h_away_goals_avg": np.nan,
            }
        n = len(hist)
        home_wins = sum(
            1 for m in hist
            if (m["home"] == home_team and m["outcome"] == 1)
            or (m["away"] == home_team and m["outcome"] == -1)
        )
        draws = sum(1 for m in hist if m["outcome"] == 0)
        home_goals = [
            m["home_goals"] if m["home"] == home_team else m["away_goals"]
            for m in hist
        ]
        away_goals = [
            m["away_goals"] if m["away"] == away_team else m["home_goals"]
            for m in hist
        ]
        return {
            "h2h_home_win_rate":   home_wins / n,
            "h2h_draw_rate":       draws / n,
            "h2h_away_win_rate":   (n - home_wins - draws) / n,
            "h2h_home_goals_avg":  float(np.mean(home_goals)),
            "h2h_away_goals_avg":  float(np.mean(away_goals)),
        }

    def _rest_features(
        self,
        last_date: Optional[pd.Timestamp],
        match_date: datetime,
        prefix: str,
    ) -> Dict:
        if last_date is None or pd.isna(last_date):
            return {
                f"days_since_last_{prefix}": np.nan,
                f"{prefix}_fatigue": 0,
                f"{prefix}_rested": 1,
                f"{prefix}_rest_days": np.nan,
            }
        days = (pd.Timestamp(match_date) - last_date).days
        return {
            f"days_since_last_{prefix}": days,
            f"{prefix}_fatigue": int(days < 4),
            f"{prefix}_rested": int(days >= 7),
            f"{prefix}_rest_days": float(min(days, 30)),
        }

    def _standings_features(
        self, home_team: str, away_team: str, season: str
    ) -> Dict:
        empty = {
            "rank_diff": np.nan,
            "pts_diff": np.nan,
            "win_rate_diff_season": np.nan,
            "form_wins_diff_last5": np.nan,
            "home_win_rate_season": np.nan,
            "away_win_rate_season": np.nan,
            "home_form_wins_last5": np.nan,
            "away_form_wins_last5": np.nan,
        }
        if self.standings.empty:
            return empty

        s = self.standings[self.standings["season"] == season]
        if s.empty:
            # Fall back to latest season in standings
            s = self.standings

        h_row = s[s["team_name"].str.lower() == home_team.lower()]
        a_row = s[s["team_name"].str.lower() == away_team.lower()]
        if h_row.empty or a_row.empty:
            return empty

        h = h_row.iloc[0]
        a = a_row.iloc[0]

        def _wr(played, wins):
            return wins / played if played and played > 0 else np.nan

        h_home_wr = _wr(h.get("home_played"), h.get("home_wins"))
        a_away_wr = _wr(a.get("away_played"), a.get("away_wins"))

        def _form_wins(form_str):
            if isinstance(form_str, str):
                return form_str[-5:].count("W")
            return np.nan

        h_fw = _form_wins(h.get("form"))
        a_fw = _form_wins(a.get("form"))

        return {
            "rank_diff":             float(a.get("rank", 0) - h.get("rank", 0)),
            "pts_diff":              float(h.get("points", 0) - a.get("points", 0)),
            "win_rate_diff_season":  float(h_home_wr - a_away_wr) if not np.isnan(h_home_wr) and not np.isnan(a_away_wr) else np.nan,
            "form_wins_diff_last5":  float(h_fw - a_fw) if not np.isnan(h_fw) and not np.isnan(a_fw) else np.nan,
            "home_win_rate_season":  float(h_home_wr),
            "away_win_rate_season":  float(a_away_wr),
            "home_form_wins_last5":  float(h_fw),
            "away_form_wins_last5":  float(a_fw),
        }

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(
        self,
        fixture: Dict,
        bookmaker_odds: Optional[Dict] = None,
    ) -> MatchPrediction:
        """
        Predict outcome for an upcoming fixture dict from API-Football.

        fixture keys expected:
            fixture_id, home_team, away_team, date (ISO str),
            kickoff, league, league_code, season
        bookmaker_odds (optional):
            {"home": 2.1, "draw": 3.4, "away": 3.8}
        """
        home_raw  = fixture["home_team"]
        away_raw  = fixture["away_team"]
        home_team = self.resolve_team_name(home_raw)
        away_team = self.resolve_team_name(away_raw)
        league_code = fixture.get("league_code", "E0")
        season      = fixture.get("season", "2024-25")

        try:
            match_date = datetime.fromisoformat(fixture["date"])
        except Exception:
            match_date = datetime.now()

        if self.model is None:
            return MatchPrediction(
                fixture_id      = fixture.get("fixture_id"),
                home_team       = home_raw,
                away_team       = away_raw,
                date            = fixture["date"][:10],
                league          = fixture.get("league", ""),
                league_code     = league_code,
                kickoff         = fixture.get("kickoff", ""),
                prob_home       = 0.45,
                prob_draw       = 0.27,
                prob_away       = 0.28,
                predicted_outcome = "Home Win",
                confidence      = 0.45,
                model_available = False,
                note            = "Model not trained yet. Run: python main.py train",
            )

        X = self.build_feature_row(home_team, away_team, match_date, league_code, season)
        proba = self.model.predict_proba(X)[0]  # [P(away), P(draw), P(home)]

        prob_away, prob_draw, prob_home = float(proba[0]), float(proba[1]), float(proba[2])
        max_p = max(prob_home, prob_draw, prob_away)
        outcomes = {prob_home: "Home Win", prob_draw: "Draw", prob_away: "Away Win"}
        predicted = outcomes[max_p]

        value_bets = []
        if bookmaker_odds:
            value_bets = self._detect_value(
                prob_home, prob_draw, prob_away, bookmaker_odds
            )

        return MatchPrediction(
            fixture_id        = fixture.get("fixture_id"),
            home_team         = home_raw,
            away_team         = away_raw,
            date              = fixture["date"][:10],
            league            = fixture.get("league", ""),
            league_code       = league_code,
            kickoff           = fixture.get("kickoff", ""),
            prob_home         = prob_home,
            prob_draw         = prob_draw,
            prob_away         = prob_away,
            predicted_outcome = predicted,
            confidence        = max_p,
            value_bets        = value_bets,
            model_available   = True,
        )

    # ------------------------------------------------------------------
    # Value bet detection
    # ------------------------------------------------------------------

    def _detect_value(
        self,
        prob_h: float,
        prob_d: float,
        prob_a: float,
        odds: Dict,
    ) -> List[Dict]:
        bets = []
        mapping = [
            ("Home Win", "home", prob_h),
            ("Draw",     "draw", prob_d),
            ("Away Win", "away", prob_a),
        ]
        for label, key, prob in mapping:
            o = odds.get(key)
            if not o or o <= 1:
                continue
            imp = 1 / o
            total_imp = sum(1/odds.get(k, 99) for k in ["home", "draw", "away"] if odds.get(k))
            norm_imp = imp / total_imp if total_imp > 0 else imp
            edge = prob - norm_imp
            ev = prob * (o - 1) - (1 - prob)
            if edge > 0.03:
                kelly = max(0.0, min(0.05, 0.25 * edge / (o - 1)))
                bets.append({
                    "outcome":      label,
                    "odds":         round(o, 2),
                    "model_prob":   round(prob, 3),
                    "implied_prob": round(norm_imp, 3),
                    "edge":         round(edge, 3),
                    "ev":           round(ev, 3),
                    "kelly_pct":    round(kelly * 100, 2),
                })
        bets.sort(key=lambda x: x["edge"], reverse=True)
        return bets

    # ------------------------------------------------------------------
    # Team insight (for UI)
    # ------------------------------------------------------------------

    def get_team_insight(self, team_name: str) -> Dict:
        """Return last-5 form, Elo, home/away record for a team card."""
        resolved = self.resolve_team_name(team_name)
        form = self.form_state.get(resolved, [])[-5:]
        elo  = self.elo_state.get(resolved, ELO_BASE)
        last = self.last_match_date.get(resolved)

        form_str = "".join(
            "W" if m["outcome"] == 1 else ("D" if m["outcome"] == 0 else "L")
            for m in form
        )
        return {
            "team":     resolved,
            "elo":      round(elo, 0),
            "form":     form_str or "—",
            "last_match": last.strftime("%Y-%m-%d") if last else None,
        }
