"""
ml_predictor.py
---------------
Match-winner prediction using a Random Forest classifier.

Features used
-------------
- Toss winner (binary: is toss winner = team1?)
- Toss decision (bat / field → encoded)
- Season (ordinal)
- Historical win rate of each team (computed from training data)
- Head-to-head win rate (team1 vs team2 historically)

This is intentionally kept transparent so users can understand
what drives the prediction. A gradient boosting model (XGBoost)
would be more accurate in production; swap the estimator to upgrade.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ipl_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ipl_model_meta.joblib")


class IPLPredictor:
    """
    Encapsulates feature engineering, training, serialisation, and prediction.
    """

    def __init__(self):
        self.model: GradientBoostingClassifier | None = None
        self.team_win_rates: dict = {}
        self.h2h_win_rates: dict = {}
        self.teams: list = []
        self.accuracy: float = 0.0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_features(self, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = matches[
            matches["winner"].notna() &
            (matches["winner"] != "No Result")
        ].copy().reset_index(drop=True)

        # Pre-compute cumulative win rates (no data leakage: use only past rows)
        team_wins: dict[str, list] = {}
        team_played: dict[str, list] = {}
        h2h_wins: dict[tuple, list] = {}
        h2h_played: dict[tuple, list] = {}

        rows = []
        for _, row in df.iterrows():
            t1, t2 = row["team1"], row["team2"]
            season = row["season"]
            toss_is_t1 = 1 if row["toss_winner"] == t1 else 0
            toss_bat = 1 if row["toss_decision"] == "bat" else 0

            # Historical win rates (prior to this match)
            t1_wr = (sum(team_wins.get(t1, [])) / max(len(team_played.get(t1, [1])), 1))
            t2_wr = (sum(team_wins.get(t2, [])) / max(len(team_played.get(t2, [1])), 1))

            key = tuple(sorted([t1, t2]))
            h2h_wr = (sum(h2h_wins.get(key, [])) / max(len(h2h_played.get(key, [1])), 1))
            # h2h_wr from t1's perspective
            if key[0] != t1:
                h2h_wr = 1 - h2h_wr

            rows.append({
                "toss_is_t1": toss_is_t1,
                "toss_bat": toss_bat,
                "season_norm": (season - 2008) / 15,
                "t1_win_rate": round(t1_wr, 3),
                "t2_win_rate": round(t2_wr, 3),
                "h2h_win_rate": round(h2h_wr, 3),
                "win_rate_diff": round(t1_wr - t2_wr, 3),
            })

            # Update accumulators
            t1_won = 1 if row["winner"] == t1 else 0
            team_wins.setdefault(t1, []).append(t1_won)
            team_wins.setdefault(t2, []).append(1 - t1_won)
            team_played.setdefault(t1, []).append(1)
            team_played.setdefault(t2, []).append(1)
            h2h_wins.setdefault(key, []).append(t1_won if key[0] == t1 else 1 - t1_won)
            h2h_played.setdefault(key, []).append(1)

        X = pd.DataFrame(rows)
        y = (df["winner"] == df["team1"]).astype(int)  # 1 = team1 wins

        # Store final win rates for prediction
        for team in set(df["team1"].tolist() + df["team2"].tolist()):
            played = len(team_played.get(team, []))
            wins = sum(team_wins.get(team, []))
            self.team_win_rates[team] = wins / max(played, 1)
        self.h2h_win_rates = {
            k: sum(v) / len(v) for k, v in h2h_wins.items()
        }
        self.teams = sorted(set(df["team1"].tolist() + df["team2"].tolist()))

        return X, y

    # ------------------------------------------------------------------
    # Train & persist
    # ------------------------------------------------------------------
    def train(self, matches: pd.DataFrame) -> float:
        X, y = self._build_features(matches)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, preds) * 100, 1)

        joblib.dump(self, MODEL_PATH)
        logger.info(f"Model trained – accuracy: {self.accuracy}%")
        return self.accuracy

    @staticmethod
    def load() -> "IPLPredictor":
        return joblib.load(MODEL_PATH)

    @staticmethod
    def is_trained() -> bool:
        return os.path.exists(MODEL_PATH)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, team1: str, team2: str,
                toss_winner: str, toss_decision: str,
                season: int) -> dict:
        """
        Returns probability dict: {team1: float, team2: float}
        """
        t1_wr = self.team_win_rates.get(team1, 0.5)
        t2_wr = self.team_win_rates.get(team2, 0.5)
        key = tuple(sorted([team1, team2]))
        h2h = self.h2h_win_rates.get(key, 0.5)
        if key[0] != team1:
            h2h = 1 - h2h

        X = pd.DataFrame([{
            "toss_is_t1": 1 if toss_winner == team1 else 0,
            "toss_bat": 1 if toss_decision == "Bat" else 0,
            "season_norm": (season - 2008) / 15,
            "t1_win_rate": t1_wr,
            "t2_win_rate": t2_wr,
            "h2h_win_rate": h2h,
            "win_rate_diff": t1_wr - t2_wr,
        }])
        proba = self.model.predict_proba(X)[0]
        return {team1: round(proba[1] * 100, 1), team2: round(proba[0] * 100, 1)}

    def feature_importance(self) -> pd.DataFrame:
        features = ["toss_is_t1", "toss_bat", "season_norm",
                    "t1_win_rate", "t2_win_rate", "h2h_win_rate", "win_rate_diff"]
        return pd.DataFrame({
            "feature": features,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=True)
