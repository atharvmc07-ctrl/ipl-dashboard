"""
data_loader.py
--------------
Handles downloading, caching, and loading of IPL datasets.
Source: Kaggle IPL Complete Dataset (matches.csv + deliveries.csv)
Falls back to synthetic data if network unavailable.
"""

import os
import pandas as pd
import numpy as np
import requests
import io
import logging

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_PATH = os.path.join(DATA_DIR, "deliveries.csv")

# ---------------------------------------------------------------------------
# Public columns expected downstream – kept as constants for easy refactoring
# ---------------------------------------------------------------------------
MATCH_COLS = [
    "id", "season", "city", "date", "team1", "team2",
    "toss_winner", "toss_decision", "result", "dl_applied",
    "winner", "win_by_runs", "win_by_wickets",
    "player_of_match", "venue", "umpire1", "umpire2",
]
DELIVERY_COLS = [
    "match_id", "inning", "batting_team", "bowling_team",
    "over", "ball", "batsman", "non_striker", "bowler",
    "is_super_over", "wide_runs", "bye_runs", "legbye_runs",
    "noball_runs", "penalty_runs", "batsman_runs",
    "extra_runs", "total_runs", "player_dismissed",
    "dismissal_kind", "fielder",
]


# ---------------------------------------------------------------------------
# Synthetic dataset generator (used when real data is absent)
# ---------------------------------------------------------------------------
def _generate_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce a representative synthetic IPL dataset for 2008-2023 (16 seasons).
    Realistic enough to power all dashboard features.
    """
    rng = np.random.default_rng(42)

    teams = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
        "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants",
        "Gujarat Titans",
    ]
    batsmen = [
        "V Kohli", "RG Sharma", "DA Warner", "KL Rahul", "MS Dhoni",
        "AB de Villiers", "SK Raina", "RR Pant", "HH Pandya", "KA Pollard",
        "SR Watson", "G Gambhir", "CH Gayle", "AM Rahane", "S Dhawan",
        "PP Shaw", "IS Sodhi", "Q de Kock", "JM Bairstow", "KS Williamson",
    ]
    bowlers = [
        "JJ Bumrah", "SL Malinga", "DW Steyn", "B Kumar", "RA Jadeja",
        "Harbhajan Singh", "YS Chahal", "PP Chawla", "A Nehra", "Z Khan",
        "SP Narine", "DJ Bravo", "MM Sharma", "A Mishra", "UT Yadav",
    ]
    cities = ["Mumbai", "Chennai", "Bangalore", "Kolkata", "Hyderabad",
              "Delhi", "Jaipur", "Mohali", "Pune", "Ahmedabad"]
    venues = [
        "Wankhede Stadium", "MA Chidambaram Stadium",
        "M Chinnaswamy Stadium", "Eden Gardens",
        "Rajiv Gandhi International Cricket Stadium",
        "Arun Jaitley Stadium", "Sawai Mansingh Stadium",
        "Punjab Cricket Association IS Bindra Stadium",
        "Maharashtra Cricket Association Stadium",
        "Narendra Modi Stadium",
    ]

    matches_rows = []
    deliveries_rows = []
    match_id = 1

    for season in range(2008, 2024):
        n_matches = rng.integers(56, 75)
        for _ in range(n_matches):
            t1, t2 = rng.choice(teams, size=2, replace=False)
            toss_winner = rng.choice([t1, t2])
            toss_decision = rng.choice(["bat", "field"])
            winner = rng.choice([t1, t2, None], p=[0.46, 0.46, 0.08])
            result = "normal" if winner else "no result"
            win_by_runs = int(rng.integers(1, 80)) if (winner and toss_decision == "bat") else 0
            win_by_wickets = int(rng.integers(1, 10)) if (winner and toss_decision == "field") else 0
            city_idx = rng.integers(len(cities))

            matches_rows.append({
                "id": match_id,
                "season": season,
                "city": cities[city_idx],
                "date": f"{season}-04-{rng.integers(1,30):02d}",
                "team1": t1, "team2": t2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "result": result, "dl_applied": 0,
                "winner": winner,
                "win_by_runs": win_by_runs,
                "win_by_wickets": win_by_wickets,
                "player_of_match": rng.choice(batsmen),
                "venue": venues[city_idx],
                "umpire1": "Umpire A", "umpire2": "Umpire B",
            })

            # Generate ~240 deliveries per match (2 innings × ~120 balls)
            for inning in [1, 2]:
                bat_team = t1 if inning == 1 else t2
                bowl_team = t2 if inning == 1 else t1
                inning_batsmen = rng.choice(batsmen, size=8, replace=False)
                inning_bowlers = rng.choice(bowlers, size=5, replace=False)
                current_batsman = inning_batsmen[0]
                batsman_idx = 0
                wickets = 0

                for over in range(20):
                    bowler = inning_bowlers[over % len(inning_bowlers)]
                    for ball in range(1, 7):
                        wide = int(rng.random() < 0.04)
                        noball = int(rng.random() < 0.02)
                        batsman_runs = int(rng.choice([0,1,2,3,4,6], p=[0.35,0.3,0.1,0.02,0.15,0.08]))
                        extra_runs = wide + noball
                        dismissed = ""
                        dismissal = ""
                        if not wide and not noball and rng.random() < 0.045 and wickets < 9:
                            dismissed = current_batsman
                            dismissal = rng.choice(["caught","bowled","lbw","run out","stumped"])
                            wickets += 1
                            batsman_idx = min(batsman_idx + 1, len(inning_batsmen) - 1)
                            current_batsman = inning_batsmen[batsman_idx]

                        deliveries_rows.append({
                            "match_id": match_id,
                            "inning": inning,
                            "batting_team": bat_team,
                            "bowling_team": bowl_team,
                            "over": over + 1,
                            "ball": ball,
                            "batsman": current_batsman,
                            "non_striker": inning_batsmen[(batsman_idx + 1) % len(inning_batsmen)],
                            "bowler": bowler,
                            "is_super_over": 0,
                            "wide_runs": wide,
                            "bye_runs": 0, "legbye_runs": 0,
                            "noball_runs": noball,
                            "penalty_runs": 0,
                            "batsman_runs": batsman_runs,
                            "extra_runs": extra_runs,
                            "total_runs": batsman_runs + extra_runs,
                            "player_dismissed": dismissed,
                            "dismissal_kind": dismissal,
                            "fielder": "",
                        })
            match_id += 1

    matches_df = pd.DataFrame(matches_rows)
    deliveries_df = pd.DataFrame(deliveries_rows)
    return matches_df, deliveries_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_data(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IPL matches and deliveries DataFrames.

    Priority:
      1. Local CSV files in /data directory
      2. Synthetic data (always available as fallback)

    Returns
    -------
    matches_df, deliveries_df
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_synthetic and os.path.exists(MATCHES_PATH) and os.path.exists(DELIVERIES_PATH):
        logger.info("Loading data from local CSV files...")
        matches_df = pd.read_csv(MATCHES_PATH)
        deliveries_df = pd.read_csv(DELIVERIES_PATH)
        matches_df, deliveries_df = _clean_data(matches_df, deliveries_df)
        return matches_df, deliveries_df

    logger.info("Generating synthetic IPL data...")
    matches_df, deliveries_df = _generate_synthetic_data()
    # Cache locally for repeat runs
    matches_df.to_csv(MATCHES_PATH, index=False)
    deliveries_df.to_csv(DELIVERIES_PATH, index=False)
    return matches_df, deliveries_df


def _clean_data(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame):
    """Standardise column names and types for real Kaggle datasets."""
    # Kaggle dataset uses 'id' for match id; deliveries uses 'id' too
    if "id" in matches_df.columns:
        matches_df = matches_df.rename(columns={"id": "id"})
    if "id" in deliveries_df.columns:
        deliveries_df = deliveries_df.rename(columns={"id": "match_id"})

    # Fill missing winners
    matches_df["winner"] = matches_df.get("winner", pd.Series(dtype=str)).fillna("No Result")
    matches_df["win_by_runs"] = matches_df.get("win_by_runs", pd.Series(dtype=int)).fillna(0).astype(int)
    matches_df["win_by_wickets"] = matches_df.get("win_by_wickets", pd.Series(dtype=int)).fillna(0).astype(int)

    # Deliveries
    for col in ["player_dismissed", "dismissal_kind", "fielder"]:
        if col in deliveries_df.columns:
            deliveries_df[col] = deliveries_df[col].fillna("")

    return matches_df, deliveries_df
