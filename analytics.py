"""
analytics.py
------------
Stateless analytics functions.  Every function accepts DataFrames (already
filtered by season / team / player upstream) and returns a DataFrame or
scalar that the page modules can feed straight to Plotly.

Design principles
-----------------
* Pure functions – no global state, easy to unit-test.
* Each function has a single responsibility.
* Heavy computation is cached by Streamlit's @st.cache_data at the call site.
"""

import pandas as pd
import numpy as np


# ============================================================
# OVERVIEW
# ============================================================

def overview_stats(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict:
    """
    Return a dict of headline KPIs for the Overview page.
    """
    total_matches = len(matches)
    seasons = sorted(matches["season"].unique())
    teams = sorted(set(matches["team1"].tolist() + matches["team2"].tolist()))
    total_runs = int(deliveries["total_runs"].sum())
    total_wickets = int(deliveries["player_dismissed"].astype(bool).sum())
    total_sixes = int((deliveries["batsman_runs"] == 6).sum())
    total_fours = int((deliveries["batsman_runs"] == 4).sum())
    avg_first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id")["total_runs"].sum()
        .mean()
    )
    return {
        "total_matches": total_matches,
        "total_seasons": len(seasons),
        "total_teams": len(teams),
        "total_runs": total_runs,
        "total_wickets": total_wickets,
        "total_sixes": total_sixes,
        "total_fours": total_fours,
        "avg_first_innings_score": round(avg_first_innings, 1),
        "seasons": seasons,
        "teams": teams,
    }


def matches_per_season(matches: pd.DataFrame) -> pd.DataFrame:
    return (
        matches.groupby("season").size()
        .reset_index(name="matches")
        .sort_values("season")
    )


def toss_impact(matches: pd.DataFrame) -> pd.DataFrame:
    """Proportion of matches won by toss winner, by decision."""
    df = matches[matches["winner"].notna() & (matches["winner"] != "No Result")].copy()
    df["toss_won_match"] = df["toss_winner"] == df["winner"]
    return (
        df.groupby("toss_decision")["toss_won_match"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "wins", "count": "total"})
        .assign(win_pct=lambda x: (x["wins"] / x["total"] * 100).round(1))
        .reset_index()
    )


# ============================================================
# BATTING
# ============================================================

def batting_scorecard(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated batting stats per player.
    Returns: batsman, runs, balls, 4s, 6s, strike_rate, innings, average, highest_score
    """
    df = deliveries.copy()

    runs = (
        df.groupby("batsman")["batsman_runs"]
        .sum()
        .reset_index(name="runs")
    )
    balls = (
        df[df["wide_runs"] == 0]
        .groupby("batsman")
        .size()
        .reset_index(name="balls_faced")
    )
    fours = (
        df[df["batsman_runs"] == 4]
        .groupby("batsman").size()
        .reset_index(name="fours")
    )
    sixes = (
        df[df["batsman_runs"] == 6]
        .groupby("batsman").size()
        .reset_index(name="sixes")
    )

    # Innings played (unique match_id appearances)
    innings = (
        df.groupby("batsman")["match_id"].nunique()
        .reset_index(name="innings")
    )

    # Dismissals
    dismissals = (
        df[df["player_dismissed"] != ""]
        .groupby("player_dismissed").size()
        .reset_index()
        .rename(columns={"player_dismissed": "batsman", 0: "dismissals"})
    )

    # Highest individual score per match
    match_scores = (
        df.groupby(["batsman", "match_id"])["batsman_runs"]
        .sum()
        .reset_index()
    )
    highest = (
        match_scores.groupby("batsman")["batsman_runs"]
        .max()
        .reset_index(name="highest_score")
    )

    # Merge all together
    scorecard = (
        runs
        .merge(balls, on="batsman", how="left")
        .merge(fours, on="batsman", how="left")
        .merge(sixes, on="batsman", how="left")
        .merge(innings, on="batsman", how="left")
        .merge(dismissals, on="batsman", how="left")
        .merge(highest, on="batsman", how="left")
        .fillna(0)
    )
    scorecard["strike_rate"] = (
        scorecard["runs"] / scorecard["balls_faced"] * 100
    ).round(2)
    scorecard["average"] = np.where(
        scorecard["dismissals"] > 0,
        (scorecard["runs"] / scorecard["dismissals"]).round(2),
        scorecard["runs"],
    )
    # Filter noise – minimum 5 innings
    scorecard = scorecard[scorecard["innings"] >= 5].copy()
    return scorecard.sort_values("runs", ascending=False).reset_index(drop=True)


def batsman_match_timeline(deliveries: pd.DataFrame, matches: pd.DataFrame,
                           batsman: str) -> pd.DataFrame:
    """Runs scored per match for a single batsman, with season info."""
    df = deliveries[deliveries["batsman"] == batsman].copy()
    per_match = (
        df.groupby("match_id")["batsman_runs"]
        .sum()
        .reset_index(name="runs")
    )
    per_match = per_match.merge(matches[["id", "season"]], left_on="match_id",
                                right_on="id", how="left")
    return per_match.sort_values("match_id")


def phase_analysis(deliveries: pd.DataFrame, batsman: str) -> pd.DataFrame:
    """Runs and strike rate by powerplay / middle / death for a batsman."""
    df = deliveries[deliveries["batsman"] == batsman].copy()

    def phase(over):
        if over <= 6:
            return "Powerplay (1-6)"
        elif over <= 15:
            return "Middle (7-15)"
        else:
            return "Death (16-20)"

    df["phase"] = df["over"].apply(phase)
    agg = (
        df.groupby("phase")
        .agg(runs=("batsman_runs", "sum"), balls=("batsman_runs", "count"))
        .reset_index()
    )
    agg["strike_rate"] = (agg["runs"] / agg["balls"] * 100).round(1)
    return agg


# ============================================================
# BOWLING
# ============================================================

def bowling_scorecard(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated bowling stats per bowler.
    Returns: bowler, matches, overs, runs_conceded, wickets, economy, average, best_figures
    """
    df = deliveries.copy()

    # Legal deliveries (exclude wides & no-balls for over calculation)
    legal = df[(df["wide_runs"] == 0) & (df["noball_runs"] == 0)]
    overs_df = (
        legal.groupby("bowler").size()
        .div(6).round(1)
        .reset_index(name="overs")
    )

    runs_df = (
        df.groupby("bowler")["total_runs"]
        .sum()
        .reset_index(name="runs_conceded")
    )

    wickets_df = (
        df[df["dismissal_kind"].isin(
            ["caught", "bowled", "lbw", "caught and bowled", "stumped", "hit wicket"]
        )]
        .groupby("bowler").size()
        .reset_index(name="wickets")
    )

    matches_df = (
        df.groupby("bowler")["match_id"].nunique()
        .reset_index(name="matches")
    )

    # Best figures: highest wickets in a single match (tie-break: fewest runs)
    match_figures = (
        df[df["dismissal_kind"].isin(
            ["caught", "bowled", "lbw", "caught and bowled", "stumped", "hit wicket"]
        )]
        .groupby(["bowler", "match_id"]).size()
        .reset_index(name="wkts")
    )
    match_runs = (
        df.groupby(["bowler", "match_id"])["total_runs"]
        .sum()
        .reset_index(name="m_runs")
    )
    match_figures = match_figures.merge(match_runs, on=["bowler", "match_id"])
    best = (
        match_figures.sort_values(["wkts", "m_runs"], ascending=[False, True])
        .groupby("bowler")
        .first()
        .reset_index()
        .rename(columns={"wkts": "best_wkts", "m_runs": "best_runs"})
    )
    best["best_figures"] = best["best_wkts"].astype(str) + "/" + best["best_runs"].astype(str)

    scorecard = (
        overs_df
        .merge(runs_df, on="bowler", how="left")
        .merge(wickets_df, on="bowler", how="left")
        .merge(matches_df, on="bowler", how="left")
        .merge(best[["bowler", "best_figures"]], on="bowler", how="left")
        .fillna({"wickets": 0, "best_figures": "0/0"})
    )
    scorecard["economy"] = (scorecard["runs_conceded"] / scorecard["overs"].replace(0, np.nan)).round(2)
    scorecard["average"] = np.where(
        scorecard["wickets"] > 0,
        (scorecard["runs_conceded"] / scorecard["wickets"]).round(2),
        np.inf,
    )
    scorecard["wickets"] = scorecard["wickets"].astype(int)
    # Minimum 10 overs bowled
    scorecard = scorecard[scorecard["overs"] >= 10].copy()
    return scorecard.sort_values("wickets", ascending=False).reset_index(drop=True)


# ============================================================
# TEAM
# ============================================================

def team_win_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Win counts, losses, win percentage per team."""
    df = matches[matches["winner"].notna() & (matches["winner"] != "No Result")].copy()

    all_teams = set(df["team1"].tolist() + df["team2"].tolist())
    records = []
    for team in all_teams:
        played = len(df[(df["team1"] == team) | (df["team2"] == team)])
        won = len(df[df["winner"] == team])
        records.append({"team": team, "played": played, "won": won,
                        "lost": played - won})

    result = pd.DataFrame(records)
    result["win_pct"] = (result["won"] / result["played"] * 100).round(1)
    return result.sort_values("win_pct", ascending=False).reset_index(drop=True)


def head_to_head(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """Return H2H record between two teams."""
    df = matches[
        ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
        ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    ].copy()
    df = df[df["winner"].notna() & (df["winner"] != "No Result")]

    a_wins = int((df["winner"] == team_a).sum())
    b_wins = int((df["winner"] == team_b).sum())
    total = len(df)
    return {
        "total": total, "a_wins": a_wins, "b_wins": b_wins,
        "team_a": team_a, "team_b": team_b,
    }


def team_season_performance(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Win % per season for a team."""
    df = matches[(matches["team1"] == team) | (matches["team2"] == team)].copy()
    df = df[df["winner"].notna() & (df["winner"] != "No Result")]
    seasons = []
    for season, grp in df.groupby("season"):
        played = len(grp)
        won = int((grp["winner"] == team).sum())
        seasons.append({"season": season, "played": played, "won": won,
                        "win_pct": round(won / played * 100, 1)})
    return pd.DataFrame(seasons)


def run_rate_by_over(deliveries: pd.DataFrame, team: str = None) -> pd.DataFrame:
    """Average runs scored per over number (powerplay / middle / death analysis)."""
    df = deliveries[deliveries["inning"] == 1].copy()
    if team:
        df = df[df["batting_team"] == team]
    return (
        df.groupby("over")["total_runs"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"total_runs": "avg_runs"})
    )
