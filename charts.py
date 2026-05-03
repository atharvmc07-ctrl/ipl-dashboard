"""
charts.py
---------
Centralised Plotly figure factory.

All charts share a consistent dark theme (DARK_TEMPLATE).
Each function returns a go.Figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DARK_BG = "#0e1117"
CARD_BG = "#1a1f2e"
ACCENT = "#e84545"
ACCENT2 = "#f5a623"
ACCENT3 = "#53d8fb"
GRID = "#2a2f3e"
TEXT = "#e0e0e0"
MUTED = "#8b949e"

DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family="Inter, sans-serif", size=13),
        title=dict(font=dict(size=16, color=TEXT), x=0.02),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=[ACCENT, ACCENT2, ACCENT3, "#7b68ee", "#2ecc71",
                  "#e74c3c", "#3498db", "#9b59b6", "#1abc9c", "#f39c12"],
    )
)


def _apply(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template=DARK_TEMPLATE, height=height)
    return fig


# ---------------------------------------------------------------------------
# Overview charts
# ---------------------------------------------------------------------------

def matches_per_season_bar(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="season", y="matches",
                 title="Matches Played Per Season",
                 labels={"season": "Season", "matches": "Matches"},
                 color="matches",
                 color_continuous_scale=[[0, "#1a1f2e"], [1, ACCENT]])
    fig.update_coloraxes(showscale=False)
    return _apply(fig)


def toss_impact_pie(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=[f"{r['toss_decision'].title()} – Won" for _, r in df.iterrows()],
        values=df["wins"].tolist(),
        hole=0.55,
        marker=dict(colors=[ACCENT, ACCENT2], line=dict(color=CARD_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(title="Matches Won After Winning Toss by Decision")
    return _apply(fig, height=340)


# ---------------------------------------------------------------------------
# Batting charts
# ---------------------------------------------------------------------------

def top_run_scorers_bar(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.head(n).sort_values("runs")
    fig = px.bar(top, x="runs", y="batsman", orientation="h",
                 title=f"Top {n} Run Scorers",
                 labels={"runs": "Total Runs", "batsman": ""},
                 color="runs",
                 color_continuous_scale=[[0, "#1a2e3a"], [1, ACCENT3]])
    fig.update_coloraxes(showscale=False)
    return _apply(fig, height=480)


def strike_rate_vs_average_scatter(df: pd.DataFrame) -> go.Figure:
    top = df[df["innings"] >= 10].head(50)
    fig = px.scatter(
        top, x="average", y="strike_rate",
        size="runs", color="innings",
        hover_name="batsman",
        title="Strike Rate vs Average (bubble = total runs)",
        labels={"average": "Batting Average", "strike_rate": "Strike Rate"},
        color_continuous_scale=[[0, ACCENT2], [1, ACCENT]],
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
    return _apply(fig, height=420)


def batsman_timeline(df: pd.DataFrame, batsman: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["match_id"], y=df["runs"],
        mode="lines+markers",
        line=dict(color=ACCENT3, width=1.5),
        marker=dict(size=5, color=ACCENT3),
        name="Runs",
        fill="tozeroy",
        fillcolor="rgba(83,216,251,0.08)",
    ))
    # Rolling average
    df = df.copy()
    df["rolling_avg"] = df["runs"].rolling(10, min_periods=3).mean()
    fig.add_trace(go.Scatter(
        x=df["match_id"], y=df["rolling_avg"],
        mode="lines", line=dict(color=ACCENT, width=2, dash="dot"),
        name="10-match avg",
    ))
    fig.update_layout(title=f"{batsman} – Match-by-Match Runs",
                      xaxis_title="Match", yaxis_title="Runs")
    return _apply(fig, height=360)


def phase_bar(df: pd.DataFrame, batsman: str) -> go.Figure:
    fig = go.Figure()
    order = ["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"]
    df["phase"] = pd.Categorical(df["phase"], categories=order, ordered=True)
    df = df.sort_values("phase")
    fig.add_trace(go.Bar(
        x=df["phase"], y=df["runs"],
        name="Runs", marker_color=ACCENT,
        text=df["runs"], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.add_trace(go.Scatter(
        x=df["phase"], y=df["strike_rate"],
        name="Strike Rate", yaxis="y2",
        mode="lines+markers",
        line=dict(color=ACCENT2, width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title=f"{batsman} – Performance by Phase",
        yaxis=dict(title="Runs"),
        yaxis2=dict(title="Strike Rate", overlaying="y", side="right", showgrid=False),
        barmode="group",
    )
    return _apply(fig, height=360)


def player_comparison_radar(scorecard: pd.DataFrame,
                             p1: str, p2: str) -> go.Figure:
    metrics = ["runs", "average", "strike_rate", "sixes", "fours"]
    labels = ["Total Runs", "Average", "Strike Rate", "Sixes", "Fours"]

    def get_vals(player):
        row = scorecard[scorecard["batsman"] == player]
        if row.empty:
            return [0] * len(metrics)
        return [float(row[m].values[0]) for m in metrics]

    v1, v2 = get_vals(p1), get_vals(p2)

    # Normalize to 0-100
    max_vals = [max(v1[i], v2[i], 1) for i in range(len(metrics))]
    v1n = [v1[i] / max_vals[i] * 100 for i in range(len(metrics))]
    v2n = [v2[i] / max_vals[i] * 100 for i in range(len(metrics))]

    fig = go.Figure()
    for vals, name, color in [(v1n, p1, ACCENT), (v2n, p2, ACCENT3)]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=name,
            line=dict(color=color),
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba")
                       if "rgb" in color else color + "26",
            opacity=0.9,
        ))
    fig.update_layout(
        title=f"Player Comparison: {p1} vs {p2}",
        polar=dict(
            bgcolor=CARD_BG,
            angularaxis=dict(tickfont=dict(color=MUTED)),
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=MUTED),
                            gridcolor=GRID, tickvals=[25, 50, 75, 100]),
        ),
    )
    return _apply(fig, height=420)


# ---------------------------------------------------------------------------
# Bowling charts
# ---------------------------------------------------------------------------

def top_wicket_takers_bar(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.head(n).sort_values("wickets")
    fig = px.bar(top, x="wickets", y="bowler", orientation="h",
                 title=f"Top {n} Wicket Takers",
                 labels={"wickets": "Wickets", "bowler": ""},
                 color="wickets",
                 color_continuous_scale=[[0, "#2e1a1a"], [1, ACCENT]])
    fig.update_coloraxes(showscale=False)
    return _apply(fig, height=480)


def economy_vs_wickets_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="economy", y="wickets",
        size="overs", color="average",
        hover_name="bowler",
        title="Economy Rate vs Wickets (bubble = overs bowled)",
        labels={"economy": "Economy Rate", "wickets": "Wickets"},
        color_continuous_scale=[[0, ACCENT3], [1, ACCENT]],
    )
    return _apply(fig, height=420)


# ---------------------------------------------------------------------------
# Team charts
# ---------------------------------------------------------------------------

def team_win_pct_bar(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="team", y="win_pct",
                 title="Team Win Percentage (All Seasons)",
                 labels={"team": "", "win_pct": "Win %"},
                 color="win_pct",
                 color_continuous_scale=[[0, "#1a2e1a"], [1, "#2ecc71"]],
                 text=df["win_pct"].astype(str) + "%")
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig, height=400)


def team_season_line(df: pd.DataFrame, team: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["season"], y=df["win_pct"],
        mode="lines+markers",
        line=dict(color=ACCENT2, width=2.5),
        marker=dict(size=8, color=ACCENT2, line=dict(color=CARD_BG, width=2)),
        fill="tozeroy",
        fillcolor="rgba(245,166,35,0.08)",
    ))
    fig.update_layout(title=f"{team} – Win % by Season",
                      xaxis_title="Season", yaxis_title="Win %")
    return _apply(fig, height=340)


def run_rate_by_over_line(df: pd.DataFrame, team: str = "All Teams") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["over"], y=df["avg_runs"],
        mode="lines+markers",
        line=dict(color=ACCENT3, width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(83,216,251,0.07)",
    ))
    # Add phase shading
    for start, end, label, col in [
        (1, 6, "Powerplay", "rgba(83,216,251,0.06)"),
        (16, 20, "Death", "rgba(232,69,69,0.06)"),
    ]:
        fig.add_vrect(x0=start - 0.5, x1=end + 0.5,
                      fillcolor=col, line_width=0,
                      annotation_text=label,
                      annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    fig.update_layout(title=f"Average Runs Per Over – {team}",
                      xaxis_title="Over", yaxis_title="Avg Runs")
    return _apply(fig, height=340)


def h2h_donut(h2h: dict) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=[h2h["team_a"], h2h["team_b"]],
        values=[h2h["a_wins"], h2h["b_wins"]],
        hole=0.6,
        marker=dict(colors=[ACCENT, ACCENT3], line=dict(color=CARD_BG, width=3)),
        textinfo="label+value",
        textfont=dict(color=TEXT, size=13),
    ))
    fig.update_layout(
        title=f"Head-to-Head: {h2h['team_a']} vs {h2h['team_b']}",
        annotations=[dict(text=f"{h2h['total']}<br>Games", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color=MUTED)],
    )
    return _apply(fig, height=360)


# ---------------------------------------------------------------------------
# ML charts
# ---------------------------------------------------------------------------

def win_probability_gauge(team: str, probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": f"{team}<br><span style='font-size:0.8em;color:{MUTED}'>Win Probability</span>"},
        number={"suffix": "%", "font": {"color": TEXT, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED},
            "bar": {"color": ACCENT if probability > 50 else ACCENT3},
            "bgcolor": GRID,
            "bordercolor": GRID,
            "steps": [
                {"range": [0, 40], "color": "rgba(83,216,251,0.15)"},
                {"range": [60, 100], "color": "rgba(232,69,69,0.15)"},
            ],
            "threshold": {
                "line": {"color": ACCENT2, "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    return _apply(fig, height=280)


def feature_importance_bar(df: pd.DataFrame) -> go.Figure:
    labels = {
        "toss_is_t1": "Won Toss",
        "toss_bat": "Chose to Bat",
        "season_norm": "Season",
        "t1_win_rate": "Team 1 Win Rate",
        "t2_win_rate": "Team 2 Win Rate",
        "h2h_win_rate": "H2H Win Rate",
        "win_rate_diff": "Win Rate Differential",
    }
    df = df.copy()
    df["feature"] = df["feature"].map(labels).fillna(df["feature"])
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 title="Model Feature Importance",
                 labels={"importance": "Importance", "feature": ""},
                 color="importance",
                 color_continuous_scale=[[0, "#1a1a2e"], [1, ACCENT2]])
    fig.update_coloraxes(showscale=False)
    return _apply(fig, height=340)
