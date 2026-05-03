"""
app.py
------
Entry point for the IPL Stats Dashboard.

Run with:
    streamlit run app.py

Architecture
------------
* Sidebar: global filters (season, team, player search)
* Main area: page modules rendered based on navigation selection
* Data is loaded once per session and cached via st.cache_data
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.pages import overview, batting, bowling, teams, predictor

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Stats Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #12161f;
        border-right: 1px solid #1e2330;
    }
    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* ── Tabs ── */
    .stTabs [role="tab"] { color: #8b949e; }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #e84545;
        border-bottom: 2px solid #e84545;
    }
    /* ── DataFrames ── */
    [data-testid="stDataFrame"] { border: 1px solid #2a2f3e; border-radius: 8px; }
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #e84545, #c0392b);
        color: white; border: none; border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { opacity: 0.9; }
    /* ── Selectbox ── */
    .stSelectbox > div > div { background: #1a1f2e; border-color: #2a2f3e; }
    /* ── Hide Streamlit branding ── */
    #MainMenu, footer { visibility: hidden; }
    /* ── Divider ── */
    hr { border-color: #2a2f3e; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading IPL data…")
def get_data():
    return load_data()


matches_raw, deliveries_raw = get_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px;">
        <span style="font-size:40px;">🏏</span>
        <h2 style="margin:4px 0 2px; color:#e84545; font-size:22px;">IPL Dashboard</h2>
        <span style="font-size:11px; color:#8b949e;">2008 – 2023 · Production Edition</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["🏠 Overview", "🏏 Batting", "🎯 Bowling", "🏆 Teams", "🤖 Predictor"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔧 Filters")

    # Season filter
    all_seasons = sorted(matches_raw["season"].unique())
    season_range = st.select_slider(
        "Season Range",
        options=all_seasons,
        value=(all_seasons[0], all_seasons[-1]),
    )

    # Team filter
    all_teams = sorted(set(matches_raw["team1"].tolist() + matches_raw["team2"].tolist()))
    selected_teams = st.multiselect(
        "Filter Teams",
        options=all_teams,
        default=[],
        placeholder="All teams",
    )

    # Player search
    all_players = sorted(deliveries_raw["batsman"].unique())
    player_search = st.selectbox(
        "Quick Player Search",
        options=[""] + all_players,
        index=0,
        placeholder="Search player…",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;text-align:center;'>"
        "Data: Kaggle IPL Dataset<br>Built with Streamlit + Plotly</div>",
        unsafe_allow_html=True,
    )

# ── Apply filters ──────────────────────────────────────────────────────────
def apply_filters(matches: pd.DataFrame, deliveries: pd.DataFrame,
                  season_range: tuple, selected_teams: list):
    # Season filter
    matches = matches[
        matches["season"].between(season_range[0], season_range[1])
    ]
    # Team filter
    if selected_teams:
        matches = matches[
            matches["team1"].isin(selected_teams) |
            matches["team2"].isin(selected_teams)
        ]
    # Filter deliveries to remaining match IDs
    valid_ids = matches["id"].tolist()
    deliveries = deliveries[deliveries["match_id"].isin(valid_ids)]
    return matches, deliveries


matches, deliveries = apply_filters(
    matches_raw, deliveries_raw, season_range, selected_teams
)

# Guard: need data
if matches.empty:
    st.warning("⚠️ No matches match the selected filters. Please widen the season range or team selection.")
    st.stop()

# ── Route pages ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    overview.render(matches, deliveries)

elif page == "🏏 Batting":
    batting.render(matches, deliveries)

elif page == "🎯 Bowling":
    bowling.render(matches, deliveries)

elif page == "🏆 Teams":
    teams.render(matches, deliveries)

elif page == "🤖 Predictor":
    predictor.render(matches, deliveries)
