# 🏏 IPL Stats Dashboard

A production-quality Indian Premier League analytics dashboard built with **Python**, **Streamlit**, and **Plotly**. Features 5 pages of interactive analysis plus an ML-powered match winner predictor.

---

## 📸 Features

| Page | What You Get |
|------|-------------|
| **Overview** | KPI cards, matches per season, toss impact analysis |
| **Batting** | Top scorers, SR vs Average scatter, player deep-dive, phase analysis, player comparison radar |
| **Bowling** | Top wicket takers, economy vs wickets scatter, bowler deep-dive, dismissal breakdown |
| **Teams** | Win %, head-to-head, season trends, run-rate-by-over heatmap |
| **ML Predictor** | Gradient Boosting match winner prediction + feature importance |

All pages respond to the **global filters** in the sidebar:
- Season range slider
- Multi-team filter
- Quick player search

---

## 🗂 Project Structure

```
ipl_dashboard/
├── app.py                    # Streamlit entry point, routing, filters
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── config.toml           # Dark theme config
├── data/                     # Auto-generated CSVs cached here
│   ├── matches.csv
│   ├── deliveries.csv
│   └── ipl_model.joblib      # Trained ML model (auto-created)
└── src/
    ├── __init__.py
    ├── data_loader.py         # Data loading & synthetic fallback
    ├── analytics.py           # Pure stats functions (no Streamlit)
    ├── charts.py              # Plotly figure factory
    ├── ml_predictor.py        # GradientBoosting model
    └── pages/
        ├── __init__.py
        ├── overview.py
        ├── batting.py
        ├── bowling.py
        ├── teams.py
        └── predictor.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add real IPL data

Download from Kaggle:
- **Dataset**: [IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Alternatively: [IPL Ball-by-Ball 2008–2022](https://www.kaggle.com/datasets/vora1011/ipl-2008-to-2021-all-match-dataset)

Place `matches.csv` and `deliveries.csv` in the `data/` folder.

> **Without real data**, the dashboard auto-generates a synthetic 16-season dataset (2008–2023) that powers all features realistically.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🗄 Dataset Schema

### matches.csv (expected columns)
| Column | Description |
|--------|-------------|
| `id` | Unique match ID |
| `season` | Year (2008–2023) |
| `team1`, `team2` | Competing teams |
| `toss_winner`, `toss_decision` | Toss details |
| `winner` | Winning team |
| `win_by_runs`, `win_by_wickets` | Margin |
| `player_of_match` | MOTM award |
| `venue`, `city` | Location |

### deliveries.csv (expected columns)
| Column | Description |
|--------|-------------|
| `match_id` | FK to matches.id |
| `inning` | 1 or 2 |
| `batting_team`, `bowling_team` | Teams |
| `over`, `ball` | Ball position |
| `batsman`, `bowler` | Players |
| `batsman_runs`, `total_runs` | Runs scored |
| `player_dismissed`, `dismissal_kind` | Wicket info |

---

## 🤖 ML Predictor

The predictor uses a **Gradient Boosting Classifier** (scikit-learn) with features:

1. **Toss won by Team 1** (binary)
2. **Toss decision** (bat/field)
3. **Season** (normalized)
4. **Team 1 historical win rate** (computed from training data, no leakage)
5. **Team 2 historical win rate**
6. **Head-to-head win rate** (Team 1 vs Team 2)
7. **Win rate differential**

Model is trained once on first load, then serialized to `data/ipl_model.joblib`.

**To improve accuracy:**
- Add venue as a feature (home advantage)
- Include squad/player ratings
- Switch to XGBoost or LightGBM
- Add recent form (last N match win rate)

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t ipl-dashboard .

# Run
docker run -p 8501:8501 ipl-dashboard
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Deploy — it will auto-install `requirements.txt`

> Add real data via GitHub or use the synthetic fallback.

---

## 🔧 Configuration

Edit `.streamlit/config.toml` to change theme colors, or set environment variables:

```bash
STREAMLIT_SERVER_PORT=8080 streamlit run app.py
```

---

## 📈 Potential Improvements

- [ ] Add auction price / player value analysis
- [ ] Venue-wise performance breakdown
- [ ] Live match integration (CricAPI or ESPN Cricinfo scrape)
- [ ] Points table simulator
- [ ] XGBoost upgrade for ML model
- [ ] Export reports to PDF
- [ ] PostgreSQL backend for multi-user production use

---

## 📄 License

MIT — free to use and modify.
