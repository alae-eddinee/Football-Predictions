# Alae's Pitch — ML Football Predictions

A personal fun project by **Alae-Eddine Dahane**.

ML-powered football match outcome predictor and betting market inefficiency detector. Combines an XGBoost + LightGBM ensemble with Elo ratings, team form, and head-to-head history to forecast results across Europe's top leagues.

---

## Features

- **Live fixtures** from Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Championship, Eredivisie, and Primeira Liga
- **ML predictions** — XGBoost + LightGBM ensemble with calibrated probabilities
- **Elo rating system** — dynamic team strength tracking across seasons
- **Value bet detection** — Kelly criterion stake sizing when model edge exceeds bookmaker odds
- **One-click training** — full pipeline (ingest → train → serve) from the UI
- **Dark UI** — sleek, modern interface with real-time updates

---

## Tech Stack

| Layer | Tech |
|---|---|
| ML Models | XGBoost 2.0, LightGBM 4.3, scikit-learn |
| Feature Engineering | Elo ratings, rolling form, H2H, rest days |
| Backend API | FastAPI + Uvicorn |
| Data Sources | football-data.co.uk, API-Football, ESPN, Understat |
| Frontend | Vanilla JS SPA (no framework) |
| Deployment | Vercel (Python runtime) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- API-Football key (free plan works) → [api-football.com](https://www.api-football.com)
- football-data.org key (free) → [football-data.org](https://www.football-data.org)

### Install

```bash
git clone https://github.com/alae-eddinee/Football-Predictions.git
cd Football-Predictions
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your API keys:
# API_FOOTBALL_KEY=your_key_here
# FOOTBALL_DATA_KEY=your_key_here
```

### Run

```bash
# Start the web server (serves UI + API)
python app/server.py
```

Open `http://localhost:8000` — on first load, click **Predict** on any match to trigger the training pipeline automatically.

### CLI Pipeline

```bash
python main.py ingest      # Download historical match data
python main.py train       # Train the ML ensemble
python main.py backtest    # Simulate betting strategy
python main.py run         # Full pipeline (ingest → train → backtest)
```

---

## Project Structure

```
Football-Predictions/
├── app/
│   ├── server.py          # FastAPI backend
│   ├── predictor.py       # Live prediction engine
│   └── static/
│       └── index.html     # Frontend SPA
├── src/
│   ├── data/              # API clients + ingestion
│   ├── features/          # Feature engineering (Elo, form, H2H)
│   ├── models/            # Ensemble training + evaluation
│   ├── betting/           # Value detection + Kelly criterion
│   └── visualization/     # Plots + reports
├── configs/
│   └── config.yaml        # League, model, and betting config
├── api/
│   └── index.py           # Vercel serverless entry point
├── main.py                # CLI entry point
├── vercel.json            # Vercel deployment config
└── requirements.txt
```

---

## Supported Leagues

| League | Country | Code |
|---|---|---|
| Premier League | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | E0 |
| Championship | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | E1 |
| La Liga | 🇪🇸 Spain | SP1 |
| Bundesliga | 🇩🇪 Germany | D1 |
| Serie A | 🇮🇹 Italy | I1 |
| Ligue 1 | 🇫🇷 France | F1 |
| Eredivisie | 🇳🇱 Netherlands | N1 |
| Primeira Liga | 🇵🇹 Portugal | P1 |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Model readiness + history size |
| `GET` | `/api/fixtures?date=YYYY-MM-DD` | Fixtures for a date |
| `POST` | `/api/predict` | Predict a match outcome |
| `GET` | `/api/leagues` | Supported leagues |
| `GET` | `/api/team/{name}` | Team Elo, form, last match |
| `POST` | `/api/pipeline/run` | Trigger ingest + train |
| `GET` | `/api/pipeline/status` | Pipeline progress |

---

## Deploy to Vercel

> **Note**: The ML model (trained `.pkl` file) and match data are not included in the repo due to size. For full predictions, train locally first and consider storing the model in cloud storage.

```bash
npm i -g vercel
vercel
```

The app will serve fixtures without predictions until a trained model is available. To deploy with predictions, upload `models/ensemble.pkl` to an accessible location and adjust `predictor.py` to load from that URL.

---

## How It Works

1. **Data ingestion** — Downloads 5+ seasons of match results from football-data.co.uk
2. **Feature engineering** — Computes Elo ratings (updated after each match), rolling form (last 5 games), H2H record, rest days, and league standings
3. **Model training** — Trains XGBoost and LightGBM classifiers (3-class: Home/Draw/Away), calibrates probabilities with isotonic regression, builds weighted ensemble
4. **Prediction** — For upcoming fixtures, replays Elo history to the match date, computes current form, and runs both models
5. **Value detection** — Compares model probabilities against bookmaker implied probabilities; flags edges above threshold with Kelly criterion sizing

---

## Disclaimer

This project is for educational and entertainment purposes only. It is not financial or betting advice. Past predictive performance does not guarantee future results.

---

## License

MIT © 2026 [Alae-Eddine Dahane](https://github.com/alae-eddinee)
