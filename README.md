# FinIntel AI — AI Stock Intelligence Platform

> Not a stock predictor. An AI Financial Intelligence Platform combining **ML prediction · sentiment analysis · market regime detection · probabilistic forecasting · LLM reasoning** for India, US, Crypto, and Global Markets.

## What It Does

FinIntel AI analyses any stock ticker through 10 specialised AI agents, aggregates their signals into a calibrated probability, and delivers a plain-English explanation a retail investor can act on.

## Agent Architecture

| Agent | What It Does | Status |
|---|---|---|
| PredictioncAgent | Pure Transformer direction classifier — 56.5% accuracy, +4.4% edge | ✅ Live |
| Sentiment Agent | FinBERT on live RSS news + Reddit (5 subreddits) | ✅ Live |
| Risk Agent | Volatility → LOW / MEDIUM / HIGH | ✅ Live |
| Technical Agent | RSI, MACD, SMA → trend + signal labels | ✅ Live |
| Regime Agent | TRENDING_BULL/BEAR, VOLATILE_BULL/BEAR, SIDEWAYS | ✅ Live |
| MultiTimeframe Agent | 1M / 3M / 6M / 1Y human-readable labels | ✅ Live |
| Probability Agent | Aggregates all signals → Bullish %, Bearish %, Confidence | ✅ Live |
| Decision Agent | BUY / SELL / HOLD | ✅ Live |
| LLMReasoning Agent | Plain-English explanation of all signals | ✅ Live |
| Portfolio Agent | Diversification, sector exposure, correlation | 🔲 Planned |

## PredictionAgent — ML Details

- Architecture: Pure Transformer (3 blocks, multi-head attention, learnable positional encoding)
- Target: Binary direction — UP >0.5% in 5 trading days
- Features: 18 (price, momentum, volume, returns, volatility)
- Dataset: 28,182 rows · 22 tickers · 5 years
- Test Accuracy: 56.51% | Edge over baseline: +4.42%
- Validated with walk-forward backtesting (5 folds)

### ML Development History

| Attempt | Result | Root Cause |
|---|---|---|
| LSTM regression | 44% direction accuracy | MSE loss teaches "tomorrow ≈ today" |
| LSTM classification | 50% — coin flip | Dead training, collapsed output |
| Pure Transformer (current) | 56.51%, +4.42% edge | Full attention over 60-bar window |

## Tech Stack

| Layer | Technology |
|---|---|
| ML / DL | TensorFlow / Keras |
| NLP | Hugging Face Transformers (FinBERT) |
| Classical ML | scikit-learn |
| Task Queue | Celery + Redis |
| Backend | Flask |
| Frontend | React + Vite |
| External Data | SEC EDGAR Form 4 API, FINRA Reg SHO API |

## Setup

```bash
pip install tensorflow keras scikit-learn pandas numpy yfinance
pip install flask flask-cors celery redis
pip install transformers praw feedparser joblib requests
```

```bash
# Terminal 1
redis-server

# Terminal 2
celery -A celery_app.celery worker --loglevel=info --concurrency=2 --pool=threads

# Terminal 3
python app.py

# Terminal 4
cd frontend && npm install && npm run dev
```

## Train the Model

```bash
python training/train_transformer.py --target direction
```

## Run Backtest

```bash
python -m backtesting.run_backtest --ticker AAPL --mode walkforward --folds 5
```

## API
