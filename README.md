# FinIntel AI — AI Stock Intelligence Platform

> Not a stock predictor. An AI Financial Intelligence Platform combining **ML prediction · sentiment analysis · market regime detection · probabilistic forecasting · LLM reasoning** for India, US, Crypto, and Global Markets.

## What It Does

FinIntel AI analyses any stock ticker through 10 specialised AI agents, aggregates their signals into a calibrated probability, and delivers a plain-English explanation a retail investor can act on.

## Agent Architecture

| Agent | What It Does | Status |
|---|---|---|
| PredictionAgent | Pure Transformer direction classifier — 56.5% accuracy, +4.4% edge | ✅ Live |
| SentimentAgent | FinBERT on live RSS news + Reddit (5 subreddits) | ✅ Live |
| RiskAgent | Volatility → LOW / MEDIUM / HIGH | ✅ Live |
| TechnicalAgent | RSI, MACD, SMA → trend + signal labels | ✅ Live |
| RegimeAgent | TRENDING_BULL/BEAR, VOLATILE_BULL/BEAR, SIDEWAYS | ✅ Live |
| MultiTimeframeAgent | 1M / 3M / 6M / 1Y human-readable labels | ✅ Live |
| ProbabilityAgent | Aggregates all signals → Bullish %, Bearish %, Confidence | ✅ Live |
| DecisionAgent | BUY / SELL / HOLD | ✅ Live |
| LLMReasoningAgent | Plain-English explanation of all signals | ✅ Live |
| PortfolioAgent | Diversification, sector exposure, correlation | 🔲 Planned |

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

## API
POST /analyze/<ticker> → returns task_id instantly
GET /task/<task_id> → returns progress or final result

**Start Analysis**
```json
POST /analyze/AAPL
{
  "task_id": "abc-123",
  "ticker": "AAPL",
  "status": "PENDING"
}
```

**Poll for Result**
```json
GET /task/abc-123
{
  "status": "SUCCESS",
  "result": {
    "ticker": "AAPL",
    "current_price": 213.45,
    "predicted_price": 214.20,
    "direction": "UP",
    "up_probability": 0.68,
    "sentiment": "positive",
    "decision": "BUY",
    "risk_level": "MEDIUM",
    "bullish_probability": 72,
    "bearish_probability": 28,
    "confidence": 81,
    "market_regime": "TRENDING_BULL",
    "timeframe_short": "Bullish",
    "timeframe_medium": "Neutral",
    "timeframe_long": "Bullish",
    "ai_reasoning": "AAPL is showing strong upward momentum...",
    "fetched_at": "2026-06-14T10:30:00Z"
  }
}
```

## Market Coverage

India · US · Crypto · Indices · Commodities — 22 global tickers

## Roadmap

- [x] 10-agent signal pipeline
- [x] Pure Transformer direction model
- [x] Walk-forward backtesting engine
- [x] Async Flask + Celery + Redis
- [x] SEC EDGAR insider data integration
- [x] FINRA short interest data integration
- [ ] Phase 1 — AI Summary, Warnings, Confidence engine
- [ ] Phase 2 — Investment style detection
- [ ] Phase 3 — Portfolio Intelligence
- [ ] Phase 4 — News Intelligence (RAG-based)
- [ ] Phase 5 — AI Chat Mode

## Disclaimer

For educational and research purposes only. Not financial advice.

## Author

ECE student at NIT Silchar (CGPA: 8.57). Built as a portfolio project demonstrating applied ML in financial markets.
