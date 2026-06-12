# backtest/run_backtest.py
# Entry point — run this directly to backtest any ticker
#
# Usage:
#   python run_backtest.py --ticker RELIANCE.NS --mode both
#   python run_backtest.py --ticker AAPL --mode walkforward --folds 5
#   python run_backtest.py --ticker BTC-USD --mode split

import argparse
import yfinance as yf
import pandas as pd

# Add parent dir to path so agents can be imported


from backtesting.validators import (
    WalkForwardValidator,
    TrainTestValidator
)

from backtesting.report import (
    generate_report
)


def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    print(f"Fetching {period} of data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")
    print(f"  Got {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


from agents.prediction_agent.predictor import PredictionAgent

def get_prediction_fn(ticker: str):

    print("Loading Prediction Agent...")

    agent = PredictionAgent()

    def prediction_fn(window):

        try:

            prediction = agent.predict_window(
                window
            )

            return prediction

        except Exception as e:

            print("Prediction Error:", e)

            return float(
                window["Close"].iloc[-1]
            )

    return prediction_fn


def get_train_fn():
    """
    Optional: return a function that retrains your LSTM on each fold's training data.
    Signature: train_fn(train_df: pd.DataFrame) -> prediction_fn
    
    If you don't retrain per fold, set this to None in main().
    """
    # ── Wire your retraining here ─────────────────────────────────────────────
    # from agents.prediction import PredictionAgent
    # def train_fn(train_df):
    #     agent = PredictionAgent()
    #     agent.train(train_df)
    #     return lambda window: agent.predict(window)
    # return train_fn
    # ─────────────────────────────────────────────────────────────────────────
    return None   # set to None = use same model across all folds


def main():
    parser = argparse.ArgumentParser(description="AI Stock Intelligence Platform — Backtesting Engine")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. RELIANCE.NS, AAPL, BTC-USD")
    parser.add_argument("--mode", type=str, default="both", choices=["split", "walkforward", "both"])
    parser.add_argument("--period", type=str, default="5y", help="yfinance period: 1y, 2y, 5y, 10y")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--hold", type=int, default=5, help="Holding period in trading days")
    parser.add_argument("--output", type=str, default=None, help="Save report to file path (.txt)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Backtesting: {args.ticker.upper()}")
    print(f"  Mode: {args.mode} | Period: {args.period} | Hold: {args.hold}d")
    print(f"{'='*60}\n")

    # 1. Fetch data
    df = fetch_data(args.ticker, period=args.period)

    # 2. Get prediction function
    prediction_fn = get_prediction_fn(args.ticker)
    train_fn = get_train_fn()

    results = []

    # 3. Train/Test Split
    if args.mode in ("split", "both"):
        print("\n── Train/Test Split Validation ──────────────────────────")
        validator = TrainTestValidator(train_ratio=0.8)
        result = validator.validate(
            ticker=args.ticker,
            df=df,
            prediction_fn=prediction_fn,
            holding_period=args.hold,
        )
        out = args.output.replace(".txt", "_split.txt") if args.output else None
        generate_report(result, output_path=out)
        results.append(result)

    # 4. Walk-Forward Validation
    if args.mode in ("walkforward", "both"):
        print("\n── Walk-Forward Validation ──────────────────────────────")
        validator = WalkForwardValidator(n_folds=args.folds)
        result = validator.validate(
            ticker=args.ticker,
            df=df,
            prediction_fn=prediction_fn,
            holding_period=args.hold,
            train_fn=train_fn,
        )
        out = args.output.replace(".txt", "_wf.txt") if args.output else None
        generate_report(result, output_path=out)
        results.append(result)

    print("Done.\n")
    return results


if __name__ == "__main__":
    main()
