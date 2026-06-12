# backtest/validators.py
# Walk-forward validation + simple train/test split

import pandas as pd
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass, field

from backtesting.metrics import (
    PerformanceMetrics,
    compute_metrics
)

from backtesting.signal_simulator import (
    simulate_signals,
    build_equity_curve
)


@dataclass
class ValidationFold:
    """Result of one walk-forward fold."""
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: PerformanceMetrics
    n_trades: int


@dataclass
class WalkForwardResult:
    ticker: str
    folds: List[ValidationFold] = field(default_factory=list)
    combined_metrics: Optional[PerformanceMetrics] = None

    # Aggregated across folds
    avg_sharpe: float = 0.0
    avg_win_rate: float = 0.0
    avg_total_return: float = 0.0
    avg_max_drawdown: float = 0.0
    avg_prediction_accuracy: float = 0.0
    consistency_score: float = 0.0   # % of folds that were profitable


@dataclass
class TrainTestResult:
    ticker: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    overfit_score: float = 0.0   # train_return / test_return — >2 suggests overfitting


class TrainTestValidator:
    """
    Simple train/test split.
    Fast to run, useful for initial sanity-checking.
    """
    def __init__(self, train_ratio: float = 0.8):
        self.train_ratio = train_ratio

    def validate(
        self,
        ticker: str,
        df: pd.DataFrame,
        prediction_fn: Optional[Callable] = None,
        holding_period: int = 5,
    ) -> TrainTestResult:
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Train set
        train_signals, train_trades, train_preds = simulate_signals(
            train_df, prediction_fn=prediction_fn, holding_period=holding_period
        )
        train_equity = build_equity_curve(train_trades, train_df)
        train_benchmark = train_df["Close"] / train_df["Close"].iloc[0] * 100_000
        train_metrics = compute_metrics(train_trades, train_equity, train_benchmark, train_preds)

        # Test set
        test_signals, test_trades, test_preds = simulate_signals(
            test_df, prediction_fn=prediction_fn, holding_period=holding_period
        )
        test_equity = build_equity_curve(test_trades, test_df)
        test_benchmark = test_df["Close"] / test_df["Close"].iloc[0] * 100_000
        test_metrics = compute_metrics(test_trades, test_equity, test_benchmark, test_preds)

        # Overfit score
        if test_metrics.total_return_pct != 0:
            overfit_score = train_metrics.total_return_pct / test_metrics.total_return_pct
        else:
            overfit_score = float("inf")

        return TrainTestResult(
            ticker=ticker,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            test_start=str(test_df.index[0].date()),
            test_end=str(test_df.index[-1].date()),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            overfit_score=overfit_score,
        )


class WalkForwardValidator:
    """
    Walk-forward validation — the gold standard for time-series strategy testing.

    How it works:
        [ TRAIN ][ TEST ]
             [ TRAIN ][ TEST ]
                  [ TRAIN ][ TEST ]

    Each fold trains only on past data and tests on unseen future data.
    This is the only honest way to test a time-series model.
    """
    def __init__(
        self,
        n_folds: int = 5,
        train_size_pct: float = 0.6,   # each training window = 60% of total data
        test_size_pct: float = 0.1,    # each test window = 10% of total data
        anchored: bool = False,        # True = expanding window, False = rolling window
    ):
        self.n_folds = n_folds
        self.train_size_pct = train_size_pct
        self.test_size_pct = test_size_pct
        self.anchored = anchored

    def _build_folds(self, df: pd.DataFrame):
        n = len(df)
        test_size = int(n * self.test_size_pct)
        train_size = int(n * self.train_size_pct)
        folds = []

        for i in range(self.n_folds):
            test_end = n - (self.n_folds - 1 - i) * test_size
            test_start = test_end - test_size
            if self.anchored:
                train_start = 0
            else:
                train_start = max(0, test_start - train_size)
            train_end = test_start

            if train_end - train_start < 100:   # need at least 100 bars
                continue

            folds.append({
                "train": df.iloc[train_start:train_end],
                "test": df.iloc[test_start:test_end],
            })

        return folds

    def validate(
        self,
        ticker: str,
        df: pd.DataFrame,
        prediction_fn: Optional[Callable] = None,
        holding_period: int = 5,
        train_fn: Optional[Callable] = None,  # callable(train_df) → fitted model
    ) -> WalkForwardResult:
        """
        Args:
            ticker: symbol string
            df: full historical OHLCV DataFrame
            prediction_fn: your LSTM predict callable OR None (naive baseline used)
            holding_period: days to hold each trade
            train_fn: optional callable to retrain your LSTM on each fold's training data
                      signature: train_fn(train_df) → prediction_fn
        """
        folds_data = self._build_folds(df)
        result = WalkForwardResult(ticker=ticker)

        all_trades = []
        all_equity_points = []
        all_preds = []

        for i, fold in enumerate(folds_data):
            train_df = fold["train"]
            test_df = fold["test"]

            # Optionally retrain model on each fold's training window
            active_pred_fn = prediction_fn
            if train_fn is not None:
                try:
                    active_pred_fn = train_fn(train_df)
                except Exception as e:
                    print(f"[WalkForward] Fold {i+1} retrain failed: {e}. Using previous model.")

            signals, trades, preds = simulate_signals(
                test_df,
                prediction_fn=active_pred_fn,
                holding_period=holding_period,
            )

            equity = build_equity_curve(trades, test_df)
            benchmark = test_df["Close"] / test_df["Close"].iloc[0] * 100_000
            metrics = compute_metrics(trades, equity, benchmark, preds)

            fold_result = ValidationFold(
                fold_number=i + 1,
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                metrics=metrics,
                n_trades=len(trades),
            )

            result.folds.append(fold_result)
            all_trades.extend(trades)
            all_preds.extend(preds)

            print(f"  Fold {i+1}: Return={metrics.total_return_pct:.1f}% | "
                  f"Sharpe={metrics.sharpe_ratio:.2f} | "
                  f"WinRate={metrics.win_rate_pct:.1f}% | "
                  f"Trades={len(trades)}")

        # Aggregate stats across folds
        if result.folds:
            result.avg_sharpe = np.mean([f.metrics.sharpe_ratio for f in result.folds])
            result.avg_win_rate = np.mean([f.metrics.win_rate_pct for f in result.folds])
            result.avg_total_return = np.mean([f.metrics.total_return_pct for f in result.folds])
            result.avg_max_drawdown = np.mean([f.metrics.max_drawdown_pct for f in result.folds])
            result.avg_prediction_accuracy = np.mean([
                f.metrics.prediction_direction_accuracy for f in result.folds
                if f.metrics.prediction_direction_accuracy > 0
            ] or [0])
            result.consistency_score = sum(
                1 for f in result.folds if f.metrics.total_return_pct > 0
            ) / len(result.folds) * 100

        return result
