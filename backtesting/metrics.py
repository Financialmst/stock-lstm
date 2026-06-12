# backtest/metrics.py
# Performance metrics computed from a trade log and equity curve

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    signal: str          # BUY / SELL / HOLD
    direction: str       # LONG / SHORT
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0

    def __post_init__(self):
        if self.direction == "LONG":
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price * 100
        self.pnl_abs = self.pnl_pct / 100 * self.entry_price


@dataclass
class PerformanceMetrics:
    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0     # buy-and-hold
    alpha_pct: float = 0.0                # strategy - benchmark

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility_annualized: float = 0.0
    beta: float = 0.0
    calmar_ratio: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0            # gross profit / gross loss
    avg_trade_duration_days: float = 0.0

    # Signal accuracy
    signal_accuracy_pct: float = 0.0      # % of BUY signals that were profitable
    prediction_mae: float = 0.0           # mean absolute error of LSTM price predictions
    prediction_direction_accuracy: float = 0.0  # % of times LSTM got direction right

    # Period info
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0


def compute_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    predictions: Optional[List[dict]] = None,
    risk_free_rate: float = 0.065,   # 6.5% for India (RBI repo rate approx)
) -> PerformanceMetrics:
    m = PerformanceMetrics()

    if not trades or equity_curve.empty:
        return m

    m.start_date = str(equity_curve.index[0].date())
    m.end_date = str(equity_curve.index[-1].date())
    m.trading_days = len(equity_curve)
    years = m.trading_days / 252

    # --- Return metrics ---
    m.total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    m.annualized_return_pct = ((1 + m.total_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    m.benchmark_return_pct = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    m.alpha_pct = m.total_return_pct - m.benchmark_return_pct

    # --- Risk metrics ---
    daily_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_curve.pct_change().dropna()

    m.volatility_annualized = daily_returns.std() * np.sqrt(252) * 100

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    m.sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < 0]
    m.sortino_ratio = (excess_returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    m.max_drawdown_pct = drawdown.min()

    m.calmar_ratio = m.annualized_return_pct / abs(m.max_drawdown_pct) if m.max_drawdown_pct != 0 else 0

    # Beta vs benchmark
    aligned = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) > 10 and aligned.iloc[:, 1].var() > 0:
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        m.beta = cov[0][1] / aligned.iloc[:, 1].var()

    # --- Trade metrics ---
    m.total_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate_pct = len(wins) / len(trades) * 100 if trades else 0
    m.avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
    m.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0

    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # --- Prediction accuracy ---
    if predictions:
        correct_direction = sum(
            1 for p in predictions
            if np.sign(p["predicted_price"] - p["actual_open"]) == np.sign(p["actual_close"] - p["actual_open"])
        )
        m.prediction_direction_accuracy = correct_direction / len(predictions) * 100

        mae_values = [abs(p["predicted_price"] - p["actual_close"]) for p in predictions]
        m.prediction_mae = np.mean(mae_values)

        buy_signals = [t for t in trades if t.signal == "BUY"]
        m.signal_accuracy_pct = len([t for t in buy_signals if t.pnl_pct > 0]) / len(buy_signals) * 100 if buy_signals else 0

    return m
