# backtest/signal_simulator.py
# Replays the full agent pipeline over historical OHLCV data
# without look-ahead bias.
from agents.prediction_agent.indicators import (
    add_technical_indicators
)
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from backtesting.metrics import Trade


@dataclass
class SignalRecord:
    date: str
    price: float
    predicted_price: float
    technical_trend: str
    rsi: float
    macd: float
    signal_line: float
    decision: str          # BUY / SELL / HOLD
    confidence: float
    regime: str


def compute_technical_indicators(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename Close to Close_yfin to match training data column name
    if "Close" in df.columns and "Close_yfin" not in df.columns:
        df["Close_yfin"] = df["Close"]

    df = add_technical_indicators(df)
    return df


def _technical_decision(row: pd.Series) -> Tuple[str, float]:
    """Simple rule-based technical decision. Replace with your TechnicalAgent logic."""
    score = 0
    signals = 0

    if not np.isnan(row.get("RSI_14", np.nan)):
        signals += 1
        if row["RSI_14"] < 35:
            score += 1
        elif row["RSI_14"] > 65:
            score -= 1

    if not np.isnan(row.get("MACD", np.nan)):
        signals += 1
        if row["MACD"] > row["Signal_Line"]:
            score += 1
        else:
            score -= 1

    if not np.isnan(row.get("SMA_20", np.nan)):
        signals += 1
        if row["Close"] > row["SMA_20"]:
            score += 1
        elif row["Close"] < row["SMA_20"]:
            score -= 1

    if signals == 0:
        return "HOLD", 0.5

    norm = score / signals   # -1 to +1
    if norm > 0.3:
        return "BUY", 0.5 + norm * 0.4
    elif norm < -0.3:
        return "SELL", 0.5 + abs(norm) * 0.4
    else:
        return "HOLD", 0.5


def _detect_regime(row: pd.Series) -> str:
    """Replicates your RegimeAgent logic."""
    if np.isnan(row.get("SMA_10", np.nan)) or np.isnan(row.get("SMA_20", np.nan)):
        return "UNKNOWN"

    trending_up = (
    row["Close"] >
    row["SMA_10"] >
    row["SMA_20"]
)
    high_vol = row.get("Volatility", 0) > 0.25

    if trending_up and not high_vol:
        return "TRENDING_BULL"
    elif trending_up and high_vol:
        return "VOLATILE_BULL"
    elif not trending_up and not high_vol:
        return "TRENDING_BEAR"
    elif not trending_up and high_vol:
        return "VOLATILE_BEAR"
    else:
        return "SIDEWAYS"


def simulate_signals(
    df: pd.DataFrame,
    prediction_fn=None,        # callable: df_window → predicted_price (your LSTM)
    sequence_length: int = 60,
    holding_period: int = 5,   # days to hold a position before exiting
) -> Tuple[List[SignalRecord], List[Trade], List[dict]]:
    """
    Replays the full signal pipeline over historical data.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        prediction_fn: your LSTM model callable. If None, uses a naive baseline.
        sequence_length: LSTM lookback window
        holding_period: days to hold each trade

    Returns:
        signals: list of SignalRecord per bar
        trades: list of completed Trade objects
        predictions: list of {date, predicted_price, actual_open, actual_close}
    """
    df = df.copy()
    df = compute_technical_indicators(df)
    df = df.dropna(subset=["SMA_20"])   # need at least 50 bars of history

    signals: List[SignalRecord] = []
    trades: List[Trade] = []
    predictions_log: List[dict] = []

    i = sequence_length
    while i < len(df) - holding_period:
        row = df.iloc[i]
        window = df.iloc[i - sequence_length: i]
        current_date = str(df.index[i].date())
        current_price = row["Close"]

        # --- LSTM Prediction ---
        if prediction_fn is not None:
            try:
                predicted_price = prediction_fn(window)
            except Exception as e:
                print("Prediction Error:", e)
                predicted_price = current_price  # fallback
        else:
            # Naive baseline: predict next close = current close (random walk)
            predicted_price = current_price * (1 + np.random.normal(0, 0.005))

        predictions_log.append({
            "date": current_date,
            "predicted_price": predicted_price,
            "actual_open": row["Open"],
            "actual_close": df.iloc[i + holding_period]["Close"],
        })

        # --- Technical signal ---
        tech_decision, tech_confidence = _technical_decision(row)

        # --- Regime ---
        regime = _detect_regime(row)

        # --- Combine: final decision ---
        pred_direction = "BUY" if predicted_price > current_price * 1.005 else (
            "SELL" if predicted_price < current_price * 0.995 else "HOLD"
        )

        # Simple fusion: agree → use signal, disagree → HOLD
        if pred_direction == tech_decision and pred_direction != "HOLD":
            final_decision = pred_direction
            confidence = min(
    tech_confidence * 1.1,
    1.0
)
        elif pred_direction == "HOLD" or tech_decision == "HOLD":
            final_decision = pred_direction if pred_direction != "HOLD" else tech_decision
            confidence = tech_confidence * 0.85
        else:
            # LSTM says BUY, technical says SELL or vice versa — no trade
            final_decision = "HOLD"
            confidence = 0.5

        # Regime filter: don't go long in TRENDING_BEAR or VOLATILE_BEAR
        if final_decision == "BUY" and regime in ("TRENDING_BEAR", "VOLATILE_BEAR"):
            final_decision = "HOLD"
            confidence *= 0.7

        confidence = min(confidence, 1.0)

        signals.append(SignalRecord(
            date=current_date,
            price=current_price,
            predicted_price=predicted_price,
            technical_trend=tech_decision,
            rsi=row.get("RSI_14", 0),
            macd=row.get("MACD", 0),
            signal_line=row.get("Signal_Line", 0),
            decision=final_decision,
            confidence=confidence,
            regime=regime,
        ))

        # --- Simulate trade ---
        if final_decision in ("BUY", "SELL"):
            exit_idx = i + holding_period
            exit_price = df.iloc[exit_idx]["Close"]
            exit_date = str(df.index[exit_idx].date())

            trades.append(Trade(
                ticker=df.get("ticker", ["UNKNOWN"])[0] if "ticker" in df.columns else "UNKNOWN",
                entry_date=current_date,
                exit_date=exit_date,
                entry_price=current_price,
                exit_price=exit_price,
                signal=final_decision,
                direction="LONG" if final_decision == "BUY" else "SHORT",
            ))

            i += holding_period   # skip forward — we're in a trade
        else:
            i += 1

    return signals, trades, predictions_log


def build_equity_curve(
    trades: List[Trade],
    df: pd.DataFrame,
    initial_capital: float = 100_000,
) -> pd.Series:
    """Build a daily equity curve from the trade log."""
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = initial_capital
    capital = initial_capital

    trade_map = {}
    for t in trades:
        trade_map[t.exit_date] = t

    for i in range(1, len(df)):
        date_str = str(df.index[i].date())
        if date_str in trade_map:
            t = trade_map[date_str]
            capital *= (1 + t.pnl_pct / 100)
        equity.iloc[i] = capital

    return equity.ffill()
