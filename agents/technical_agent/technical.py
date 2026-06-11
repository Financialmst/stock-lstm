import yfinance as yf
import pandas as pd


class TechnicalAgent:

    def __init__(self):

        print("Loading Technical Agent...")

        print("Technical Agent Ready")


    def analyze_technical(self, ticker):

        # -----------------------------
        # DOWNLOAD STOCK DATA
        # -----------------------------
        df = yf.download(

            ticker,

            period="6mo",

            auto_adjust=True

        )

        # -----------------------------
        # FIX MULTIINDEX
        # -----------------------------
        if hasattr(df.columns, "levels"):

            df.columns = (
                df.columns.get_level_values(0)
            )

        # -----------------------------
        # RSI
        # -----------------------------
        delta = df["Close"].diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()

        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (
            100 / (1 + rs)
        )

        latest_rsi = float(
            rsi.iloc[-1]
        )

        # -----------------------------
        # MOVING AVERAGES
        # -----------------------------
        sma_20 = (
            df["Close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        sma_50 = (
            df["Close"]
            .rolling(50)
            .mean()
            .iloc[-1]
        )

        # -----------------------------
        # MACD
        # -----------------------------
        ema_12 = (
            df["Close"]
            .ewm(span=12)
            .mean()
        )

        ema_26 = (
            df["Close"]
            .ewm(span=26)
            .mean()
        )

        macd = ema_12 - ema_26

        signal = (
            macd
            .ewm(span=9)
            .mean()
        )

        latest_macd = float(
            macd.iloc[-1]
        )

        latest_signal = float(
            signal.iloc[-1]
        )

        # -----------------------------
        # TREND DETECTION
        # -----------------------------
        if sma_20 > sma_50:

            trend = "BULLISH"

        else:

            trend = "BEARISH"

        # -----------------------------
        # RSI SIGNAL
        # -----------------------------
        if latest_rsi > 70:

            rsi_signal = "OVERBOUGHT"

        elif latest_rsi < 30:

            rsi_signal = "OVERSOLD"

        else:

            rsi_signal = "NEUTRAL"

        # -----------------------------
        # MACD SIGNAL
        # -----------------------------
        if latest_macd > latest_signal:

            macd_signal = "BULLISH"

        else:

            macd_signal = "BEARISH"

        # -----------------------------
        # RETURN
        # -----------------------------
        return {

            "ticker": ticker,

            "rsi":
            round(latest_rsi, 2),

            "rsi_signal":
            rsi_signal,

            "macd":
            round(latest_macd, 2),

            "macd_signal":
            macd_signal,

            "sma_20":
            round(float(sma_20), 2),

            "sma_50":
            round(float(sma_50), 2),

            "technical_trend":
            trend

        }