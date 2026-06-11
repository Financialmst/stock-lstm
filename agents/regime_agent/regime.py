import yfinance as yf
import numpy as np


class RegimeAgent:

    def __init__(self):

        print("Loading Regime Agent...")

        print("Regime Agent Ready")


    def detect_regime(self, ticker):

        # --------------------------------
        # DOWNLOAD DATA
        # --------------------------------
        df = yf.download(

            ticker,

            period="1y",

            auto_adjust=True

        )

        # --------------------------------
        # FIX MULTIINDEX
        # --------------------------------
        if hasattr(df.columns, "levels"):

            df.columns = (
                df.columns.get_level_values(0)
            )

        # --------------------------------
        # MOVING AVERAGES
        # --------------------------------
        sma_50 = (
            df["Close"]
            .rolling(50)
            .mean()
        )

        sma_200 = (
            df["Close"]
            .rolling(200)
            .mean()
        )

        latest_sma50 = sma_50.iloc[-1]

        latest_sma200 = sma_200.iloc[-1]

        # --------------------------------
        # VOLATILITY
        # --------------------------------
        returns = (
            df["Close"]
            .pct_change()
            .dropna()
        )

        volatility = (
            np.std(returns) * 100
        )

        # --------------------------------
        # REGIME DETECTION
        # --------------------------------

        # Bullish Trend
        if (

            latest_sma50 > latest_sma200

            and

            volatility < 2

        ):

            regime = "TRENDING_BULL"


        # Bearish Trend
        elif (

            latest_sma50 < latest_sma200

            and

            volatility < 2

        ):

            regime = "TRENDING_BEAR"


        # Panic / High Volatility
        elif volatility > 4:

            regime = "PANIC_VOLATILE"


        # Sideways
        else:

            regime = "SIDEWAYS"

        # --------------------------------
        # RETURN
        # --------------------------------
        return {

            "ticker": ticker,

            "market_regime": regime,

            "volatility":
            round(float(volatility), 2),

            "sma_50":
            round(float(latest_sma50), 2),

            "sma_200":
            round(float(latest_sma200), 2)

        }