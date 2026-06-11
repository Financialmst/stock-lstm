import yfinance as yf
import numpy as np


class RiskAgent:

    def __init__(self):

        print("Loading Risk Agent...")

        print("Risk Agent Ready")

    def calculate_risk(self, ticker):

        # -----------------------------
        # DOWNLOAD DATA
        # -----------------------------
        df = yf.download(

            ticker,

            period="6mo",

            auto_adjust=True

        )
        if hasattr(df.columns, "levels"):
          df.columns = df.columns.get_level_values(0)

        # -----------------------------
        # DAILY RETURNS
        # -----------------------------
        df["Returns"] = df["Close"].pct_change()

        # -----------------------------
        # VOLATILITY
        # -----------------------------
        volatility = np.std(
            df["Returns"].dropna()
        )

        volatility_percent = (
            volatility * 100
        )

        # -----------------------------
        # RISK CLASSIFICATION
        # -----------------------------
        if volatility_percent < 2:

            risk = "LOW"

        elif volatility_percent < 4:

            risk = "MEDIUM"

        else:

            risk = "HIGH"

        # -----------------------------
        # RETURN RESULT
        # -----------------------------
        return {

            "ticker": ticker,

            "volatility_percent": round(
                float(volatility_percent),
                2
            ),

            "risk_level": risk

        }