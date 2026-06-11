import yfinance as yf
import numpy as np


class PortfolioAgent:

    def __init__(self):

        print("Loading Portfolio Agent...")

        print("Portfolio Agent Ready")


    def analyze_portfolio(

        self,

        tickers

    ):

        portfolio_data = {}

        volatilities = []

        total_return = 0

        # --------------------------------
        # ANALYZE EACH STOCK
        # --------------------------------
        for ticker in tickers:

            df = yf.download(

                ticker,

                period="6mo",

                auto_adjust=True

            )

            # Fix MultiIndex
            if hasattr(df.columns, "levels"):

                df.columns = (
                    df.columns.get_level_values(0)
                )

            returns = (
                df["Close"]
                .pct_change()
                .dropna()
            )

            volatility = (
                np.std(returns) * 100
            )

            start_price = (
                df["Close"].iloc[0]
            )

            end_price = (
                df["Close"].iloc[-1]
            )

            performance = (

                (
                    end_price
                    - start_price
                )

                / start_price

            ) * 100

            portfolio_data[ticker] = {

                "performance":
                round(float(performance), 2),

                "volatility":
                round(float(volatility), 2)

            }

            volatilities.append(
                volatility
            )

            total_return += performance

        # --------------------------------
        # OVERALL METRICS
        # --------------------------------
        avg_return = (
            total_return / len(tickers)
        )

        avg_volatility = (
            sum(volatilities)
            / len(volatilities)
        )

        # --------------------------------
        # RISK LEVEL
        # --------------------------------
        if avg_volatility < 2:

            portfolio_risk = "LOW"

        elif avg_volatility < 4:

            portfolio_risk = "MEDIUM"

        else:

            portfolio_risk = "HIGH"

        # --------------------------------
        # DIVERSIFICATION
        # --------------------------------
        if len(tickers) < 3:

            diversification = "POOR"

        elif len(tickers) < 6:

            diversification = "MODERATE"

        else:

            diversification = "GOOD"

        # --------------------------------
        # RETURN
        # --------------------------------
        return {

            "portfolio":

            portfolio_data,

            "average_return":
            round(float(avg_return), 2),

            "average_volatility":
            round(float(avg_volatility), 2),

            "portfolio_risk":
            portfolio_risk,

            "diversification":
            diversification

        }