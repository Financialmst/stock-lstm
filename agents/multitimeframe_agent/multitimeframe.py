import yfinance as yf


class MultiTimeframeAgent:

    def __init__(self):

        print("Loading Multi-Timeframe Agent...")

        print("Multi-Timeframe Agent Ready")


    def analyze_timeframes(self, ticker):

        timeframes = {

            "1M": "1mo",

            "3M": "3mo",

            "6M": "6mo",

            "1Y": "1y"

        }

        results = {}

        for label, period in timeframes.items():

            # -----------------------------
            # DOWNLOAD DATA
            # -----------------------------
            df = yf.download(

                ticker,

                period=period,

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
            # PRICE CHANGE
            # -----------------------------
            start_price = (
                df["Close"].iloc[0]
            )

            end_price = (
                df["Close"].iloc[-1]
            )

            percent_change = (

                (
                    end_price
                    - start_price
                )

                / start_price

            ) * 100

            # -----------------------------
            # TREND
            # -----------------------------
            if percent_change > 5:

                trend = "BULLISH"

            elif percent_change < -5:

                trend = "BEARISH"

            else:

                trend = "NEUTRAL"

            # -----------------------------
            # STORE RESULT
            # -----------------------------
            results[label] = {

                "trend": trend,

                "percent_change":
                round(float(percent_change), 2)

            }

        return {

            "ticker": ticker,

            "timeframes": results

        }