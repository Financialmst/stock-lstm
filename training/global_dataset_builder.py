import yfinance as yf
import pandas as pd

from agents.prediction_agent.indicators import (
    add_technical_indicators
)


# --------------------------------
# GLOBAL TICKERS
# --------------------------------

TICKERS = [

    # --------------------------------
    # US STOCKS
    # --------------------------------
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",

    # --------------------------------
    # INDIA
    # --------------------------------
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",

    # --------------------------------
    # INDICES
    # --------------------------------
    "^NSEI",
    "^GSPC",
    "^DJI",
    "^IXIC",

    # --------------------------------
    # CRYPTO
    # --------------------------------
    "BTC-USD",
    "ETH-USD",

    # --------------------------------
    # COMMODITIES
    # --------------------------------
    "GC=F",
    "CL=F"

]


all_dataframes = []


# --------------------------------
# DOWNLOAD DATA
# --------------------------------
for ticker in TICKERS:

    try:

        print(f"Downloading {ticker}...")

        df = yf.download(

            ticker,

            period="5y",

            auto_adjust=True

        )

        # --------------------------------
        # FIX MULTIINDEX
        # --------------------------------
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        # --------------------------------
        # BASIC FEATURES
        # --------------------------------
        df['Ticker'] = ticker

        df['Close_yfin'] = df['Close']

        # --------------------------------
        # TECHNICAL INDICATORS
        # --------------------------------
        df = add_technical_indicators(df)

        # --------------------------------
        # TARGET
        # --------------------------------
        df['Target'] = (
            df['Close']
            .shift(-1)
        )

        df = df.dropna()

        all_dataframes.append(df)

        print(f"{ticker} complete")

    except Exception as e:

        print(f"ERROR with {ticker}")

        print(e)


# --------------------------------
# COMBINE EVERYTHING
# --------------------------------
final_df = pd.concat(

    all_dataframes,

    ignore_index=True

)

# --------------------------------
# SAVE DATASET
# --------------------------------
final_df.to_csv(

    "data/global_market_dataset.csv",

    index=False

)

print("GLOBAL DATASET CREATED")

print(final_df.shape)