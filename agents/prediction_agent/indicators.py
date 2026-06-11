import pandas as pd
import numpy as np


def add_technical_indicators(df):

    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential moving averages
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Rolling statistics
    df['Rolling_STD_10'] = df['Close'].rolling(window=10).std()
    df['Rolling_Max_10'] = df['Close'].rolling(window=10).max()
    df['Rolling_Min_10'] = df['Close'].rolling(window=10).min()

    # Momentum
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

    # RSI
    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()

    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss

    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()

    ema26 = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = ema12 - ema26

    df['Signal_Line'] = (
        df['MACD']
        .ewm(span=9, adjust=False)
        .mean()
    )

    # Bollinger Width
    rolling_std = df['Close'].rolling(window=20).std()

    upper_band = df['SMA_20'] + (2 * rolling_std)

    lower_band = df['SMA_20'] - (2 * rolling_std)

    bollinger_width = upper_band - lower_band

    df['Bollinger_Width'] = bollinger_width
    return df