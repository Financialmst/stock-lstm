import numpy as np
import yfinance as yf
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from .indicators import add_technical_indicators


FEATURE_COLUMNS = [

    'Close_yfin',
    'SMA_10',
    'SMA_20',
    'EMA_10',
    'EMA_20',
    'Rolling_STD_10',
    'Rolling_Max_10',
    'Rolling_Min_10',
    'Momentum_10',
    'RSI_14',
    'MACD',
    'Signal_Line',
    'Bollinger_Width'

]


class PredictionAgent:

    def __init__(self):

        print("Loading Prediction Agent...")

        self.model = load_model(
            "agents/prediction_agent/saved_models/model.h5"
        )

        self.feature_scaler = joblib.load(
            "agents/prediction_agent/saved_models/feature_scaler.save"
        )

        self.target_scaler = joblib.load(
            "agents/prediction_agent/saved_models/target_scaler.save"
        )

        print("Prediction Agent Ready")

    # --------------------------------
    # FETCH STOCK DATA
    # --------------------------------
    def fetch_stock_data(self, ticker):

        df = yf.download(

            ticker,

            period="1y",

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

        return df

    # --------------------------------
    # PREPROCESS
    # --------------------------------
    def preprocess_data(self, df):

        df['Close_yfin'] = df['Close']

        df = add_technical_indicators(df)

        df = df.dropna()

        feature_df = df[FEATURE_COLUMNS]

        scaled_data = (
            self.feature_scaler
            .transform(feature_df)
        )

        return scaled_data, df

    # --------------------------------
    # PREDICT
    # --------------------------------
    def predict(self, ticker):

        # --------------------------------
        # FETCH DATA
        # --------------------------------
        df = self.fetch_stock_data(
            ticker
        )

        scaled_data, processed_df = (
            self.preprocess_data(df)
        )

        # --------------------------------
        # LAST 60 TIMESTEPS
        # --------------------------------
        last_60 = scaled_data[-60:]

        X_test = np.array([last_60])

        # --------------------------------
        # MODEL PREDICTION
        # --------------------------------
        prediction = self.model.predict(
            X_test
        )

        print("RAW MODEL OUTPUT:")
        print(prediction)

        predicted_price = (

            self.target_scaler
            .inverse_transform(
                prediction
            )[0][0]

        )

        print("INVERSE TRANSFORMED:")
        print(predicted_price)

        # --------------------------------
        # CURRENT PRICE
        # --------------------------------
        current_price = (

            processed_df['Close']
            .iloc[-1]

        )

        # --------------------------------
        # PREDICTION SAFETY CAP
        # --------------------------------
        max_move = current_price * 0.15

        predicted_price = max(

            current_price - max_move,

            min(

                predicted_price,

                current_price + max_move

            )

        )

        # --------------------------------
        # TREND
        # --------------------------------
        trend = (

            "bullish"

            if predicted_price > current_price

            else "bearish"

        )

        # --------------------------------
        # CONFIDENCE
        # --------------------------------
        price_difference = abs(

            predicted_price
            - current_price

        )

        confidence = min(

            price_difference / current_price,

            1.0

        )

        confidence_percent = round(

            confidence * 100,

            2

        )

        # --------------------------------
        # DECISION
        # --------------------------------
        if predicted_price > current_price * 1.02:

            decision = "BUY"

        elif predicted_price < current_price * 0.98:

            decision = "SELL"

        else:

            decision = "HOLD"

        # --------------------------------
        # RETURN RESULT
        # --------------------------------
        result = {

            "ticker":
            ticker,

            "current_price":
            round(float(current_price), 2),

            "predicted_price":
            round(float(predicted_price), 2),

            "trend":
            trend,

            "confidence":
            confidence_percent,

            "decision":
            decision

        }

        return result