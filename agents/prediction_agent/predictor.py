# agents/prediction_agent/predictor.py
# Updated to work with the direction-classification model.
# predict_window() now returns a float price (for backtest compatibility)
# but internally uses direction probability to shift the current price.

import numpy as np
import pandas as pd
import joblib
import json
import os
from tensorflow.keras.models import load_model

MODEL_DIR = "agents/prediction_agent/saved_models"


class PredictionAgent:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.config = None
        self._load()

    def _load(self):
        config_path = os.path.join(MODEL_DIR, "model_config.json")
        scaler_path = os.path.join(MODEL_DIR, "feature_scaler.save")

        # Try .keras first, fall back to .h5
        model_path = os.path.join(MODEL_DIR, "model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "model.h5")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found in {MODEL_DIR}")

        self.model = load_model(model_path)
        self.feature_scaler = joblib.load(scaler_path)

        if os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            # Backwards compatibility with old model
            self.config = {
                "sequence_length": 60,
                "feature_columns": [
                    'Close_yfin', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
                    'Rolling_STD_10', 'Rolling_Max_10', 'Rolling_Min_10',
                    'Momentum_10', 'RSI_14', 'MACD', 'Signal_Line', 'Bollinger_Width'
                ],
                "predict_direction": False,
                "forward_bars": 5,
            }

    def predict_window(self, window: pd.DataFrame) -> float:
        print(f"DEBUG columns: {window.columns.tolist()}
        if "Close_yfin" not in window.columns and "Close" in window.columns:
                window = window.copy()
                window["Close_yfin"] = window["Close"]
        feature_cols = self.config["feature_columns"]
        seq_len = self.config["sequence_length"]
        predict_direction = self.config.get("predict_direction", False)

        # Align columns — use only what the model was trained on
        available = [c for c in feature_cols if c in window.columns]
        if len(available) < len(feature_cols):
            missing = set(feature_cols) - set(available)
            print(f"[PredictionAgent] Warning: missing features {missing}. Using zeros.")

        X = np.zeros((seq_len, len(feature_cols)))
        for j, col in enumerate(feature_cols):
            if col in window.columns:
                vals = window[col].values[-seq_len:]
                X[-len(vals):, j] = vals

        # Scale
        X_scaled = self.feature_scaler.transform(X)
        X_input = X_scaled[np.newaxis, :, :]   # shape: (1, seq_len, n_features)

        raw_output = self.model.predict(X_input, verbose=0)[0][0]
        print(f"up_prob={raw_output:.3f}  confidence={abs(raw_output-0.5)*2:.3f}  direction={'UP' if raw_output>0.5 else 'DOWN'}")
        current_price = float(window["Close_yfin"].iloc[-1] if "Close_yfin" in window.columns
                              else window["Close"].iloc[-1])

        if predict_direction:
            # raw_output is a probability (0-1): >0.5 = UP, <0.5 = DOWN
            # Convert to a price prediction the backtest can use
            up_prob = float(raw_output)
            # Expected directional move: scale by typical 5-day move (1%)
            expected_move = (up_prob - 0.5) * 0.04   # ±2% at full confidence
            predicted_price = current_price * (1 + expected_move)

            # Also expose the raw probability for downstream use
            self.last_up_probability = up_prob
            self.last_direction = "UP" if up_prob > 0.5 else "DOWN"
            self.last_confidence = abs(up_prob - 0.5) * 2   # 0-1 scale
        else:
            # Old regression model — raw_output is a scaled price
            # If you saved a target_scaler, inverse transform here:
            target_scaler_path = os.path.join(MODEL_DIR, "target_scaler.save")
            if os.path.exists(target_scaler_path):
                target_scaler = joblib.load(target_scaler_path)
                predicted_price = float(target_scaler.inverse_transform([[raw_output]])[0][0])
            else:
                predicted_price = float(raw_output)

            self.last_up_probability = 1.0 if predicted_price > current_price else 0.0
            self.last_direction = "UP" if predicted_price > current_price else "DOWN"
            self.last_confidence = min(abs(predicted_price - current_price) / current_price * 10, 1.0)
            # Add this temporarily to predict_window() after computing raw_output
        

        return predicted_price

    def get_signal(self, window: pd.DataFrame) -> dict:
        """
        Higher-level interface — returns a full signal dict.
        Use this in your main analysis pipeline.
        """
        predicted_price = self.predict_window(window)
        current_price = float(window["Close_yfin"].iloc[-1] if "Close_yfin" in window.columns
                              else window["Close"].iloc[-1])

        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "direction": self.last_direction,
            "up_probability": self.last_up_probability,
            "confidence": self.last_confidence,
            "signal": "BUY" if self.last_up_probability > 0.55 else (
                       "SELL" if self.last_up_probability < 0.45 else "HOLD"
            ),
        }