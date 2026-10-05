# agents/prediction_agent/predictor.py
import numpy as np
import pandas as pd
import json
import os
import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# ── Custom layers must be defined and registered BEFORE load_model is called ──

@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim * self.att._num_heads,
            "num_heads": self.att._num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "dropout_rate": self.dropout1.rate,
        })
        return config


@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def call(self, x):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config


MODEL_DIR = "agents/prediction_agent/saved_models"


def _load_model_and_config(model_name: str):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    config_path = os.path.join(MODEL_DIR, f"{model_name}_config.json")
    if not os.path.exists(model_path):
        return None, None
    model = load_model(model_path)
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    return model, config


class PredictionAgent:
    def __init__(self):
        self.vol_model = None
        self.vol_config = None
        self.dir_model = None
        self.dir_config = None
        self.scaler = None
        self._load()

    def _load(self):
        # Load global scaler — fitted on all training data during training
        scaler_path = os.path.join(MODEL_DIR, "feature_scaler_transformer.save")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("[PredictionAgent] Global scaler loaded")
        else:
            print("[PredictionAgent] ⚠️  No scaler found — will use per-window scaling (degraded accuracy)")

        self.vol_model, self.vol_config = _load_model_and_config("transformer_volatility")
        self.dir_model, self.dir_config = _load_model_and_config("transformer_direction")

        if self.vol_model:
            print("[PredictionAgent] Volatility model loaded")
        if self.dir_model:
            print("[PredictionAgent] Direction model loaded")

    def _prepare_input(self, window: pd.DataFrame, config: dict) -> np.ndarray:
        # Ensure Close_yfin exists
        if "Close_yfin" not in window.columns and "Close" in window.columns:
            window = window.copy()
            window["Close_yfin"] = window["Close"]

        feature_cols = config.get("feature_columns", [])
        seq_len = config.get("sequence_length", 60)

        # Engineer missing features on the fly
        if "Return_1d" not in window.columns:
            window = window.copy()
            window["Return_1d"] = window["Close_yfin"].pct_change(1)
            window["Return_5d"] = window["Close_yfin"].pct_change(5)
            window["High_Low_range"] = (window["High"] - window["Low"]) / window["Close_yfin"]
            window["Volume_Change"] = window["Volume"].pct_change()
            window["Volume_SMA_ratio"] = window["Volume"] / window["Volume"].rolling(20).mean()
            window = window.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Build feature matrix
        X = np.zeros((seq_len, len(feature_cols)))
        for j, col in enumerate(feature_cols):
            if col in window.columns:
                vals = window[col].values[-seq_len:]
                X[-len(vals):, j] = vals

        # ── Use saved global scaler (same distribution as training) ──
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            # Fallback: per-window scaling — less accurate
            from sklearn.preprocessing import MinMaxScaler
            X_scaled = MinMaxScaler().fit_transform(X)

        return X_scaled[np.newaxis, :, :]  # (1, seq_len, n_features)

    def predict_direction(self, window: pd.DataFrame) -> dict:
        if self.dir_model is None:
            return {"signal": "HOLD", "up_probability": 0.5, "confidence": 0.0}
        X = self._prepare_input(window, self.dir_config)
        prob = float(self.dir_model.predict(X, verbose=0)[0][0])
        return {
            "signal": "BUY" if prob > 0.55 else ("SELL" if prob < 0.45 else "HOLD"),
            "up_probability": prob,
            "confidence": abs(prob - 0.5) * 2,
        }

    def predict_volatility(self, window: pd.DataFrame) -> dict:
        if self.vol_model is None:
            return {"label": "UNKNOWN", "probability": 0.5, "confidence": 0.0, "high_vol_probability": 0.5}
        X = self._prepare_input(window, self.vol_config)
        prob = float(self.vol_model.predict(X, verbose=0)[0][0])
        return {
            "label": "HIGH" if prob > 0.5 else "LOW",
            "probability": prob,
            "confidence": abs(prob - 0.5) * 2,
            "high_vol_probability": prob,
        }

    def predict_window(self, window: pd.DataFrame) -> float:
        """Backtest-compatible interface."""
        current_price = float(
            window["Close_yfin"].iloc[-1] if "Close_yfin" in window.columns
            else window["Close"].iloc[-1]
        )
        dir_result = self.predict_direction(window)
        prob = dir_result["up_probability"]
        self.last_up_probability = prob
        self.last_direction = "UP" if prob > 0.5 else "DOWN"
        self.last_confidence = dir_result["confidence"]
        return current_price * (1 + (prob - 0.5) * 0.04)

    def get_signal(self, window: pd.DataFrame) -> dict:
        """Full signal for the main analysis pipeline."""
        current_price = float(
            window["Close_yfin"].iloc[-1] if "Close_yfin" in window.columns
            else window["Close"].iloc[-1]
        )
        dir_result = self.predict_direction(window)
        vol_result = self.predict_volatility(window)
        return {
            "current_price": current_price,
            "predicted_price": current_price * (1 + (dir_result["up_probability"] - 0.5) * 0.04),
            "direction": "UP" if dir_result["up_probability"] > 0.5 else "DOWN",
            "up_probability": dir_result["up_probability"],
            "signal": dir_result["signal"],
            "confidence": dir_result["confidence"],
            "volatility_forecast": vol_result["label"],
            "volatility_probability": vol_result["high_vol_probability"],
            "volatility_confidence": vol_result["confidence"],
        }
