# training/train_transformer.py
from tensorflow.keras.utils import register_keras_serializable
import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8
TICKER_COLUMN = 'Ticker'
FORWARD_BARS = 5
FEATURE_COLUMNS = [
    'Close_yfin',
    'SMA_10', 'SMA_20',
    'EMA_10', 'EMA_20',
    'Rolling_STD_10',
    'Rolling_Max_10',
    'Rolling_Min_10',
    'Momentum_10',
    'RSI_14',
    'MACD', 'Signal_Line',
    'Bollinger_Width',
    'Return_1d',
    'Return_5d',
    'High_Low_range',
    'Volume_Change',
    'Volume_SMA_ratio',
]

OUTPUT_DIR = "agents/prediction_agent/saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def build_transformer(seq_len, n_features, embed_dim=64, num_heads=4,
                      ff_dim=128, n_blocks=3, dropout=0.1,
                      output_activation='sigmoid'):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(embed_dim)(inputs)
    x = PositionalEncoding(seq_len, embed_dim)(x)
    for _ in range(n_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation=output_activation)(x)
    return Model(inputs, outputs)


def load_and_engineer(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")
    df['Return_1d'] = df.groupby(TICKER_COLUMN)['Close_yfin'].pct_change(1)
    df['Return_5d'] = df.groupby(TICKER_COLUMN)['Close_yfin'].pct_change(5)
    df['High_Low_range'] = (df['High'] - df['Low']) / df['Close_yfin']
    df['Volume_Change'] = df.groupby(TICKER_COLUMN)['Volume'].pct_change()
    df['Volume_SMA_ratio'] = df['Volume'] / df.groupby(TICKER_COLUMN)['Volume'].transform(
        lambda x: x.rolling(20).mean()
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS)
    print(f"  After feature engineering: {df.shape}")
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df['_daily_range'] = (df['High'] - df['Low']) / df['Close_yfin']
    df['_avg_range'] = df.groupby(TICKER_COLUMN)['_daily_range'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['_vol_target'] = (
        df.groupby(TICKER_COLUMN)['_daily_range'].shift(-1) > df['_avg_range']
    ).astype(float)
    df['_fwd_5d'] = df.groupby(TICKER_COLUMN)['Close_yfin'].shift(-5)
    df['_dir_target'] = (df['_fwd_5d'] > df['Close_yfin'] * 1.005).astype(float)
    df = df.dropna(subset=['_vol_target', '_dir_target'])
    return df


def build_sequences(df, target_col):
    tickers = df[TICKER_COLUMN].unique()
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    all_train_raw = []  # for global scaler

    for ticker in tickers:
        tdf = df[df[TICKER_COLUMN] == ticker].copy().reset_index(drop=True)
        if len(tdf) < SEQUENCE_LENGTH + 30:
            print(f"  Skipping {ticker} — not enough data ({len(tdf)} rows)")
            continue

        split_idx = int(len(tdf) * TRAIN_RATIO)
        train_df = tdf.iloc[:split_idx]
        test_df = tdf.iloc[split_idx:]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
        X_test_scaled = scaler.transform(test_df[FEATURE_COLUMNS])

        # Collect raw train features for global scaler
        all_train_raw.append(train_df[FEATURE_COLUMNS].values)

        def make_seqs(X_scaled, y_vals):
            Xs, ys = [], []
            for i in range(SEQUENCE_LENGTH, len(X_scaled)):
                Xs.append(X_scaled[i - SEQUENCE_LENGTH: i])
                ys.append(y_vals.iloc[i])
            return np.array(Xs), np.array(ys)

        X_tr, y_tr = make_seqs(X_train_scaled, train_df[target_col])
        X_te, y_te = make_seqs(X_test_scaled, test_df[target_col])

        if len(X_tr) > 0 and len(X_te) > 0:
            X_train_all.append(X_tr)
            y_train_all.append(y_tr)
            X_test_all.append(X_te)
            y_test_all.append(y_te)
            print(f"  {ticker:15s}: train={len(X_tr)}, test={len(X_te)}")

    # ── Save global scaler fitted on all training data ──────────────
    all_train_combined = np.concatenate(all_train_raw, axis=0)
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_train_combined)
    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler_transformer.save")
    joblib.dump(global_scaler, scaler_path)
    print(f"  Global scaler saved → {scaler_path}")

    return (
        np.concatenate(X_train_all),
        np.concatenate(y_train_all),
        np.concatenate(X_test_all),
        np.concatenate(y_test_all),
    )


def train_model(X_train, y_train, X_test, y_test, model_name, target_col):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    up_pct = y_train.mean() * 100
    print(f"  Class balance: {up_pct:.1f}% positive, {100-up_pct:.1f}% negative")
    print(f"{'='*60}")

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes.astype(int), weights))

    model = build_transformer(
        seq_len=SEQUENCE_LENGTH,
        n_features=len(FEATURE_COLUMNS),
        embed_dim=64, num_heads=4, ff_dim=128, n_blocks=3, dropout=0.15,
    )
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    model.fit(X_train, y_train, epochs=60, batch_size=64,
              validation_split=0.1, callbacks=callbacks,
              class_weight=class_weight, shuffle=False, verbose=1)

    print(f"\nEvaluating {model_name}...")
    y_pred = (model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    baseline = max(y_train.mean(), 1 - y_train.mean())
    edge = acc - baseline

    print(f"  Test Accuracy      : {acc*100:.2f}%")
    print(f"  Majority Baseline  : {baseline*100:.2f}%")
    print(f"  Edge over baseline : {edge*100:+.2f}%")
    if edge > 0.02:
        print(f"  ✅ {model_name} has meaningful edge — deploying")
    elif edge > 0:
        print(f"  🟡 {model_name} marginal edge — monitor closely")
    else:
        print(f"  🔴 {model_name} no edge")

    save_path = os.path.join(OUTPUT_DIR, f"{model_name}.keras")
    model.save(save_path)
    print(f"  Saved: {save_path}")

    config = {
        "model_name": model_name,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns": FEATURE_COLUMNS,
        "target": target_col,
        "test_accuracy": round(acc * 100, 2),
        "edge_over_baseline": round(edge * 100, 2),
        "architecture": "transformer",
    }
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return model, acc, edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["volatility", "direction", "both"], default="both")
    parser.add_argument("--data", default="data/global_market_dataset.csv")
    args = parser.parse_args()

    df = load_and_engineer(args.data)
    df = build_targets(df)
    results = {}

    if args.target in ("volatility", "both"):
        print("\n── VOLATILITY MODEL ─────────────────────────────────────")
        X_tr, y_tr, X_te, y_te = build_sequences(df, '_vol_target')
        _, acc, edge = train_model(X_tr, y_tr, X_te, y_te, "transformer_volatility", "_vol_target")
        results["volatility"] = {"accuracy": acc, "edge": edge}

    if args.target in ("direction", "both"):
        print("\n── DIRECTION MODEL ──────────────────────────────────────")
        X_tr, y_tr, X_te, y_te = build_sequences(df, '_dir_target')
        _, acc, edge = train_model(X_tr, y_tr, X_te, y_te, "transformer_direction", "_dir_target")
        results["direction"] = {"accuracy": acc, "edge": edge}

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for name, r in results.items():
        status = "✅" if r["edge"] > 0.02 else ("🟡" if r["edge"] > 0 else "🔴")
        print(f"  {name:12s}: Accuracy={r['accuracy']*100:.2f}%  Edge={r['edge']*100:+.2f}%  {status}")


if __name__ == "__main__":
    main()
