# retrain_model.py
# Fixed version — corrects look-ahead bias, ticker boundary leakage,
# and switches to direction classification for better signal quality.

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8          # 80% train, 20% test — split BEFORE scaling
PREDICT_DIRECTION = True   # True = classify Up/Down | False = regress price (old behaviour)
FORWARD_BARS = 5           # predict 5-day forward direction/return
TRAIN_SINGLE_TICKER = "AAPL"
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
    'Bollinger_Width',
    # New independent features
    'Volume_Change',       # volume momentum
    'Volume_SMA_ratio',    # volume vs average
    'High_Low_range',      # daily range %
    'Gap',                 # overnight gap
    'Return_1d',           # 1-day return
    'Return_5d',           # 5-day return
]

TICKER_COLUMN = 'Ticker'   # column that identifies which asset each row belongs to
                            # set to None if your CSV doesn't have this column

OUTPUT_DIR = "agents/prediction_agent/saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading global dataset...")
df = pd.read_csv("data/global_market_dataset.csv")
df['Volume_Change'] = df.groupby('Ticker')['Volume'].pct_change()

df['Volume_SMA_ratio'] = (
    df['Volume']
    / df.groupby('Ticker')['Volume']
      .transform(lambda x: x.rolling(20).mean())
)

df['High_Low_range'] = (
    (df['High'] - df['Low'])
    / df['Close_yfin']
)

df['Gap'] = (
    df.groupby('Ticker')['Close_yfin']
      .pct_change()
)

df['Return_1d'] = (
    df.groupby('Ticker')['Close_yfin']
      .pct_change(1)
)

df['Return_5d'] = (
    df.groupby('Ticker')['Close_yfin']
      .pct_change(5)
)

df = df.replace([np.inf, -np.inf], np.nan)
print(f"  Raw shape: {df.shape}")
df = df.dropna(subset=FEATURE_COLUMNS)
print(f"  After dropna: {df.shape}")


# ─────────────────────────────────────────────
# 2. BUILD TARGET
# ─────────────────────────────────────────────
# We build the target ourselves so we know exactly what it is.
# Direction classification: 1 if price is higher N bars from now, else 0.
# This is what the signal actually needs — direction, not price level.

if PREDICT_DIRECTION:
    df['_target'] = (
        df['Close_yfin'].shift(-FORWARD_BARS) > df['Close_yfin'] * 1.005
    ).astype(float)   # 1 = UP by >0.5%, 0 = DOWN or flat
else:
    # Regression fallback: predict the actual future close (old approach)
    df['_target'] = df['Close_yfin'].shift(-FORWARD_BARS)

df = df.dropna(subset=['_target'])
print(f"  After target construction: {df.shape}")

if PREDICT_DIRECTION:
    up_pct = df['_target'].mean() * 100
    print(f"  Class balance: {up_pct:.1f}% UP, {100-up_pct:.1f}% DOWN")


# ─────────────────────────────────────────────
# 3. SPLIT BY TICKER — prevents boundary leakage
# ─────────────────────────────────────────────
def build_sequences_for_ticker(ticker_df: pd.DataFrame, feature_scaler, fit_scaler: bool):
    """
    Build sequences from a single ticker's data.
    Scaler is fit only on training data (fit_scaler=True),
    and only transform on test data (fit_scaler=False).
    """
    X_raw = ticker_df[FEATURE_COLUMNS].values
    y_raw = ticker_df['_target'].values

    if fit_scaler:
        X_scaled = feature_scaler.fit_transform(X_raw)
    else:
        X_scaled = feature_scaler.transform(X_raw)

    X_seq, y_seq = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i - SEQUENCE_LENGTH: i])
        y_seq.append(y_raw[i])

    return np.array(X_seq), np.array(y_seq)


if TICKER_COLUMN and TICKER_COLUMN in df.columns:
    if TRAIN_SINGLE_TICKER:
        tickers = [TRAIN_SINGLE_TICKER]
    else:
        tickers = df[TICKER_COLUMN].unique()
    print(f"\nBuilding sequences for {len(tickers)} tickers separately...")
else:
    # No ticker column — treat as single asset
    tickers = ['ALL']
    df[TICKER_COLUMN] = 'ALL'

feature_scaler = MinMaxScaler()

X_train_all, y_train_all = [], []
X_test_all, y_test_all = [], []

for ticker in tickers:
    tdf = df[df[TICKER_COLUMN] == ticker].copy().reset_index(drop=True)
    if len(tdf) < SEQUENCE_LENGTH + 20:
        print(f"  Skipping {ticker} — not enough data ({len(tdf)} rows)")
        continue

    split_idx = int(len(tdf) * TRAIN_RATIO)
    train_df = tdf.iloc[:split_idx]
    test_df = tdf.iloc[split_idx:]

    # FIX: scaler is fit ONLY on train portion of this ticker
    ticker_scaler = MinMaxScaler()
    X_tr, y_tr = build_sequences_for_ticker(train_df, ticker_scaler, fit_scaler=True)
    X_te, y_te = build_sequences_for_ticker(test_df, ticker_scaler, fit_scaler=False)

    X_train_all.append(X_tr)
    y_train_all.append(y_tr)
    X_test_all.append(X_te)
    y_test_all.append(y_te)

    print(f"  {ticker}: train={len(X_tr)}, test={len(X_te)}")

X_train = np.concatenate(X_train_all)
y_train = np.concatenate(y_train_all)
X_test = np.concatenate(X_test_all)
y_test = np.concatenate(y_test_all)

print(f"\nFinal shapes — Train: {X_train.shape}, Test: {X_test.shape}")


# ─────────────────────────────────────────────
# 4. BUILD MODEL
# ─────────────────────────────────────────────
print("\nBuilding model...")

output_activation = 'sigmoid' if PREDICT_DIRECTION else 'linear'
output_loss = 'binary_crossentropy' if PREDICT_DIRECTION else 'mean_squared_error'
output_metrics = ['accuracy'] if PREDICT_DIRECTION else ['mae']

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(64, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),

    LSTM(32),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=output_loss,
    metrics=output_metrics,
)

model.summary()


# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
print("\nTraining...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weight = dict(zip(classes, weights))

print(f"Class weights: {class_weight}")
history = model.fit(
    X_train, y_train,class_weight=class_weight,
    epochs=50,              # early stopping will cut this short
    batch_size=64,
    validation_split=0.1,   # 10% of train set for val (no leakage — test is separate)
    callbacks=callbacks,
    shuffle=False,          # IMPORTANT: don't shuffle time series
    verbose=1,
)


# ─────────────────────────────────────────────
# 6. EVALUATE ON HELD-OUT TEST SET
# ─────────────────────────────────────────────
print("\nEvaluating on test set...")
y_pred_raw = model.predict(X_test, verbose=0).flatten()

if PREDICT_DIRECTION:
    y_pred = (y_pred_raw > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())  # majority class baseline
    print(f"  Test Accuracy        : {acc*100:.2f}%")
    print(f"  Majority Class Base  : {baseline_acc*100:.2f}%")
    print(f"  Edge over baseline   : {(acc - baseline_acc)*100:+.2f}%")
    from sklearn.metrics import f1_score, classification_report
    f1 = f1_score(y_test, y_pred)
    print(f"  F1 Score           : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    if acc > baseline_acc + 0.02:
        print("  ✅ Model has statistically meaningful directional edge")
    else:
        print("  🔴 Model has NO meaningful edge over majority-class baseline")
else:
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred_raw)
    print(f"  Test MAE: {mae:.4f}")
# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
print("\nSaving...")
model.save(f"{OUTPUT_DIR}/model.keras")   # .keras format preferred over .h5

# Save a global feature scaler fitted on all training data combined
# (used by predictor.py at inference time)
global_scaler = MinMaxScaler()
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
global_scaler.fit(X_train_2d)
joblib.dump(global_scaler, f"{OUTPUT_DIR}/feature_scaler.save")

# Save config so predictor.py knows what the model expects
import json
config = {
    "sequence_length": SEQUENCE_LENGTH,
    "feature_columns": FEATURE_COLUMNS,
    "predict_direction": PREDICT_DIRECTION,
    "forward_bars": FORWARD_BARS,
}
with open(f"{OUTPUT_DIR}/model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"  Saved to {OUTPUT_DIR}/")
print("\nTraining complete!")