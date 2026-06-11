import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

# --------------------------------
# FEATURE COLUMNS
# --------------------------------
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

# --------------------------------
# LOAD GLOBAL DATASET
# --------------------------------
print("Loading global dataset...")

df = pd.read_csv(
    "data/global_market_dataset.csv"
)

print("Dataset Shape:")
print(df.shape)

# --------------------------------
# DROP NaN
# --------------------------------
df = df.dropna()

# --------------------------------
# FEATURES + TARGET
# --------------------------------
print("Preparing features...")

X = df[FEATURE_COLUMNS]

y = df[['Target']]

# --------------------------------
# SCALERS
# --------------------------------
feature_scaler = MinMaxScaler()

target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(
    X
)

y_scaled = target_scaler.fit_transform(
    y
)

# --------------------------------
# CREATE SEQUENCES
# --------------------------------
print("Creating sequences...")

X_sequences = []

y_sequences = []

sequence_length = 60

for i in range(

    sequence_length,

    len(X_scaled)

):

    X_sequences.append(

        X_scaled[
            i-sequence_length:i
        ]

    )

    y_sequences.append(
        y_scaled[i]
    )

X_sequences = np.array(
    X_sequences
)

y_sequences = np.array(
    y_sequences
)

print("X shape:")
print(X_sequences.shape)

print("y shape:")
print(y_sequences.shape)

# --------------------------------
# BUILD MODEL
# --------------------------------
print("Building model...")

model = Sequential()

# --------------------------------
# LSTM 1
# --------------------------------
model.add(

    LSTM(

        64,

        return_sequences=True,

        input_shape=(

            X_sequences.shape[1],

            X_sequences.shape[2]

        )

    )

)

model.add(
    Dropout(0.2)
)

# --------------------------------
# LSTM 2
# --------------------------------
model.add(
    LSTM(64)
)

model.add(
    Dropout(0.2)
)

# --------------------------------
# OUTPUT
# --------------------------------
model.add(
    Dense(1)
)

# --------------------------------
# COMPILE
# --------------------------------
model.compile(

    optimizer='adam',

    loss='mean_squared_error'

)

# --------------------------------
# TRAIN
# --------------------------------
print("Training model...")

history = model.fit(

    X_sequences,

    y_sequences,

    epochs=20,

    batch_size=32,

    validation_split=0.1

)

# --------------------------------
# SAVE MODEL
# --------------------------------
print("Saving model...")

model.save(

    "agents/prediction_agent/saved_models/model.h5"

)

# --------------------------------
# SAVE SCALERS
# --------------------------------
joblib.dump(

    feature_scaler,

    "agents/prediction_agent/saved_models/feature_scaler.save"

)

joblib.dump(

    target_scaler,

    "agents/prediction_agent/saved_models/target_scaler.save"

)

print("Training complete!")