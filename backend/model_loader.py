import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Required features for your trained LSTM Autoencoder
FEATURES_7 = [
    "volt", "current", "soc",
    "max_single_volt", "min_single_volt",
    "max_temp", "min_temp"
]

def load_all():
    """
    Loads the model + scaler with clean relative paths.
    Works both locally and in deployments (Render, Railway, etc.).
    """
    MODEL_PATH = "models/lstm_autoencoder_mileage.keras"
    SCALER_PATH = "models/scaler.pkl"

    print("🔋 Loading model from:", MODEL_PATH)
    model = load_model(MODEL_PATH)

    print("📐 Loading scaler from:", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    SEQ_LEN = 256

    print("📏 Sequence length:", SEQ_LEN)
    print("✨ Model & Scaler Loaded Successfully!")

    return model, scaler, SEQ_LEN


def prepare_sequences(df: pd.DataFrame, scaler, seq_len: int):
    """
    Extracts 7 features → scales them → builds LSTM input sequences.
    """
    # Make sure all required columns exist
    missing = [c for c in FEATURES_7 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract the 7 exact model features
    df_features = df[FEATURES_7].copy()

    # Scale them using the trained scaler
    scaled_values = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_values, columns=FEATURES_7)

    sequences = []

    # Create overlapping LSTM sequences of length seq_len
    for i in range(len(scaled_df) - seq_len):
        seq = scaled_df.iloc[i:i + seq_len].values
        sequences.append(seq)

    sequences = np.array(sequences)

    return sequences, scaled_df
