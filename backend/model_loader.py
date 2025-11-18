import joblib
import numpy as np
import pandas as pd
# Use the correct Keras import path
from tensorflow.keras.models import load_model 
import os # NEW: Added for path checking
from pathlib import Path # NEW: Added for robust path handling

# Required features for your trained LSTM Autoencoder
FEATURES_7 = [
    "volt", "current", "soc",
    "max_single_volt", "min_single_volt",
    "max_temp", "min_temp"
]

def load_all():
    """
    Loads the model + scaler with clean relative paths.
    """
    # Use Pathlib for system-independent path joining
    BASE_DIR = Path(os.getcwd()) 
    MODEL_PATH = BASE_DIR / "models" / "lstm_autoencoder_mileage.keras"
    SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

    # NEW: CRITICAL PATH CHECK
    if not MODEL_PATH.exists():
        print(f"🚨 FATAL: Model file NOT found at: {MODEL_PATH.resolve()}")
        # This will prevent the Keras crash and force a visible FileNotFoundError in logs
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

    print("🔋 Loading model from:", MODEL_PATH)
    # The load_model function is the direct source of the error.
    # If the environment pins are correct, the file itself is the problem.
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
