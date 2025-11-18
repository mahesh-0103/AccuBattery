import numpy as np
import pandas as pd
import joblib # Required for joblib.load(SCALER_PATH)
# Use the correct Keras import path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed # Import necessary layers
import os
from pathlib import Path

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
    # Render's root directory is /opt/render/project/src/, which is where your 'backend' folder is located.
    # We must start from the parent of 'backend' to use 'models/'
    
    # In Render environment, the working directory is the 'backend' folder itself (because of Root Directory setting).
    # Therefore, we set BASE_DIR to the current working directory.
    BASE_DIR = Path(os.getcwd())
    
    # Path inside the backend directory: backend/models/...
    MODEL_PATH = BASE_DIR / "models" / "lstm_autoencoder_mileage.keras"
    SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

    # CRITICAL PATH CHECK
    if not MODEL_PATH.exists():
        print(f"🚨 FATAL: Model file NOT found at: {MODEL_PATH.resolve()}")
        # This will prevent the Keras crash and force a visible FileNotFoundError in logs
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

    print("🔋 Loading model from:", MODEL_PATH)

    # --- START OF FIX: Using custom_objects to bypass ValueError ---
    # Since the model was saved with an older Keras convention, we must define 
    # the standard layers as 'custom objects'. This often forces the loader 
    # to use the layer class definition from the current TF/Keras version,
    # and ignore unrecognized configuration keys like 'batch_input_shape'.
    
    # We include all common layers used in an LSTM Autoencoder, mapped to their current classes.
    custom_layer_config = {
        'LSTM': LSTM,
        'Dense': Dense,
        'Dropout': Dropout,
        'RepeatVector': RepeatVector,
        'TimeDistributed': TimeDistributed
        # Add any other custom layers you may have used here.
    }

    try:
        model = load_model(MODEL_PATH, custom_objects=custom_layer_config)
    except ValueError as e:
        print(f"❌ Keras Load Error (Attempting Compile=False): {e}")
        # Secondary fix: Try disabling compilation, which sometimes fails due to optimizer config mismatch
        model = load_model(MODEL_PATH, custom_objects=custom_layer_config, compile=False)
        print("✅ Model loaded successfully using compile=False flag.")
    # --- END OF FIX ---


    # Ensure joblib is imported for the next line
    # NOTE: You must ensure joblib is in your requirements.txt
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
