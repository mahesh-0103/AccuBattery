import numpy as np
import pandas as pd
import joblib 
import os
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector # Only need to load if not using a compiled model

from typing import Tuple, List, Dict
from pathlib import Path

# --- CONFIGURATION FOR HYBRID MODEL ARTIFACTS ---
# Define the paths for the four required artifacts relative to the BASE_DIR/models/
LSTM_ENCODER_MODEL_NAME = "lstm_feature_encoder.h5"
LGBM_CLASSIFIER_MODEL_NAME = "lgbm_anomaly_classifier.txt"
LSTM_SEQUENCE_SCALER_NAME = "lstm_sequence_scaler.pkl"
LGBM_FEATURE_SCALER_NAME = "lgbm_feature_scaler.pkl"

# --- FEATURE DEFINITIONS (Must match the training script) ---
# Raw columns in the time-series batch (including timestamp if present)
COLUMNS: List[str] = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
N_FEATURES_RAW = len(COLUMNS) # 8 features
SEQ_LEN_RAW = 256
LSTM_INPUT_SEQ_LEN = SEQ_LEN_RAW - 1

# --- 1. FEATURE ENGINEERING FUNCTION (Copied from training script) ---

def extract_handcrafted_features(batch: np.ndarray) -> pd.Series:
    """Extracts statistical and domain-specific handcrafted features."""
    df = pd.DataFrame(batch, columns=COLUMNS)
    feats = {}
    
    # Statistical features for key physical variables (excluding 'timestamp')
    for col in COLUMNS[:-1]: 
        feats[f'{col}_mean'] = df[col].mean()
        feats[f'{col}_std'] = df[col].std()
        feats[f'{col}_min'] = df[col].min()
        feats[f'{col}_max'] = df[col].max()
        
        # Change rate features
        feats[f'{col}_diff_mean'] = df[col].diff().mean()
        feats[f'{col}_diff_std'] = df[col].diff().std()
        
        # Volatility features
        feats[f'{col}_rolling_std_10'] = df[col].rolling(window=10).std().mean()
        
    # Domain-specific range features
    feats['volt_range_avg'] = df['max_single_volt'].mean() - df['min_single_volt'].mean()
    feats['temp_range_avg'] = df['max_temp'].mean() - df['min_temp'].mean()
    
    return pd.Series(feats)


# --- 2. LOADER FUNCTION (Loads all four artifacts) ---

def load_all_hybrid() -> Tuple[Sequential, lgb.Booster, joblib.Parallel, joblib.Parallel, int]:
    """
    Loads all four required artifacts for the hybrid model pipeline.
    """
    BASE_DIR = Path(os.getcwd())
    MODELS_DIR = BASE_DIR / "hybrid_models" # Assumes model saving was to 'hybrid_models' folder
    
    LSTM_ENCODER_PATH = MODELS_DIR / LSTM_ENCODER_MODEL_NAME
    LGBM_CLASSIFIER_PATH = MODELS_DIR / LGBM_CLASSIFIER_MODEL_NAME
    LSTM_SCALER_PATH = MODELS_DIR / LSTM_SEQUENCE_SCALER_NAME
    LGBM_SCALER_PATH = MODELS_DIR / LGBM_FEATURE_SCALER_NAME

    # 1. Load Keras LSTM Encoder
    if not LSTM_ENCODER_PATH.exists():
        raise FileNotFoundError(f"LSTM Encoder Model NOT found at: {LSTM_ENCODER_PATH.resolve()}")

    print(f"🧠 Loading LSTM Encoder from: {LSTM_ENCODER_PATH}")
    # Custom objects are usually only needed if using custom layers, but kept for robustness.
    # The LSTM encoder was saved as a Sequential model of layers up to the LSTM layer.
    lstm_encoder = load_model(LSTM_ENCODER_PATH, compile=False) 
    
    # 2. Load LightGBM Classifier
    if not LGBM_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"LightGBM Classifier NOT found at: {LGBM_CLASSIFIER_PATH.resolve()}")

    print(f"🌳 Loading LightGBM Classifier from: {LGBM_CLASSIFIER_PATH}")
    lgbm_classifier = lgb.Booster(model_file=str(LGBM_CLASSIFIER_PATH))

    # 3. Load Scalers
    if not LSTM_SCALER_PATH.exists():
        raise FileNotFoundError(f"LSTM Sequence Scaler NOT found at: {LSTM_SCALER_PATH.resolve()}")
    if not LGBM_SCALER_PATH.exists():
        raise FileNotFoundError(f"LightGBM Feature Scaler NOT found at: {LGBM_SCALER_PATH.resolve()}")
        
    print(f"📐 Loading scalers...")
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    lgbm_scaler = joblib.load(LGBM_SCALER_PATH)

    print("\n✨ All Hybrid Components Loaded Successfully!")

    return lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler, SEQ_LEN_RAW


# --- 3. INFERENCE FUNCTION (Processes raw data to final prediction) ---

def predict_anomaly(
    raw_sequence_df: pd.DataFrame, 
    lstm_encoder: Sequential, 
    lgbm_classifier: lgb.Booster, 
    lstm_scaler: joblib.Parallel, 
    lgbm_scaler: joblib.Parallel
) -> float:
    """
    Processes a single raw time-series sequence (256x8) through the hybrid pipeline 
    and returns the anomaly probability (0.0 to 1.0).
    """
    
    # 1. Prepare Raw Data (Assumes input is a single DataFrame/Batch)
    if raw_sequence_df.shape[0] != SEQ_LEN_RAW:
        raise ValueError(f"Input sequence must have exactly {SEQ_LEN_RAW} rows.")
        
    # Convert to numpy array (N, L, F) where N=1
    raw_sequence_batch = raw_sequence_df[COLUMNS].values[np.newaxis, :, :] # Shape (1, 256, 8)

    # 2. Handcrafted Features
    handcrafted_feats = extract_handcrafted_features(raw_sequence_batch[0, :, :])
    X_feats_handcrafted = handcrafted_feats.values.reshape(1, -1) # Shape (1, Feature_Count)

    # 3. LSTM Embeddings
    # Scale the sequence
    scaled_seq = lstm_scaler.transform(raw_sequence_batch.reshape(-1, N_FEATURES_RAW)).reshape(1, SEQ_LEN_RAW, N_FEATURES_RAW)
    
    # Define input for encoder (L-1 steps)
    lstm_input = scaled_seq[:, :LSTM_INPUT_SEQ_LEN, :] # Shape (1, 255, 8)
    
    # Generate embedding
    X_feats_emb = lstm_encoder.predict(lstm_input, verbose=0) # Shape (1, LSTM_EMBEDDING_DIM)

    # 4. Combine Features
    X_full = np.hstack([X_feats_handcrafted, X_feats_emb])

    # 5. Scale Combined Features
    X_scaled_lgbm = lgbm_scaler.transform(X_full)

    # 6. Predict using LightGBM
    # LightGBM predicts probability by default
    anomaly_probability = lgbm_classifier.predict(X_scaled_lgbm)[0]
    
    return anomaly_probability


# --- EXAMPLE USAGE ---

if __name__ == '__main__':
    # NOTE: This part will fail if the model files are not present in ./hybrid_models/
    try:
        lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler, seq_len = load_all_hybrid()

        # --- CREATE MOCK DATA FOR TESTING ---
        print("\n--- Testing Inference with Mock Data ---")
        # Create a mock DataFrame matching the expected shape (256 rows, 8 columns)
        data = {col: np.random.rand(seq_len) * (10 if 'volt' in col else 1) for col in COLUMNS}
        mock_df = pd.DataFrame(data)
        
        # Ensure 'timestamp' is present and numerical
        mock_df['timestamp'] = np.arange(seq_len)
        
        # Predict
        prob = predict_anomaly(mock_df, lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler)
        
        print(f"\nPrediction Complete:")
        print(f"Input Sequence Size: {mock_df.shape}")
        print(f"Predicted Anomaly Probability: {prob:.4f}")

    except FileNotFoundError as e:
        print(f"\n🛑 Setup Error: {e}")
        print("Please ensure your trained model files are correctly placed in the './hybrid_models/' directory.")
        
    except ValueError as e:
        print(f"\n🛑 Data Error: {e}")
        
    except Exception as e:
        print(f"\n🛑 An unexpected error occurred: {e}")
