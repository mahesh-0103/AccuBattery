import numpy as np
import pandas as pd
import joblib 
import os
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential

from typing import Tuple, List
from pathlib import Path

# --- CONFIGURATION FOR HYBRID MODEL ARTIFACTS ---
# These names must match the files saved by your training script
LSTM_ENCODER_MODEL_NAME = "lstm_feature_encoder.h5"
LGBM_CLASSIFIER_MODEL_NAME = "lgbm_anomaly_classifier.txt"
LSTM_SEQUENCE_SCALER_NAME = "lstm_sequence_scaler.pkl"
LGBM_FEATURE_SCALER_NAME = "lgbm_feature_scaler.pkl"

# --- FEATURE DEFINITIONS (Must match the training script) ---
COLUMNS: List[str] = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']
N_FEATURES_RAW = len(COLUMNS) # 8 features
SEQ_LEN_RAW = 256
LSTM_INPUT_SEQ_LEN = SEQ_LEN_RAW - 1

# --- 1. HANDCRAFTED FEATURE ENGINEERING ---

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


# --- 2. LOADER FUNCTION ---

def load_all_hybrid() -> Tuple[Sequential, lgb.Booster, joblib.Parallel, joblib.Parallel]:
    """
    Loads all four required artifacts for the hybrid model pipeline.
    """
    # Assuming the artifacts are in a 'hybrid_models' folder relative to the script's execution path
    BASE_DIR = Path(os.getcwd())
    MODELS_DIR = BASE_DIR / "hybrid_models" 
    
    LSTM_ENCODER_PATH = MODELS_DIR / LSTM_ENCODER_MODEL_NAME
    LGBM_CLASSIFIER_PATH = MODELS_DIR / LGBM_CLASSIFIER_MODEL_NAME
    LSTM_SCALER_PATH = MODELS_DIR / LSTM_SEQUENCE_SCALER_NAME
    LGBM_SCALER_PATH = MODELS_DIR / LGBM_FEATURE_SCALER_NAME

    # --- PATH CHECK AND LOADING ---
    for path in [LSTM_ENCODER_PATH, LGBM_CLASSIFIER_PATH, LSTM_SCALER_PATH, LGBM_SCALER_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Artifact NOT found: {path.resolve()}. Ensure all files are in the './hybrid_models/' directory.")

    # 1. Load Keras LSTM Encoder (compile=False bypasses optimizer loading, increasing compatibility)
    print(f"🧠 Loading LSTM Encoder from: {LSTM_ENCODER_PATH}")
    lstm_encoder = load_model(LSTM_ENCODER_PATH, compile=False) 
    
    # 2. Load LightGBM Classifier
    print(f"🌳 Loading LightGBM Classifier from: {LGBM_CLASSIFIER_PATH}")
    lgbm_classifier = lgb.Booster(model_file=str(LGBM_CLASSIFIER_PATH))

    # 3. Load Scalers
    print(f"📐 Loading scalers...")
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    lgbm_scaler = joblib.load(LGBM_SCALER_PATH)

    print("\n✨ All Hybrid Components Loaded Successfully!")

    return lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler


# --- 3. INFERENCE FUNCTION ---

def predict_anomaly(
    raw_sequence_df: pd.DataFrame, 
    lstm_encoder: Sequential, 
    lgbm_classifier: lgb.Booster, 
    lstm_scaler: joblib.Parallel, 
    lgbm_scaler: joblib.Parallel
) -> float:
    """
    Processes a single raw time-series sequence through the hybrid pipeline 
    and returns the anomaly probability (0.0 to 1.0).
    """
    
    # 1. Validation and Raw Data Preparation
    if raw_sequence_df.shape[0] != SEQ_LEN_RAW:
        raise ValueError(f"Input sequence must have exactly {SEQ_LEN_RAW} rows.")
        
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
    anomaly_probability = lgbm_classifier.predict(X_scaled_lgbm)[0]
    
    return anomaly_probability
