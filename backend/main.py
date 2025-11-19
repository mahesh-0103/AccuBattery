# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from io import BytesIO
from fpdf import FPDF
import uvicorn
import io
import os

# -----------------------------------------------------
# CONFIGURATION AND MOCK UTILITY FUNCTIONS
# -----------------------------------------------------

# --- Constants (MUST match your trained model configuration) ---
FEATURES = ['voltage', 'current', 'temperature', 'soc', 'soh', 'impedance', 'd_voltage', 'd_current']
SEQ_LEN = 128 # Sequence length your model was trained on
ANOMALY_THRESHOLD = 0.30 
ML_WEIGHT = 0.7 
PHYSICS_WEIGHT = 1.0 - ML_WEIGHT

# --- UTILITY: Model Loading (Adjusted for your 'models' folder and filenames) ---

def load_all():
    """
    Loads the main anomaly detection model (LSTM Encoder) and its sequence scaler,
    using the file paths in the 'models/' folder.
    """
    try:
        # Suppress TensorFlow warnings related to CPU features
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load the Keras/TF Model (your lstm_feature_encoder.h5)
        model = load_model("models/lstm_feature_encoder.h5", compile=False)
        print("Loaded: LSTM Feature Encoder (main model)")

        # Load the data scaler (your lstm_sequence_scaler.pkl)
        scaler = joblib.load("models/lstm_sequence_scaler.pkl")
        print("Loaded: LSTM Sequence Scaler")

        return model, scaler, SEQ_LEN
    except Exception as e:
        # Raise HTTPException for the API to handle the error at startup
        raise HTTPException(status_code=500, detail=f"Model loading failed! Check if files are in the 'models/' folder: {e}")


def prepare_sequences(df: pd.DataFrame, scaler, seq_len: int):
    """Scales the DataFrame and creates overlapping sequences for prediction."""
    
    if not all(f in df.columns for f in FEATURES):
        missing = set(FEATURES) - set(df.columns)
        raise ValueError(f"Input CSV is missing required features: {missing}")
        
    scaled_data = scaler.transform(df[FEATURES])
    scaled_df = pd.DataFrame(scaled_data, columns=FEATURES)

    sequences = []
    # Stop the range so the last sequence ends exactly at the end of the data
    for i in range(len(scaled_data) - seq_len + 1): 
        sequences.append(scaled_data[i:i + seq_len])
        
    return np.array(sequences), scaled_df

# --- UTILITY: Anomaly Core Functions ---

def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates physics-based features and a simple 'physics_score'."""
    df_phy = pd.DataFrame()

    if 'current' not in df.columns or 'voltage' not in df.columns:
        df_phy['physics_score'] = 0.0
        return df_phy

    df_phy['power'] = df['voltage'] * df['current']
    df_phy['d_voltage_dt'] = df['voltage'].diff().fillna(0)
    
    # Simple Physics Score: based on rapid voltage change
    df_phy['physics_score'] = np.abs(df_phy['d_voltage_dt']) * 10
    df_phy['physics_score'] = df_phy['physics_score'].clip(upper=1.0)
        
    return df_phy

def determine_flag(final_scores: np.ndarray) -> list:
    """Assigns a binary flag based on the final anomaly score and the threshold."""
    return (final_scores > ANOMALY_THRESHOLD).astype(int).tolist()

def analyze_sequence(scaled_df: pd.DataFrame, ml_scores: np.ndarray, df_phy: pd.DataFrame) -> pd.DataFrame:
    """Combines ML and Physics scores and generates the final results DataFrame."""
    
    # 1. Normalize ML Scores
    max_ml_score = np.max(ml_scores) if np.max(ml_scores) > 0 else 1.0
    normalized_ml_scores = ml_scores / max_ml_score
    
    # 2. Hybrid Score Calculation (Max of weighted scores)
    ml_contribution = normalized_ml_scores * ML_WEIGHT
    phy_contribution = df_phy['physics_score'].values * PHYSICS_WEIGHT
    
    final_scores = np.maximum(ml_contribution, phy_contribution)
    
    # 3. Determine Flag
    flags = determine_flag(final_scores)

    # 4. Create Final Results DataFrame
    result_df = scaled_df.copy()
    result_df["ml_score"] = normalized_ml_scores
    result_df["physics_score"] = df_phy["physics_score"].values
    result_df["anomaly_score"] = final_scores
    result_df["anomaly_flag"] = flags
    
    return result_df


# -----------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------
app = FastAPI(
    title="AccuBattery Backend",
    description="EV Battery ML + Physics Hybrid Anomaly Engine",
    version="1.1"
)

# CRITICAL MODIFICATION: Use your specific Vercel URL
VERCEL_FRONTEND_URL = "https://accu-battery-rn5xqqj4z-maheswaran-ss-projects.vercel.app" 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[VERCEL_FRONTEND_URL, "http://localhost:8000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD MODEL + SCALER (Executed on startup)
# -----------------------------------------------------
model = None
scaler = None

@app.on_event("startup")
def startup_event():
    global model, scaler
    try:
        model, scaler, _ = load_all()
        print(f"Model and Scaler loaded successfully. SEQ_LEN: {SEQ_LEN}")
    except HTTPException as e:
        # Re-raise the HTTPException to display model loading error
        print(f"FATAL: Application started, but models are unavailable: {e.detail}")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during startup: {e}")


@app.get("/")
def root():
    return {"status": "AccuBattery backend running", "model_loaded": model is not None}


# -----------------------------------------------------
# PROCESS CSV ENDPOINT (Hybrid Logic)
# -----------------------------------------------------
@app.post("/predict_csv") 
async def predict_csv(file: UploadFile = File(...)):
    global model, scaler, SEQ_LEN
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model service unavailable. Models failed to load at startup.")

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        timestamp = df["timestamp"] if "timestamp" in df.columns else None
        
        df_phy = compute_physics_features(df)
        N = len(df)

        if N < 5:
            return {"error": "CSV must contain at least 5 rows."}

        # 1. Physics-Only Fallback (Data too short for ML sequence)
        if N < SEQ_LEN:
            final_scores = df_phy["physics_score"].values
            flags = determine_flag(final_scores)

            df_final = df.copy()
            df_final["anomaly_score"] = final_scores
            df_final["anomaly_flag"] = flags

            if timestamp is not None:
                df_final.insert(0, "timestamp", timestamp.values)

            return {
                "data": df_final.to_dict(orient="records"),
                "rows": len(df_final),
                "threshold": ANOMALY_THRESHOLD,
                "mode": "physics_only"
            }

        # 2. Hybrid ML + Physics (Normal Mode)
        seq_data, scaled_df = prepare_sequences(df, scaler, SEQ_LEN)

        preds = model.predict(seq_data, verbose=0)

        # Calculate error based on the reconstruction of the last time step
        if preds.ndim == 3:
            preds = preds[:, -1, :] 

        actual_last = seq_data[:, -1, :]
        ml_scores = np.mean((actual_last - preds)**2, axis=1)

        # The first prediction corresponds to the row at index SEQ_LEN - 1
        result_df = analyze_sequence(
            scaled_df.iloc[SEQ_LEN - 1:].reset_index(drop=True),
            ml_scores,
            df_phy.iloc[SEQ_LEN - 1:].reset_index(drop=True) 
        )

        if timestamp is not None:
            result_df.insert(0, "timestamp", timestamp.iloc[SEQ_LEN - 1:].values)

        return {
            "data": result_df.to_dict(orient="records"),
            "rows": len(result_df),
            "threshold": ANOMALY_THRESHOLD,
            "mode": "hybrid"
        }

    except ValueError as ve:
         raise HTTPException(status_code=400, detail=f"Data validation error: {ve}")
    except Exception as e:
        print(f"Prediction failed with unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


# -----------------------------------------------------
# PDF EXPORT ENDPOINT
# -----------------------------------------------------
@app.post("/export_pdf")
async def export_pdf(data: dict):
    rows = data.get("rows", [])
    summary = data.get("summary", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AccuBattery Anomaly Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)

    # Summary Section
    pdf.cell(0, 8, f"Total Samples Analyzed: {summary.get('total_rows', '-')}", ln=True)
    pdf.cell(0, 8, f"Total Anomalies Detected: {summary.get('total_anomalies', '-')}", ln=True)
    pdf.cell(0, 8, f"Anomaly Threshold Used: {summary.get('threshold', ANOMALY_THRESHOLD)}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Results (First 100 Rows):", ln=True)

    pdf.set_font("Arial", "", 10)

    # Detail Table (limited to 100 rows)
    for row in rows[:100]:
        line = ", ".join(f"{k}: {v:.4f}" if isinstance(v, (float, np.float32, np.float64)) else f"{k}: {v}" for k, v in row.items())
        pdf.multi_cell(0, 5, line)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=accubattery_report.pdf"},
    )


# -----------------------------------------------------
# LOCAL DEV ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
# LOCAL DEV ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

