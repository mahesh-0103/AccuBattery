import uvicorn
import numpy as np
import pandas as pd
import io
import os
from fpdf import FPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# --- CRITICAL IMPORTS FOR HYBRID MODEL ---
# Import the custom loader and prediction functions from your local file
# Ensure 'hybrid_model_loader.py' is the correct name of your loader file
from hybrid_model_loader import load_all_hybrid, predict_anomaly 
from hybrid_model_loader import SEQ_LEN_RAW, COLUMNS # Constants for sequence length and features

# --- CONFIGURATION (UPDATED) ---
# NOTE: SEQ_LEN is now SEQ_LEN_RAW (256) and FEATURES are defined by COLUMNS (8)
ANOMALY_THRESHOLD = 0.50 # Using a default threshold; this can be adjusted

# --- GLOBAL MODEL ARTIFACTS ---
lstm_encoder = None
lgbm_classifier = None
lstm_scaler = None
lgbm_scaler = None

# -----------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------
app = FastAPI(
    title="AccuBattery Hybrid Backend",
    description="EV Battery ML + Physics Hybrid Anomaly Engine (LSTM Embeddings + LightGBM)",
    version="2.0"
)

# CRITICAL MODIFICATION: Use your specific Vercel URL for security and connectivity.
VERCEL_FRONTEND_URL = "https://accu-battery-rn5xqqj4z-maheswaran-ss-projects.vercel.app" 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[VERCEL_FRONTEND_URL, "http://localhost:8000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD HYBRID MODEL ARTIFACTS (Executed on startup)
# -----------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Loads all four hybrid model components and scalers."""
    global lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler

    # Suppress TensorFlow warnings related to CPU features
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        # load_all_hybrid loads the Keras Encoder, LightGBM Classifier, and two Scalers
        lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler = load_all_hybrid()
        print(f"Model and Scalers loaded successfully. Sequence Length: {SEQ_LEN_RAW}")
    except Exception as e:
        print(f"FATAL: Application will start but models are unavailable: {e}")
        # Set all to None to ensure /predict_csv raises a 503 error safely
        lstm_encoder = lgbm_classifier = lstm_scaler = lgbm_scaler = None


@app.get("/")
def root():
    return {"status": "AccuBattery hybrid backend running", "model_loaded": lstm_encoder is not None}


# -----------------------------------------------------
# PDF EXPORT HELPER (Uses the provided FPDF logic)
# -----------------------------------------------------
# NOTE: This function is the placeholder from your original file, kept for continuity.
def make_pdf(df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "AccuBattery Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    
    # Only prints the first row's data as a placeholder
    if not df.empty:
        for col in df.columns:
            pdf.cell(0, 8, f"{col}: {df[col].iloc[0]}", ln=True)
            
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


# -----------------------------------------------------
# PROCESS CSV ENDPOINT (Hybrid Prediction Logic)
# -----------------------------------------------------
@app.post("/predict_csv") 
async def predict_csv(file: UploadFile = File(...)):
    global lstm_encoder, lgbm_classifier, lstm_scaler, lgbm_scaler
    
    # Check for model availability
    if lstm_encoder is None or lgbm_classifier is None:
        raise HTTPException(status_code=503, detail="Model service unavailable. Models failed to load at startup.")

    try:
        content = await file.read()
        df_raw = pd.read_csv(io.BytesIO(content))

        # Check for minimum rows needed for prediction
        if len(df_raw) < SEQ_LEN_RAW:
            raise HTTPException(status_code=400, detail=f"Input CSV must contain at least {SEQ_LEN_RAW} rows for hybrid prediction.")

        # Ensure all required columns for the hybrid model are present
        missing_cols = set(COLUMNS) - set(df_raw.columns)
        if missing_cols:
            raise ValueError(f"Input CSV is missing required features: {missing_cols}")
            
        # Initialize lists to store rolling window predictions
        anomaly_scores = []
        
        # --- Rolling Window Prediction Loop (Hybrid Mode) ---
        N = len(df_raw)
        
        # Start the rolling window from the first possible prediction point
        # The loop runs (N - SEQ_LEN_RAW + 1) times, matching the number of scores
        for i in range(N - SEQ_LEN_RAW + 1):
            # Slice out the sequence (e.g., rows 0 to 255, then 1 to 256, etc.)
            sequence_slice = df_raw.iloc[i : i + SEQ_LEN_RAW].copy()
            
            # Predict anomaly for this single sequence (the score corresponds to the last row of the slice)
            # This single function handles all scaling, feature engineering, and LGBM prediction.
            score = predict_anomaly(
                sequence_slice, 
                lstm_encoder, 
                lgbm_classifier, 
                lstm_scaler, 
                lgbm_scaler
            )
            anomaly_scores.append(score)

        # Convert scores to numpy array
        final_scores = np.array(anomaly_scores)
        flags = (final_scores > ANOMALY_THRESHOLD).astype(int).tolist()

        # --- Generate Final Results DataFrame ---
        
        # The first prediction score corresponds to the (SEQ_LEN_RAW - 1) index of the raw data.
        # We only have (N - SEQ_LEN_RAW + 1) predictions.
        start_index = SEQ_LEN_RAW - 1
        
        result_df = df_raw.iloc[start_index:].reset_index(drop=True).copy()
        
        # Attach final prediction results
        result_df["anomaly_score"] = final_scores
        result_df["anomaly_flag"] = flags
        
        # Re-attach timestamp if it was in the original data
        if "timestamp" in df_raw.columns:
            result_df.insert(0, "timestamp", df_raw["timestamp"].iloc[start_index:].values)
        
        return {
            "data": result_df.to_dict(orient="records"),
            "rows": len(result_df),
            "threshold": ANOMALY_THRESHOLD,
            "mode": "hybrid_lgbm"
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data processing error: {ve}")
    except Exception as e:
        print(f"Prediction failed with unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


# -----------------------------------------------------
# PDF EXPORT ENDPOINT (Kept as is)
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

    # Detail Table (limited to 100 rows to prevent excessively large PDFs)
    for row in rows[:100]:
        # Format the row data for PDF display
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
