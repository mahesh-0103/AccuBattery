from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import joblib
import uvicorn
import io
from io import BytesIO
from fpdf import FPDF
from tensorflow.keras.models import load_model # Keep the import

# Local imports (Assuming anomaly_core.py and model_loader.py are accessible)
from anomaly_core import (
    compute_physics_features,
    determine_flag,
    analyze_sequence
)
from model_loader import load_all # Removed prepare_sequences as it's not used in this consolidated file

# -----------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------
app = FastAPI(
    title="AccuBattery Backend",
    description="EV Battery ML + Physics Hybrid Anomaly Engine",
    version="1.1"
)

# MODIFICATION: Use your specific Vercel URL for security and connectivity.
VERCEL_FRONTEND_URL = "https://accu-battery-rn5xqqj4z-maheswaran-ss-projects.vercel.app" 

app.add_middleware(
    CORSMiddleware,
    # Crucial: Allow requests only from your Vercel URL and localhost
    allow_origins=[VERCEL_FRONTEND_URL, "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD MODEL + SCALER (Using the robust load_all function)
# -----------------------------------------------------
model, scaler, seq_len = load_all()

@app.get("/")
def root():
    return {"status": "AccuBattery backend running"}

# -----------------------------------------------------
# PROCESS CSV ENDPOINT (Copied from api.py: /predict_csv)
# -----------------------------------------------------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        timestamp = df["timestamp"] if "timestamp" in df.columns else None
        df_phy = compute_physics_features(df)
        N = len(df)

        if N < 5:
            return {"error": "CSV must contain at least 5 rows."}

        if N <= seq_len:
            # Physics-only fallback mode (from api.py)
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
                "threshold": 0.30,
                "mode": "physics_only"
            }

        # ML + Physics Hybrid (normal mode - from api.py)
        # NOTE: prepare_sequences must be accessible, which it is via 'model_loader.py'
        from model_loader import prepare_sequences # Ensure it's imported correctly
        seq_data, scaled_df = prepare_sequences(df, scaler, seq_len)

        preds = model.predict(seq_data, verbose=0)

        if preds.ndim == 3:
            preds = preds[:, -1, :]

        actual_last = seq_data[:, -1, :]
        ml_scores = np.mean((actual_last - preds)**2, axis=1)

        result_df = analyze_sequence(
            scaled_df.iloc[seq_len:].reset_index(drop=True),
            ml_scores,
            df_phy.iloc[seq_len:].reset_index(drop=True)
        )

        if timestamp is not None:
            result_df.insert(0, "timestamp", timestamp.iloc[seq_len:].values)

        return {
            "data": result_df.to_dict(orient="records"),
            "rows": len(result_df),
            "threshold": 0.30,
            "mode": "hybrid"
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------
# PDF EXPORT ENDPOINT (Copied from api.py: /export_pdf)
# -----------------------------------------------------
@app.post("/export_pdf")
async def export_pdf(data: dict):
    # This logic comes from the API.py PDF export function
    rows = data.get("rows", [])
    summary = data.get("summary", {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AccuBattery Anomaly Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 8, f"Total Samples: {summary.get('total_rows', '-')}", ln=True)
    pdf.cell(0, 8, f"Total Anomalies: {summary.get('total_anomalies', '-')}", ln=True)
    pdf.cell(0, 8, f"Threshold Used: {summary.get('threshold', '-')}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Results Table:", ln=True)

    pdf.set_font("Arial", "", 10)

    for row in rows[:100]:
        line = ", ".join(f"{k}: {v}" for k, v in row.items())
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
