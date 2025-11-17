from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from io import BytesIO

# Local imports
from anomaly_core import (
    compute_physics_features,
    compute_final_score,
    determine_flag,
    analyze_sequence
)
from model_loader import load_all, prepare_sequences


app = FastAPI(
    title="AccuBattery Backend",
    description="EV Battery ML + Physics Hybrid Anomaly Engine",
    version="1.1"
)

# Load model, scaler, seq_len ONCE
model, scaler, seq_len = load_all()


@app.get("/")
def root():
    return {"status": "AccuBattery backend running"}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        # Save timestamp (if exists)
        timestamp = df["timestamp"] if "timestamp" in df.columns else None

        # ----------------------------
        # 1) Physics scoring ALWAYS WORKS
        # ----------------------------
        df_phy = compute_physics_features(df)

        N = len(df)

        # ----------------------------
        # 2) If too few rows, skip ML entirely
        # ----------------------------
        if N < 5:
            return {"error": "CSV must contain at least 5 rows."}

        if N <= seq_len:
            # Physics-only fallback mode
            final_scores = df_phy["physics_score"].values
            flags = determine_flag(final_scores)

            df_final = df.copy()
            df_final["anomaly_score"] = final_scores
            df_final["anomaly_flag"] = flags

            # Add timestamp back
            if timestamp is not None:
                df_final.insert(0, "timestamp", timestamp.values)

            return {
                "data": df_final.to_dict(orient="records"),
                "rows": len(df_final),
                "threshold": 0.30,
                "mode": "physics_only"
            }

        # ----------------------------
        # 3) ML + Physics Hybrid (normal mode)
        # ----------------------------
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

from fpdf import FPDF
from fastapi.responses import StreamingResponse
import io

@app.post("/export_pdf")
async def export_pdf(data: dict):
    """
    Expects JSON containing:
    {
        "rows": [...],
        "summary": {
            "total_rows": int,
            "total_anomalies": int,
            "threshold": float
        }
    }
    """

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

    for row in rows[:100]:  # cap rows for readability
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
