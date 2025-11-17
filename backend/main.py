from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from io import BytesIO
from fpdf import FPDF
import uvicorn

# -----------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow frontend (Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# LOAD MODEL + SCALER
# -----------------------------------------------------
MODEL_PATH = "models/lstm_autoencoder_mileage.keras"
SCALER_PATH = "models/scaler.pkl"

print("🔋 Loading model...")
model = load_model(MODEL_PATH)

print("📏 Loading scaler...")
scaler = joblib.load(SCALER_PATH)

SEQ_LEN = 256


# -----------------------------------------------------
# PDF EXPORT HELPER (NO external file needed)
# -----------------------------------------------------
def make_pdf(df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "AccuBattery Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for col in df.columns:
        pdf.cell(0, 8, f"{col}: {df[col].iloc[0]}", ln=True)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    return buffer


# -----------------------------------------------------
# PROCESS CSV ENDPOINT
# -----------------------------------------------------
@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # scale features
    values = scaler.transform(df)

    # create LSTM sequences
    sequences = [
        values[i:i+SEQ_LEN] for i in range(len(values) - SEQ_LEN)
    ]
    sequences = np.array(sequences)

    # reconstruction
    reconstructed = model.predict(sequences)

    mse = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))

    # pad anomaly scores to match df length
    padded_mse = np.pad(mse, (SEQ_LEN, 0), constant_values=mse[0])

    df["anomaly_score"] = padded_mse

    threshold = float(mse.mean() + 2 * mse.std())

    return {
        "rows": len(df),
        "threshold": threshold,
        "data": df.to_dict(orient="records")
    }


# -----------------------------------------------------
# PDF EXPORT ENDPOINT
# -----------------------------------------------------
@app.post("/export_pdf")
async def export_pdf(data: list):
    df = pd.DataFrame(data)
    pdf_buffer = make_pdf(df)
    return StreamingResponse(pdf_buffer, media_type="application/pdf")


# -----------------------------------------------------
# LOCAL DEV ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
