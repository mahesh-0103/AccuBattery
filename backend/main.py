from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model # Keep the import, but rely on dependency pinning
from io import BytesIO
from fpdf import FPDF
import uvicorn
import io # Added to support BytesIO, though often imported with io

# -----------------------------------------------------
# FASTAPI SETUP
# -----------------------------------------------------
app = FastAPI()

# IMPORTANT MODIFICATION: Use your specific Vercel URL for security and connectivity.
# Replace the placeholder URL below with your actual Vercel URL!
VERCEL_FRONTEND_URL = "https://accu-battery-rn5xqqj4z-maheswaran-ss-projects.vercel.app" 

app.add_middleware(
    CORSMiddleware,
    # MODIFICATION: Specify origins allowed to access this API
    allow_origins=[VERCEL_FRONTEND_URL, "http://localhost:3000"],
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
# NOTE: The previous TensorFlow error (ValueError) is fixed by pinning 
# the 'tensorflow' and 'keras' versions in your 'requirements.txt' file.
model = load_model(MODEL_PATH)

print("📏 Loading scaler...")
scaler = joblib.load(SCALER_PATH)

SEQ_LEN = 256


# -----------------------------------------------------
# PDF EXPORT HELPER (NO external file needed)
# -----------------------------------------------------
def make_pdf(df: pd.DataFrame):
    # ... (function content remains the same)
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "AccuBattery Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for col in df.columns:
        # Assuming you want to print more than just the first row for export
        # This implementation remains simple as per your original file.
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
    # ... (function content remains the same)
    df = pd.read_csv(file.file)

    # scale features
    values = scaler.transform(df)

    # create LSTM sequences
    sequences = [
        values[i:i+SEQ_LEN] for i in range(len(values) - SEQ_LEN)
    ]
    sequences = np.array(sequences)

    # reconstruction
    # NOTE: Set verbose=0 for cleaner production logs
    reconstructed = model.predict(sequences, verbose=0) 

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
    # ... (function content remains the same)
    df = pd.DataFrame(data)
    pdf_buffer = make_pdf(df)
    return StreamingResponse(pdf_buffer, media_type="application/pdf")


# -----------------------------------------------------
# LOCAL DEV ENTRYPOINT
# -----------------------------------------------------
if __name__ == "__main__":
    # NOTE: Railway ignores this section due to the Procfile, but it's correct for local testing.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
