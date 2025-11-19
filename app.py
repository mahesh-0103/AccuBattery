import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF # fpdf is not strictly needed anymore but kept for safety

# -------------------------------------------
# CONFIG (UPDATED)
# -------------------------------------------
# Use the deployed Render URL for the prediction endpoint
RENDER_BASE_URL = "https://accubattery.onrender.com"
PREDICT_ENDPOINT = f"{RENDER_BASE_URL}/predict_csv"
EXPORT_PDF_ENDPOINT = f"{RENDER_BASE_URL}/export_pdf"

st.set_page_config(
    page_title="AccuBattery Dashboard",
    layout="wide",
    page_icon="🔋",
)

# -------------------------------------------
# CUSTOM CSS FOR POLISHED UI
# -------------------------------------------
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #111;
        color: white;
        border: 1px solid #333;
        box-shadow: 0px 0px 8px rgba(0,0,0,0.4);
    }
    .metric-header {
        font-size: 16px;
        font-weight: 600;
        opacity: 0.7;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: -5px;
    }
    .section-title {
        font-size: 22px !important;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: -10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# HEADER
# -------------------------------------------
st.title("🔋 AccuBattery — EV Battery Anomaly Insight Engine")

unit_mode = st.radio("Select Unit System:", ["Metric (°C, V, A)", "Imperial (°F, kW)"], horizontal=True)

uploaded = st.file_uploader("📤 Upload Battery CSV Data", type=["csv"])

# -------------------------------------------
# UNIT CONVERSION
# -------------------------------------------
def convert_units(df, mode):
    df = df.copy()
    if mode == "Imperial (°F, kW)":
        # Note: Columns were assumed as 'max_temp', 'min_temp', 'volt'. 
        # Adjust column names if your final DataFrame uses different names (e.g., 'temperature', 'voltage').
        if "max_temp" in df.columns:
             df["max_temp"] = df["max_temp"] * 9/5 + 32
        if "min_temp" in df.columns:
            df["min_temp"] = df["min_temp"] * 9/5 + 32
        if "volt" in df.columns:
             df["volt"] = df["volt"] * 0.001 
    return df

# -------------------------------------------
# PDF EXPORTER (UPDATED TO USE BACKEND ENDPOINT)
# -------------------------------------------
def get_pdf_report(df, threshold, total_anom):
    """Calls the backend's /export_pdf endpoint to generate and download the PDF."""
    
    # 1. Prepare data for the backend
    payload = {
        "rows": df.to_dict(orient="records"),
        "summary": {
            "total_rows": len(df),
            "total_anomalies": total_anom,
            "threshold": threshold,
        }
    }

    try:
        # 2. Call the backend
        res = requests.post(EXPORT_PDF_ENDPOINT, json=payload)
        
        if res.status_code == 200:
            # 3. Handle the binary PDF response
            pdf_bytes = res.content
            
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name="accubattery_report.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated and ready for download.")
            
        else:
            st.error(f"Failed to generate PDF. Backend returned status: {res.status_code}")
            try:
                # Try to show JSON error if available
                st.json(res.json())
            except:
                st.write(f"Raw response: {res.text}")


    except requests.exceptions.RequestException as e:
        st.error(f"Network error while contacting PDF service: {e}")


# -------------------------------------------
# MAIN PANEL
# -------------------------------------------
if uploaded:
    with st.spinner("📡 Processing CSV via backend..."):
        # The request object needs to be sent as 'file' in a dictionary
        res = requests.post(PREDICT_ENDPOINT, files={"file": uploaded})
        result = res.json()

    if "error" in result:
        st.error(f"Backend Error: {result['error']}")
        st.stop()
    
    # Safely extract data and convert to DataFrame
    df = pd.DataFrame(result.get("data", []))
    if df.empty:
        st.error("Received empty data from the backend. Check the input file features.")
        st.stop()

    df = convert_units(df, unit_mode)
    
    # Calculate key metrics
    total_anom = df[df["anomaly_flag"].astype(str).str.contains("Anomaly")].shape[0]
    total_rows = result.get('rows', len(df))
    threshold = result.get('threshold', 0.30)

    # -------------------------------------------
    # TOP METRIC CARDS
    # -------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-card'><div class='metric-header'>Samples</div>"
                    f"<div class='metric-value'>{total_rows}</div></div>", 
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'><div class='metric-header'>Threshold</div>"
                    f"<div class='metric-value'>{threshold:.2f}</div></div>", 
                    unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'><div class='metric-header'>Anomalies</div>"
                    f"<div class='metric-value'>{total_anom}</div></div>", 
                    unsafe_allow_html=True)

    with col4:
        pct = (total_anom / total_rows) * 100 if total_rows > 0 else 0
        st.markdown("<div class='metric-card'><div class='metric-header'>Anomaly %</div>"
                    f"<div class='metric-value'>{pct:.1f}%</div></div>", 
                    unsafe_allow_html=True)

    # --- VISUALIZATIONS ---

    # Anomaly Score Trend
    st.markdown("<div class='section-title'>📈 Anomaly Score Trend</div>", unsafe_allow_html=True)
    fig = px.line(df, y="anomaly_score", title="Anomaly Score Over Time Index")
    fig.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text=f"Threshold ({threshold:.2f})")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # HEATMAP
    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    
    # DISTRIBUTION PLOTS
    st.markdown("<div class='section-title'>📊 Distribution of Key Features</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        # Check for column existence before plotting
        if "volt" in df.columns and "anomaly_flag" in df.columns:
            fig_hist1 = px.histogram(df, x="volt", color="anomaly_flag", title="Voltage Distribution")
            st.plotly_chart(fig_hist1, use_container_width=True)
        else:
            st.warning("Cannot plot Voltage: required columns ('volt', 'anomaly_flag') not found in results.")

    with colB:
        if "max_temp" in df.columns and "anomaly_flag" in df.columns:
            fig_hist2 = px.histogram(df, x="max_temp", color="anomaly_flag", title="Max Temperature Distribution")
            st.plotly_chart(fig_hist2, use_container_width=True)
        else:
            st.warning("Cannot plot Max Temp: required columns ('max_temp', 'anomaly_flag') not found in results.")
            
    st.markdown("---")

    # SCATTER MAP
    st.markdown("<div class='section-title'>⚡ Voltage vs Current Scatter</div>", unsafe_allow_html=True)
    if "current" in df.columns and "volt" in df.columns and "anomaly_flag" in df.columns:
        fig_scatter = px.scatter(df, x="current", y="volt", color="anomaly_flag", 
                                 title="Current vs Voltage colored by Anomaly Flag")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Cannot plot Scatter: required columns ('current', 'volt', 'anomaly_flag') not found in results.")
        
    st.markdown("---")

    # FINAL OUTPUT TABLE
    st.markdown("<div class='section-title'>📄 Processed Output Table</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------
    # DOWNLOAD BUTTONS (UPDATED PDF LOGIC)
    # -------------------------------------------
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.download_button(
            "⬇ Download Processed CSV", 
            df.to_csv(index=False).encode('utf-8'), 
            "accubattery_output.csv",
            mime="text/csv"
        )

    with col_dl2:
        # Trigger the backend PDF generation on button click
        if st.button("📄 Generate PDF Report"):
            get_pdf_report(df, threshold, total_anom)
