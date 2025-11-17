import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

# -------------------------------------------
# CONFIG
# -------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000/predict_csv"

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
        df["max_temp"] = df["max_temp"] * 9/5 + 32
        df["min_temp"] = df["min_temp"] * 9/5 + 32
        df["volt"] = df["volt"] * 0.001
    return df

# -------------------------------------------
# PDF EXPORTER
# -------------------------------------------
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(190, 10, txt="AccuBattery - Battery Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(190, 8, txt=f"Total Samples: {len(df)}")
    pdf.multi_cell(190, 8, txt=f"Anomalies: {len(df[df['anomaly_flag']=='Anomaly'])}")

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(190, 10, txt="Sample Data (first 20 rows)", ln=True)

    pdf.set_font("Arial", size=8)
    for i in range(min(20, len(df))):
        row_text = ", ".join([f"{col}: {df.iloc[i][col]}" for col in df.columns])
        pdf.multi_cell(190, 5, row_text)

    output_path = "accubattery_report.pdf"
    pdf.output(output_path)

    return output_path

# -------------------------------------------
# MAIN PANEL
# -------------------------------------------
if uploaded:
    with st.spinner("📡 Processing CSV via backend..."):
        res = requests.post(BACKEND_URL, files={"file": uploaded})
        result = res.json()

    if "error" in result:
        st.error(f"Backend Error: {result['error']}")
        st.stop()

    df = pd.DataFrame(result["data"])
    df = convert_units(df, unit_mode)

    # -------------------------------------------
    # TOP METRIC CARDS
    # -------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-card'><div class='metric-header'>Samples</div>"
                    f"<div class='metric-value'>{result['rows']}</div></div>", 
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'><div class='metric-header'>Threshold</div>"
                    f"<div class='metric-value'>{result['threshold']:.2f}</div></div>", 
                    unsafe_allow_html=True)

    with col3:
        total_anom = df[df["anomaly_flag"]=="Anomaly"].shape[0]
        st.markdown("<div class='metric-card'><div class='metric-header'>Anomalies</div>"
                    f"<div class='metric-value'>{total_anom}</div></div>", 
                    unsafe_allow_html=True)

    with col4:
        pct = (total_anom / len(df)) * 100
        st.markdown("<div class='metric-card'><div class='metric-header'>Anomaly %</div>"
                    f"<div class='metric-value'>{pct:.1f}%</div></div>", 
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Anomaly Score Trend</div>", unsafe_allow_html=True)
    fig = px.line(df, y="anomaly_score", title="")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------
    # HEATMAP
    # -------------------------------------------
    st.markdown("<div class='section-title'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)

    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    # -------------------------------------------
    # DISTRIBUTION PLOTS
    # -------------------------------------------
    st.markdown("<div class='section-title'>📊 Distribution of Key Features</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        fig_hist1 = px.histogram(df, x="volt", color="anomaly_flag")
        st.plotly_chart(fig_hist1, use_container_width=True)

    with colB:
        fig_hist2 = px.histogram(df, x="max_temp", color="anomaly_flag")
        st.plotly_chart(fig_hist2, use_container_width=True)

    # -------------------------------------------
    # SCATTER MAP
    # -------------------------------------------
    st.markdown("<div class='section-title'>⚡ Voltage vs Current Scatter</div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(df, x="current", y="volt", color="anomaly_flag")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------------------
    # FINAL OUTPUT TABLE
    # -------------------------------------------
    st.markdown("<div class='section-title'>📄 Processed Output Table</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # -------------------------------------------
    # DOWNLOAD BUTTONS
    # -------------------------------------------
    st.download_button("⬇ Download CSV", df.to_csv(index=False), "accubattery_output.csv")

    if st.button("📄 Export PDF Report"):
        pdf_path = generate_pdf(df)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="accubattery_report.pdf")
