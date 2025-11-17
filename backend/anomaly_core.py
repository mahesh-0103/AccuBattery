import numpy as np
import pandas as pd

# -------------------------------------------------------
# PHYSICS ENGINE
# -------------------------------------------------------

def compute_physics_features(df):
    df = df.copy()

    volt = df["volt"].values
    max_v = df["max_single_volt"].values
    min_v = df["min_single_volt"].values

    # Voltage imbalance
    df["voltage_imbalance"] = (max_v - min_v)
    df["violation_voltage_imb"] = (df["voltage_imbalance"] > 0.05).astype(int)

    # Internal resistance (avoid divide-by-zero)
    I = df["current"].replace(0, 0.001)
    df["internal_R"] = np.abs((max_v - min_v) / I)
    df["violation_internal_R"] = (df["internal_R"] > 0.1).astype(int)

    # Nominal voltage from SOC
    nominal_v = 3.60 + (df["soc"] / 100) * (4.20 - 3.60)
    df["soc_voltage_residual"] = np.abs(df["volt"] - nominal_v)
    df["violation_soc_resid"] = (df["soc_voltage_residual"] > 0.05).astype(int)

    # Physics score
    df["physics_score"] = (
        df["voltage_imbalance"] * 0.4 +
        df["internal_R"] * 0.4 +
        df["soc_voltage_residual"] * 0.2
    )

    return df


# -------------------------------------------------------
# FINAL SCORE FUSION
# -------------------------------------------------------

def compute_final_score(ml_scores, physics_scores):
    ml_norm = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min() + 1e-9)
    phy_norm = (physics_scores - physics_scores.min()) / (physics_scores.max() - physics_scores.min() + 1e-9)
    return 0.6 * ml_norm + 0.4 * phy_norm


def determine_flag(scores, threshold=0.30):
    return ["Anomaly" if s > threshold else "Normal" for s in scores]


# -------------------------------------------------------
# FUSION OUTPUT STRUCTURE
# -------------------------------------------------------

def analyze_sequence(df_slice, ml_scores, df_phy_slice):
    df_final = df_slice.copy()

    df_final["ml_score"] = ml_scores
    df_final["physics_score"] = df_phy_slice["physics_score"].values

    df_final["anomaly_score"] = compute_final_score(
        df_final["ml_score"].values,
        df_final["physics_score"].values
    )

    df_final["anomaly_flag"] = determine_flag(df_final["anomaly_score"].values)

    return df_final
