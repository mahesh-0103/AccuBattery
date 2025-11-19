# anomaly_core.py

import numpy as np
import pandas as pd

# -------------------------------------------------------
# PHYSICS ENGINE (UPDATED with Temperature Features)
# -------------------------------------------------------

def compute_physics_features(df):
    df = df.copy()

    # --- Voltage Features (Existing) ---
    volt = df["volt"].values
    max_v = df["max_single_volt"].values
    min_v = df["min_single_volt"].values
    I = df["current"].replace(0, 0.001)

    # 1. Voltage imbalance
    df["voltage_imbalance"] = (max_v - min_v)
    df["violation_voltage_imb"] = (df["voltage_imbalance"] > 0.05).astype(int)

    # 2. Internal resistance
    df["internal_R"] = np.abs((max_v - min_v) / I)
    df["violation_internal_R"] = (df["internal_R"] > 0.1).astype(int)

    # 3. SOC Voltage Residual
    nominal_v = 3.60 + (df["soc"] / 100) * (4.20 - 3.60)
    df["soc_voltage_residual"] = np.abs(df["volt"] - nominal_v)
    df["violation_soc_resid"] = (df["soc_voltage_residual"] > 0.05).astype(int)

    # --- Temperature Features (NEW) ---
    temp = df["temp"].values # Assuming 'temp' is the average temperature
    max_t = df["max_single_temp"].values # Assuming max cell temperature
    min_t = df["min_single_temp"].values # Assuming min cell temperature

    # 4. Temperature Rate of Change (dV/dt or dT/dt)
    # Rapid change suggests thermal runaway or aggressive charging/discharging
    df["temp_rate_of_change"] = np.abs(pd.Series(temp).diff().fillna(0))
    df["violation_temp_rate"] = (df["temp_rate_of_change"] > 0.5).astype(int) # E.g., > 0.5 deg C per sample

    # 5. Temperature Imbalance
    # Large spread suggests poor thermal management or localized issues
    df["temp_imbalance"] = (max_t - min_t)
    df["violation_temp_imb"] = (df["temp_imbalance"] > 3.0).astype(int) # E.g., > 3.0 deg C spread

    # 6. Final Physics Score (UPDATED WEIGHTING)
    # The sum of weighted, normalized scores
    df["physics_score"] = (
        # Voltage contributions (total 0.5)
        df["voltage_imbalance"] * 0.2 +
        df["internal_R"] * 0.2 +
        df["soc_voltage_residual"] * 0.1 +
        # Temperature contributions (total 0.5)
        df["temp_rate_of_change"].clip(upper=1.0) * 0.3 + # Clip to prevent single large spikes dominating
        df["temp_imbalance"] * 0.2
    )

    # Ensure the combined score is capped at a reasonable value (e.g., 1.0)
    df["physics_score"] = df["physics_score"].clip(upper=1.0)

    return df


# -------------------------------------------------------
# FINAL SCORE FUSION (No Change Required)
# -------------------------------------------------------

def compute_final_score(ml_scores, physics_scores):
    # Normalize both scores (0 to 1)
    ml_norm = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min() + 1e-9)
    phy_norm = (physics_scores - physics_scores.min()) / (physics_scores.max() - physics_scores.min() + 1e-9)
    
    # Weighted average (60% ML, 40% Physics)
    return 0.6 * ml_norm + 0.4 * phy_norm


def determine_flag(scores, threshold=0.30):
    return ["Anomaly" if s > threshold else "Normal" for s in scores]


# -------------------------------------------------------
# FUSION OUTPUT STRUCTURE (No Change Required)
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
