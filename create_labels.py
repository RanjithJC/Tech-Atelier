import pandas as pd
import numpy as np

# ============================================================
# LOAD ML DATASET
# ============================================================

df = pd.read_csv("data/ml_dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# CREATE STRESS RISK SCORE
# ============================================================
#
# This is a SYNTHETIC development label.
# It is NOT a clinical diagnosis.
#
# Higher values indicate a greater simulated welfare risk.
# ============================================================

df["stress_risk_score"] = (
    0.20 * (df["workload_score"] / 10) * 100
    + 0.15 * (df["fatigue_score"] / 10) * 100
    + 0.15 * ((8 - df["sleep_hours"]).clip(0, 4) / 4) * 100
    + 0.15 * (df["stress_score"] / 10) * 100
    + 0.10 * ((10 - df["mood_score"]) / 10) * 100
    + 0.10 * (df["emotional_exhaustion"] / 10) * 100
    + 0.05 * (df["deployment_days"] / 120) * 100
    + 0.05 * (df["days_since_last_leave"] / 90) * 100
    + 0.05 * (df["consecutive_duty_days"] / 14) * 100
)

# Add small natural variation
df["stress_risk_score"] += np.random.normal(
    0, 3, len(df)
)

# Keep score between 0 and 100
df["stress_risk_score"] = np.clip(
    df["stress_risk_score"],
    0,
    100
)

df["stress_risk_score"] = df["stress_risk_score"].round(1)


# ============================================================
# CREATE STRESS RISK CATEGORY
# ============================================================

def stress_category(score):

    if score < 30:
        return "Low"

    elif score < 55:
        return "Moderate"

    elif score < 75:
        return "High"

    else:
        return "Critical"


df["stress_risk"] = df["stress_risk_score"].apply(
    stress_category
)


# ============================================================
# CREATE BURNOUT RISK SCORE
# ============================================================

df["burnout_risk_score"] = (
    0.25 * (df["workload_score"] / 10) * 100
    + 0.20 * (df["fatigue_score"] / 10) * 100
    + 0.15 * (df["emotional_exhaustion"] / 10) * 100
    + 0.15 * ((8 - df["sleep_hours"]).clip(0, 4) / 4) * 100
    + 0.10 * (df["consecutive_duty_days"] / 14) * 100
    + 0.10 * (df["deployment_days"] / 120) * 100
    + 0.05 * (df["days_since_last_leave"] / 90) * 100
)

df["burnout_risk_score"] += np.random.normal(
    0,
    3,
    len(df)
)

df["burnout_risk_score"] = np.clip(
    df["burnout_risk_score"],
    0,
    100
)

df["burnout_risk_score"] = (
    df["burnout_risk_score"].round(1)
)


# ============================================================
# CREATE BURNOUT CATEGORY
# ============================================================

def burnout_category(score):

    if score < 30:
        return "Low"

    elif score < 60:
        return "Moderate"

    elif score < 80:
        return "High"

    else:
        return "Critical"


df["burnout_risk"] = df["burnout_risk_score"].apply(
    burnout_category
)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv(
    "data/final_dataset.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("LABEL CREATION COMPLETE")
print("==========================================")

print("\nStress Risk Distribution:")
print(
    df["stress_risk"].value_counts()
)

print("\nBurnout Risk Distribution:")
print(
    df["burnout_risk"].value_counts()
)

print("\nSample records:")
print(
    df[
        [
            "personnel_id",
            "week",
            "stress_risk_score",
            "stress_risk",
            "burnout_risk_score",
            "burnout_risk"
        ]
    ].head(10)
)

print("\nSaved as:")
print("data/final_dataset.csv")