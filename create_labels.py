import pandas as pd
import numpy as np

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/ml_dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# SORT BY PERSON AND WEEK
# ============================================================

df = df.sort_values(
    ["personnel_id", "week"]
).reset_index(drop=True)


# ============================================================
# CREATE NEXT-WEEK RISK INDICATORS
# ============================================================
#
# The target for the current week is based on the following
# week's condition.
#
# Example:
#
# Week 1 indicators ──────> Predict ──────> Week 2 risk
#
# This makes the problem genuinely predictive.
# ============================================================

future_cols = [
    "workload_score",
    "fatigue_score",
    "sleep_hours",
    "mood_score",
    "emotional_exhaustion",
    "deployment_days",
    "days_since_last_leave",
    "consecutive_duty_days"
]

for col in future_cols:

    df["future_" + col] = (
        df.groupby("personnel_id")[col]
        .shift(-1)
    )


# ============================================================
# REMOVE LAST WEEK OF EACH PERSON
# ============================================================
#
# The last week has no future week available.
# ============================================================

df = df[
    df["future_fatigue_score"].notna()
].copy()


# ============================================================
# CALCULATE FUTURE RISK SCORE
# ============================================================
#
# IMPORTANT:
# This score is ONLY used to create the synthetic target.
#
# The ML model will NOT receive these future_* columns.
# ============================================================

future_risk = (

    0.22 * (
        df["future_workload_score"] / 10
    )

    + 0.20 * (
        df["future_fatigue_score"] / 10
    )

    + 0.16 * (
        (8 - df["future_sleep_hours"])
        .clip(0, 4) / 4
    )

    + 0.12 * (
        (10 - df["future_mood_score"]) / 10
    )

    + 0.12 * (
        df["future_emotional_exhaustion"] / 10
    )

    + 0.07 * (
        df["future_deployment_days"] / 120
    )

    + 0.06 * (
        df["future_days_since_last_leave"] / 90
    )

    + 0.05 * (
        df["future_consecutive_duty_days"] / 14
    )
)


# Convert to percentage
future_risk = future_risk * 100


# ============================================================
# ADD TEMPORAL CHANGE COMPONENT
# ============================================================

df["fatigue_change"] = (
    df["future_fatigue_score"]
    - df["fatigue_score"]
)

df["sleep_change"] = (
    df["future_sleep_hours"]
    - df["sleep_hours"]
)

df["mood_change"] = (
    df["future_mood_score"]
    - df["mood_score"]
)

df["workload_change"] = (
    df["future_workload_score"]
    - df["workload_score"]
)

df["exhaustion_change"] = (
    df["future_emotional_exhaustion"]
    - df["emotional_exhaustion"]
)


# ============================================================
# TREND RISK
# ============================================================

trend_risk = (

    0.30 * (
        df["fatigue_change"]
        .clip(0, 5) / 5
    )

    + 0.20 * (
        (-df["sleep_change"])
        .clip(0, 4) / 4
    )

    + 0.20 * (
        (-df["mood_change"])
        .clip(0, 5) / 5
    )

    + 0.15 * (
        df["workload_change"]
        .clip(0, 5) / 5
    )

    + 0.15 * (
        df["exhaustion_change"]
        .clip(0, 5) / 5
    )
)

trend_risk = trend_risk * 100


# ============================================================
# FINAL FUTURE RISK SCORE
# ============================================================

np.random.seed(42)

df["future_risk_score"] = (
    0.85 * future_risk
    + 0.15 * trend_risk
)

# Natural variation
df["future_risk_score"] += np.random.normal(
    0,
    3,
    len(df)
)

df["future_risk_score"] = np.clip(
    df["future_risk_score"],
    0,
    100
)

df["future_risk_score"] = (
    df["future_risk_score"].round(1)
)


# ============================================================
# CREATE RISK CATEGORIES USING PERCENTILES
# ============================================================
#
# Approximately:
#
# Low       = lowest 40%
# Moderate  = next 35%
# High      = next 20%
# Critical  = highest 5%
#
# This prevents Critical from disappearing because of arbitrary
# absolute thresholds.
# ============================================================

q40 = df["future_risk_score"].quantile(0.40)
q75 = df["future_risk_score"].quantile(0.75)
q95 = df["future_risk_score"].quantile(0.95)


def stress_category(score):

    if score <= q40:
        return "Low"

    elif score <= q75:
        return "Moderate"

    elif score <= q95:
        return "High"

    else:
        return "Critical"


df["stress_risk"] = (
    df["future_risk_score"]
    .apply(stress_category)
)


# ============================================================
# BURNOUT RISK SCORE
# ============================================================

burnout_score = (

    0.25 * (
        df["future_workload_score"] / 10
    )

    + 0.20 * (
        df["future_fatigue_score"] / 10
    )

    + 0.18 * (
        df["future_emotional_exhaustion"] / 10
    )

    + 0.15 * (
        (8 - df["future_sleep_hours"])
        .clip(0, 4) / 4
    )

    + 0.10 * (
        df["future_consecutive_duty_days"] / 14
    )

    + 0.07 * (
        df["future_deployment_days"] / 120
    )

    + 0.05 * (
        df["future_days_since_last_leave"] / 90
    )
)

df["burnout_risk_score"] = (
    burnout_score * 100
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
# BURNOUT CATEGORIES USING PERCENTILES
# ============================================================

bq40 = df["burnout_risk_score"].quantile(0.40)
bq75 = df["burnout_risk_score"].quantile(0.75)
bq95 = df["burnout_risk_score"].quantile(0.95)


def burnout_category(score):

    if score <= bq40:
        return "Low"

    elif score <= bq75:
        return "Moderate"

    elif score <= bq95:
        return "High"

    else:
        return "Critical"


df["burnout_risk"] = (
    df["burnout_risk_score"]
    .apply(burnout_category)
)


# ============================================================
# SAVE DATASET
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

print("\nTotal usable records:", len(df))


print("\nStress Risk Distribution:")

stress_counts = (
    df["stress_risk"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High", "Critical"],
        fill_value=0
    )
)

print(stress_counts)


print("\nStress Risk Percentage:")

stress_percent = (
    stress_counts / len(df) * 100
).round(2)

print(stress_percent)


print("\nBurnout Risk Distribution:")

burnout_counts = (
    df["burnout_risk"]
    .value_counts()
    .reindex(
        ["Low", "Moderate", "High", "Critical"],
        fill_value=0
    )
)

print(burnout_counts)


print("\nBurnout Risk Percentage:")

burnout_percent = (
    burnout_counts / len(df) * 100
).round(2)

print(burnout_percent)


print("\nRisk thresholds:")
print("Low / Moderate threshold:", round(q40, 2))
print("Moderate / High threshold:", round(q75, 2))
print("High / Critical threshold:", round(q95, 2))


print("\nSample records:")

print(
    df[
        [
            "personnel_id",
            "week",
            "future_risk_score",
            "stress_risk",
            "burnout_risk_score",
            "burnout_risk"
        ]
    ].head(10)
)


print("\nSaved as:")
print("data/final_dataset.csv")