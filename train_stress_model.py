import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/final_dataset.csv")

print("Dataset loaded successfully")
print("Total records:", len(df))


# ============================================================
# 2. SORT DATA TEMPORALLY
# ============================================================

df = df.sort_values(
    ["personnel_id", "week"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE FUTURE TARGET
# ============================================================
#
# IMPORTANT:
#
# The model uses information available during the CURRENT
# week to predict the NEXT week's stress risk.
#
# Example:
#
# Week 4 information
#       ↓
#       AI
#       ↓
# Week 5 stress risk
#
# Therefore:
#
# target = next week's stress_risk
#
# ============================================================

df["future_stress_risk"] = (
    df.groupby("personnel_id")["stress_risk"]
    .shift(-1)
)


# ============================================================
# 4. CREATE PREVIOUS-WEEK FEATURES
# ============================================================

previous_features = [

    "fatigue_score",
    "sleep_hours",
    "mood_score",
    "workload_score",
    "emotional_exhaustion",
    "duty_hours",
    "overtime_hours",
    "performance_score",
    "activity_level",
    "attendance_score",
    "behavioral_change",
    "social_withdrawal"
]


for col in previous_features:

    df["previous_" + col] = (
        df.groupby("personnel_id")[col]
        .shift(1)
    )


# ============================================================
# 5. CREATE WEEK-TO-WEEK CHANGE FEATURES
# ============================================================
#
# Positive fatigue_change means fatigue is increasing.
#
# Negative sleep_change means sleep is decreasing.
#
# These are useful early-warning indicators.
# ============================================================

change_features = [

    "fatigue_score",
    "sleep_hours",
    "mood_score",
    "workload_score",
    "emotional_exhaustion",
    "duty_hours",
    "overtime_hours",
    "performance_score",
    "activity_level",
    "attendance_score",
    "behavioral_change",
    "social_withdrawal"
]


for col in change_features:

    df[col + "_change"] = (
        df[col]
        - df.groupby("personnel_id")[col].shift(1)
    )


# ============================================================
# 6. CREATE 3-WEEK ROLLING AVERAGES
# ============================================================
#
# These describe the person's recent condition rather than
# relying only on one week's measurement.
#
# The current week is included because it is available when
# making the prediction.
# ============================================================

rolling_features = [

    "fatigue_score",
    "sleep_hours",
    "mood_score",
    "workload_score",
    "emotional_exhaustion",
    "performance_score",
    "activity_level",
    "behavioral_change"
]


for col in rolling_features:

    df[col + "_3week_avg"] = (
        df.groupby("personnel_id")[col]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )


# ============================================================
# 7. CREATE 3-WEEK TREND FEATURES
# ============================================================
#
# Measures whether a variable is generally increasing or
# decreasing over the recent period.
#
# Simple approximation:
#
# Current value - value 2 weeks ago
#
# Positive:
#   increasing
#
# Negative:
#   decreasing
# ============================================================

trend_features = [

    "fatigue_score",
    "sleep_hours",
    "mood_score",
    "workload_score",
    "emotional_exhaustion",
    "performance_score",
    "activity_level",
    "behavioral_change"
]


for col in trend_features:

    df[col + "_trend"] = (
        df[col]
        - df.groupby("personnel_id")[col].shift(2)
    )


# ============================================================
# 8. REMOVE RECORDS WITHOUT REQUIRED TEMPORAL INFORMATION
# ============================================================
#
# First week has no previous-week data.
#
# Final week has no future target.
#
# Therefore these records cannot be used for this experiment.
# ============================================================

df = df[
    df["future_stress_risk"].notna()
].copy()

df = df[
    df["previous_fatigue_score"].notna()
].copy()


print(
    "Usable records after temporal features:",
    len(df)
)


# ============================================================
# 9. SELECT MODEL FEATURES
# ============================================================
#
# IMPORTANT:
#
# stress_score is deliberately NOT used.
#
# Current stress_risk is NOT used.
#
# Future variables are NOT used.
#
# future_stress_risk is the TARGET ONLY.
# ============================================================

features = [

    # --------------------------------------------------------
    # CURRENT WORKLOAD
    # --------------------------------------------------------

    "duty_hours",
    "consecutive_duty_days",
    "overtime_hours",
    "workload_score",

    # --------------------------------------------------------
    # CURRENT WELLBEING
    # --------------------------------------------------------

    "sleep_hours",
    "fatigue_score",
    "mood_score",
    "emotional_exhaustion",

    # --------------------------------------------------------
    # DEPLOYMENT
    # --------------------------------------------------------

    "deployment_days",
    "deployment_frequency",
    "operational_exposure",

    # --------------------------------------------------------
    # LEAVE
    # --------------------------------------------------------

    "days_since_last_leave",
    "leave_frequency",

    # --------------------------------------------------------
    # ADMINISTRATIVE / OPERATIONAL
    # --------------------------------------------------------

    "training_hours",
    "transfer_frequency",

    # --------------------------------------------------------
    # BEHAVIOR / PERFORMANCE
    # --------------------------------------------------------

    "attendance_score",
    "activity_level",
    "performance_score",
    "social_withdrawal",
    "behavioral_change",

    # --------------------------------------------------------
    # PREVIOUS WEEK
    # --------------------------------------------------------

    "previous_fatigue_score",
    "previous_sleep_hours",
    "previous_mood_score",
    "previous_workload_score",
    "previous_emotional_exhaustion",
    "previous_duty_hours",
    "previous_overtime_hours",

    "previous_performance_score",
    "previous_activity_level",
    "previous_attendance_score",
    "previous_behavioral_change",
    "previous_social_withdrawal",

    # --------------------------------------------------------
    # WEEK-TO-WEEK CHANGES
    # --------------------------------------------------------

    "fatigue_score_change",
    "sleep_hours_change",
    "mood_score_change",
    "workload_score_change",
    "emotional_exhaustion_change",
    "duty_hours_change",
    "overtime_hours_change",
    "performance_score_change",
    "activity_level_change",
    "attendance_score_change",
    "behavioral_change_change",
    "social_withdrawal_change",

    # --------------------------------------------------------
    # 3-WEEK AVERAGES
    # --------------------------------------------------------

    "fatigue_score_3week_avg",
    "sleep_hours_3week_avg",
    "mood_score_3week_avg",
    "workload_score_3week_avg",
    "emotional_exhaustion_3week_avg",
    "performance_score_3week_avg",
    "activity_level_3week_avg",
    "behavioral_change_3week_avg",

    # --------------------------------------------------------
    # 3-WEEK TRENDS
    # --------------------------------------------------------

    "fatigue_score_trend",
    "sleep_hours_trend",
    "mood_score_trend",
    "workload_score_trend",
    "emotional_exhaustion_trend",
    "performance_score_trend",
    "activity_level_trend",
    "behavioral_change_trend"
]


# ============================================================
# 10. CREATE X AND Y
# ============================================================

X = df[features]

y = df["future_stress_risk"]


print(
    "\nFeatures used by model:",
    len(features)
)

print(
    "\nFuture stress-risk distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# 11. PERSON-LEVEL TRAIN / TEST SPLIT
# ============================================================
#
# Entire personnel are assigned to either training or testing.
#
# This prevents information from the same person appearing
# in both datasets.
# ============================================================

np.random.seed(42)

personnel_ids = (
    df["personnel_id"]
    .unique()
    .tolist()
)

np.random.shuffle(
    personnel_ids
)


split_index = int(
    len(personnel_ids) * 0.80
)


train_personnel = personnel_ids[
    :split_index
]

test_personnel = personnel_ids[
    split_index:
]


train_mask = df[
    "personnel_id"
].isin(
    train_personnel
)

test_mask = df[
    "personnel_id"
].isin(
    test_personnel
)


X_train = df.loc[
    train_mask,
    features
]

y_train = df.loc[
    train_mask,
    "future_stress_risk"
]


X_test = df.loc[
    test_mask,
    features
]

y_test = df.loc[
    test_mask,
    "future_stress_risk"
]


print("\n==========================================")
print("DATA SPLIT")
print("==========================================")

print(
    "\nTraining personnel:",
    len(train_personnel)
)

print(
    "Testing personnel:",
    len(test_personnel)
)

print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 12. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=500,

    max_depth=18,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 13. TRAIN
# ============================================================

print(
    "\nTraining Random Forest..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training complete!"
)


# ============================================================
# 14. PREDICT
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 15. PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1: "
    f"{weighted_f1:.4f}"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

labels = [
    "Critical",
    "High",
    "Low",
    "Moderate"
]


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print(
    "\nConfusion Matrix:"
)

print(
    pd.DataFrame(
        cm,
        index=[
            "Actual " + x
            for x in labels
        ],
        columns=[
            "Predicted " + x
            for x in labels
        ]
    )
)


# ============================================================
# 18. HIGH + CRITICAL RECALL
# ============================================================
#
# Treat High and Critical as "elevated risk".
#
# This measures how many genuinely elevated-risk cases
# were detected by the model.
# ============================================================

high_critical_actual = (
    y_test.isin(
        ["High", "Critical"]
    )
)


high_critical_predicted = (
    pd.Series(
        y_pred,
        index=y_test.index
    ).isin(
        ["High", "Critical"]
    )
)


high_critical_recall = (
    (
        high_critical_actual
        &
        high_critical_predicted
    ).sum()
    /
    high_critical_actual.sum()
)


print(
    f"\nHigh + Critical Recall: "
    f"{high_critical_recall * 100:.2f}%"
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n==========================================")
print("TOP RISK FACTORS")
print("==========================================")


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


model_path = (
    "models/stress_risk_model_v3.pkl"
)


joblib.dump(
    model,
    model_path
)


# ============================================================
# 21. SAVE FEATURE LIST
# ============================================================

feature_path = (
    "models/stress_risk_features_v3.pkl"
)


joblib.dump(
    features,
    feature_path
)


# ============================================================
# 22. SAVE MODEL METADATA
# ============================================================

metadata = {

    "model_version": "V3",

    "target": "next_week_stress_risk",

    "training_personnel": len(
        train_personnel
    ),

    "testing_personnel": len(
        test_personnel
    ),

    "features": features,

    "accuracy": accuracy,

    "macro_f1": macro_f1,

    "weighted_f1": weighted_f1,

    "high_critical_recall":
        high_critical_recall
}


metadata_path = (
    "models/stress_risk_metadata_v3.pkl"
)


joblib.dump(
    metadata,
    metadata_path
)


print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    model_path
)

print(
    feature_path
)

print(
    metadata_path
)