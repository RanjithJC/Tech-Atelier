import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/final_dataset.csv")

print("Dataset loaded successfully")
print("Total records:", len(df))


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "duty_hours",
    "consecutive_duty_days",
    "overtime_hours",
    "workload_score",

    "sleep_hours",
    "fatigue_score",
    "stress_score",
    "mood_score",
    "emotional_exhaustion",

    "deployment_days",
    "deployment_frequency",
    "operational_exposure",

    "days_since_last_leave",
    "leave_frequency",

    "training_hours",
    "transfer_frequency",

    "attendance_score",
    "activity_level",
    "performance_score",
    "social_withdrawal",
    "behavioral_change"
]

X = df[features]

# Target we want the AI to predict
y = df["stress_risk"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 4. CREATE RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 6. PREDICT
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. EVALUATE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==========================================")
print("TOP RISK FACTORS")
print("==========================================")

print(importance.to_string(index=False))


# ============================================================
# 9. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

model_path = "models/stress_risk_model.pkl"

joblib.dump(model, model_path)

print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(model_path)