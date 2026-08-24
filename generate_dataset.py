import pandas as pd
import numpy as np
import os

# ============================================================
# PERSONNEL STRESS & WELFARE - SYNTHETIC DATASET GENERATOR
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

NUM_PERSONNEL = 5000
NUM_WEEKS = 12

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Anonymous personnel IDs
personnel_ids = [f"P{i:04d}" for i in range(1, NUM_PERSONNEL + 1)]


# ============================================================
# 1. BASIC HR DATA
# ============================================================

hr_records = []

for pid in personnel_ids:

    age = np.random.randint(21, 51)
    service_years = np.random.randint(1, max(2, age - 20))

    rank_level = np.random.choice(
        ["Junior", "Intermediate", "Senior"],
        p=[0.45, 0.40, 0.15]
    )

    transfer_frequency = np.random.poisson(1)

    training_hours = max(
        0,
        np.random.normal(10, 4)
    )

    hr_records.append([
        pid,
        age,
        service_years,
        rank_level,
        transfer_frequency,
        round(training_hours, 1)
    ])

hr_df = pd.DataFrame(
    hr_records,
    columns=[
        "personnel_id",
        "age",
        "service_years",
        "rank_level",
        "transfer_frequency",
        "training_hours"
    ]
)

hr_df.to_csv("data/hr_data.csv", index=False)


# ============================================================
# 2. DEPLOYMENT DATA
# ============================================================

deployment_records = []

for pid in personnel_ids:

    deployment_days = np.random.randint(0, 121)

    deployment_frequency = np.random.poisson(2)

    operational_exposure = np.random.randint(1, 11)

    deployment_records.append([
        pid,
        deployment_days,
        deployment_frequency,
        operational_exposure
    ])

deployment_df = pd.DataFrame(
    deployment_records,
    columns=[
        "personnel_id",
        "deployment_days",
        "deployment_frequency",
        "operational_exposure"
    ]
)

deployment_df.to_csv(
    "data/deployment_data.csv",
    index=False
)


# ============================================================
# 3. LEAVE DATA
# ============================================================

leave_records = []

for pid in personnel_ids:

    leave_days = np.random.randint(0, 16)

    days_since_leave = np.random.randint(5, 91)

    leave_frequency = np.random.randint(0, 5)

    leave_records.append([
        pid,
        leave_days,
        days_since_leave,
        leave_frequency
    ])

leave_df = pd.DataFrame(
    leave_records,
    columns=[
        "personnel_id",
        "leave_days",
        "days_since_last_leave",
        "leave_frequency"
    ]
)

leave_df.to_csv(
    "data/leave_data.csv",
    index=False
)


# ============================================================
# 4. WEEKLY WORKLOAD DATA
# ============================================================

workload_records = []

for pid in personnel_ids:

    for week in range(1, NUM_WEEKS + 1):

        duty_hours = np.random.randint(35, 71)

        consecutive_duty_days = np.random.randint(2, 15)

        overtime_hours = max(
            0,
            duty_hours - 45
        )

        workload_score = np.clip(
            2
            + (duty_hours - 40) / 7
            + np.random.normal(0, 1.2),
            1,
            10
        )

        workload_records.append([
            pid,
            week,
            duty_hours,
            consecutive_duty_days,
            overtime_hours,
            round(workload_score, 1)
        ])

workload_df = pd.DataFrame(
    workload_records,
    columns=[
        "personnel_id",
        "week",
        "duty_hours",
        "consecutive_duty_days",
        "overtime_hours",
        "workload_score"
    ]
)

workload_df.to_csv(
    "data/workload_data.csv",
    index=False
)


# ============================================================
# 5. WELLNESS SURVEY DATA
# ============================================================

wellness_records = []

for pid in personnel_ids:

    # Individual baseline characteristics
    personal_stress_factor = np.random.normal(0, 1)

    for week in range(1, NUM_WEEKS + 1):

        sleep_hours = np.clip(
            np.random.normal(7, 0.8),
            4,
            9
        )

        fatigue_score = np.clip(
            5
            - (sleep_hours - 6.5)
            + personal_stress_factor
            + np.random.normal(0, 1),
            1,
            10
        )

        stress_score = np.clip(
            5
            + personal_stress_factor
            + np.random.normal(0, 1.5),
            1,
            10
        )

        mood_score = np.clip(
            8
            - stress_score * 0.45
            + np.random.normal(0, 1),
            1,
            10
        )

        emotional_exhaustion = np.clip(
            stress_score * 0.65
            + fatigue_score * 0.35
            + np.random.normal(0, 0.8),
            1,
            10
        )

        wellness_records.append([
            pid,
            week,
            round(sleep_hours, 1),
            round(fatigue_score, 1),
            round(stress_score, 1),
            round(mood_score, 1),
            round(emotional_exhaustion, 1)
        ])

wellness_df = pd.DataFrame(
    wellness_records,
    columns=[
        "personnel_id",
        "week",
        "sleep_hours",
        "fatigue_score",
        "stress_score",
        "mood_score",
        "emotional_exhaustion"
    ]
)

wellness_df.to_csv(
    "data/wellness_data.csv",
    index=False
)


# ============================================================
# 6. BEHAVIORAL DATA
# ============================================================

behavior_records = []

for pid in personnel_ids:

    baseline_activity = np.random.uniform(70, 100)

    for week in range(1, NUM_WEEKS + 1):

        attendance_score = np.clip(
            np.random.normal(90, 6),
            60,
            100
        )

        activity_level = np.clip(
            baseline_activity + np.random.normal(0, 8),
            20,
            100
        )

        performance_score = np.clip(
            np.random.normal(80, 10),
            40,
            100
        )

        social_withdrawal = np.clip(
            np.random.normal(3.5, 1.5),
            1,
            10
        )

        behavioral_change = np.clip(
            10 - activity_level / 12
            + np.random.normal(0, 1),
            1,
            10
        )

        behavior_records.append([
            pid,
            week,
            round(attendance_score, 1),
            round(activity_level, 1),
            round(performance_score, 1),
            round(social_withdrawal, 1),
            round(behavioral_change, 1)
        ])

behavior_df = pd.DataFrame(
    behavior_records,
    columns=[
        "personnel_id",
        "week",
        "attendance_score",
        "activity_level",
        "performance_score",
        "social_withdrawal",
        "behavioral_change"
    ]
)

behavior_df.to_csv(
    "data/behavioral_data.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print(" SYNTHETIC DATASETS CREATED SUCCESSFULLY")
print("==============================================\n")

print(f"Personnel: {NUM_PERSONNEL}")
print(f"Weeks per personnel: {NUM_WEEKS}")

print("\nFiles created:")

print("1. data/hr_data.csv")
print("2. data/deployment_data.csv")
print("3. data/leave_data.csv")
print("4. data/workload_data.csv")
print("5. data/wellness_data.csv")
print("6. data/behavioral_data.csv")

print("\nDataset generation complete!")