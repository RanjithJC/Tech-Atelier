import pandas as pd
import numpy as np
import os

# ============================================================
# PERSONNEL STRESS & WELFARE
# TEMPORALLY CORRELATED SYNTHETIC DATASET GENERATOR
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

NUM_PERSONNEL = 5000
NUM_WEEKS = 12

os.makedirs("data", exist_ok=True)

personnel_ids = [
    f"P{i:04d}"
    for i in range(1, NUM_PERSONNEL + 1)
]


# ============================================================
# 1. BASIC HR DATA
# ============================================================

hr_records = []

for pid in personnel_ids:

    age = np.random.randint(21, 51)

    service_years = np.random.randint(
        1,
        max(2, age - 20)
    )

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

hr_df.to_csv(
    "data/hr_data.csv",
    index=False
)


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

    days_since_leave = np.random.randint(
        5,
        91
    )

    leave_frequency = np.random.randint(
        0,
        5
    )

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
#
# Workload now evolves over time.
#
# High latent stress → somewhat higher workload
# Low latent stress  → somewhat lower workload
#
# There is still random variation so the dataset doesn't
# become artificially deterministic.
# ============================================================

workload_records = []

# Store latent workload state for each person
workload_state = {}

for pid in personnel_ids:

    workload_state[pid] = np.random.normal(
        0,
        1
    )

    for week in range(
        1,
        NUM_WEEKS + 1
    ):

        # Gradual workload evolution
        workload_state[pid] = (

            0.75 * workload_state[pid]

            + 0.25 * np.random.normal(
                0,
                1
            )
        )

        duty_hours = np.clip(
            50
            + workload_state[pid] * 6
            + np.random.normal(0, 4),
            35,
            75
        )

        consecutive_duty_days = np.clip(
            7
            + workload_state[pid] * 1.8
            + np.random.normal(0, 2),
            2,
            14
        )

        overtime_hours = max(
            0,
            duty_hours - 45
        )

        workload_score = np.clip(
            5
            + workload_state[pid] * 1.2
            + (duty_hours - 50) / 10
            + np.random.normal(0, 0.7),
            1,
            10
        )

        workload_records.append([
            pid,
            week,
            round(duty_hours, 1),
            round(consecutive_duty_days, 1),
            round(overtime_hours, 1),
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
# 5. WELLNESS DATA
# ============================================================
#
# This is the most important change.
#
# Each person has a latent welfare state that evolves from
# one week to the next.
#
# Higher state:
#   ↓ sleep
#   ↑ fatigue
#   ↑ stress
#   ↓ mood
#   ↑ emotional exhaustion
#
# Lower state:
#   ↑ sleep
#   ↓ fatigue
#   ↓ stress
#   ↑ mood
# ============================================================

wellness_records = []

# Individual baseline + evolving stress state
stress_state = {}

for pid in personnel_ids:

    # Stable individual characteristic
    personal_baseline = np.random.normal(
        0,
        0.7
    )

    # Starting welfare state
    stress_state[pid] = np.random.normal(
        0,
        0.6
    )

    for week in range(
        1,
        NUM_WEEKS + 1
    ):

        # Get corresponding workload
        row = workload_df[
            (workload_df["personnel_id"] == pid)
            &
            (workload_df["week"] == week)
        ].iloc[0]

        workload_pressure = (
            row["workload_score"] - 5
        )

        # ----------------------------------------------------
        # OCCASIONAL SHOCKS
        # ----------------------------------------------------

        shock = 0

        # Around 8% chance of a stressful event
        if np.random.random() < 0.08:

            shock = np.random.uniform(
                0.8,
                2.0
            )

        # Around 5% chance of recovery event
        if np.random.random() < 0.05:

            shock -= np.random.uniform(
                0.8,
                1.8
            )

        # ----------------------------------------------------
        # UPDATE STRESS STATE
        # ----------------------------------------------------

        stress_state[pid] = (

            0.78 * stress_state[pid]

            + 0.20 * workload_pressure

            + 0.10 * personal_baseline

            + shock

            + np.random.normal(
                0,
                0.35
            )
        )

        # Keep latent state within reasonable limits
        stress_state[pid] = np.clip(
            stress_state[pid],
            -4,
            5
        )

        s = stress_state[pid]

        # ----------------------------------------------------
        # SLEEP
        # ----------------------------------------------------

        sleep_hours = np.clip(
            7.2
            - 0.55 * s
            + np.random.normal(0, 0.35),
            4,
            9
        )
        wellness_records.append([
            pid,
            week,
            round(sleep_hours, 1),
            
        ])


wellness_df = pd.DataFrame(
    wellness_records,
    columns=[
        "personnel_id",
        "week",
        "sleep_hours",
        
    ]
)

wellness_df.to_csv(
    "data/wellness_data.csv",
    index=False
)


# ============================================================
# 6. BEHAVIORAL DATA
# ============================================================
#
# Behavioral indicators now respond to the person's evolving
# welfare state.
# ============================================================

behavior_records = []

activity_state = {}

for pid in personnel_ids:

    activity_state[pid] = np.random.uniform(
        75,
        95
    )

    for week in range(
        1,
        NUM_WEEKS + 1
    ):

        # Get person's stress state
        s = stress_state[pid]

        # ----------------------------------------------------
        # ACTIVITY
        # ----------------------------------------------------

        activity_state[pid] = (

            0.80 * activity_state[pid]

            + 0.20 * (
                90 - 7 * max(s, 0)
            )

            + np.random.normal(
                0,
                3
            )
        )

        activity_level = np.clip(
            activity_state[pid],
            20,
            100
        )

        # ----------------------------------------------------
        # ATTENDANCE
        # ----------------------------------------------------

        attendance_score = np.clip(
            92
            - 2.5 * max(s, 0)
            + np.random.normal(0, 3),
            60,
            100
        )
        

behavior_df = pd.DataFrame(
    behavior_records,
    columns=[
        "personnel_id",
        "week",
        "attendance_score",
        "activity_level",
        
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

print(
    f"Personnel: {NUM_PERSONNEL}"
)

print(
    f"Weeks per personnel: {NUM_WEEKS}"
)

print("\nFiles created:")

print("1. data/hr_data.csv")
print("2. data/deployment_data.csv")
print("3. data/leave_data.csv")
print("4. data/workload_data.csv")
print("5. data/wellness_data.csv")
print("6. data/behavioral_data.csv")

print("\nDataset generation complete!")