import pandas as pd

# ============================================================
# LOAD DATASETS
# ============================================================

hr = pd.read_csv("data/hr_data.csv")
deployment = pd.read_csv("data/deployment_data.csv")
leave = pd.read_csv("data/leave_data.csv")
workload = pd.read_csv("data/workload_data.csv")
wellness = pd.read_csv("data/wellness_data.csv")
behavior = pd.read_csv("data/behavioral_data.csv")


# ============================================================
# DISPLAY BASIC INFORMATION
# ============================================================

print("\n========== DATASET SIZES ==========")

print("HR:", hr.shape)
print("Deployment:", deployment.shape)
print("Leave:", leave.shape)
print("Workload:", workload.shape)
print("Wellness:", wellness.shape)
print("Behavior:", behavior.shape)


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\n========== HR COLUMNS ==========")
print(hr.columns.tolist())

print("\n========== DEPLOYMENT COLUMNS ==========")
print(deployment.columns.tolist())

print("\n========== LEAVE COLUMNS ==========")
print(leave.columns.tolist())

print("\n========== WORKLOAD COLUMNS ==========")
print(workload.columns.tolist())

print("\n========== WELLNESS COLUMNS ==========")
print(wellness.columns.tolist())

print("\n========== BEHAVIOR COLUMNS ==========")
print(behavior.columns.tolist())


# ============================================================
# MERGE PERSON-LEVEL DATA
# ============================================================

personnel = hr.merge(
    deployment,
    on="personnel_id",
    how="left"
)

personnel = personnel.merge(
    leave,
    on="personnel_id",
    how="left"
)


# ============================================================
# MERGE WEEKLY DATA
# ============================================================

weekly = workload.merge(
    wellness,
    on=["personnel_id", "week"],
    how="left"
)

weekly = weekly.merge(
    behavior,
    on=["personnel_id", "week"],
    how="left"
)


# ============================================================
# COMBINE EVERYTHING
# ============================================================

final_df = weekly.merge(
    personnel,
    on="personnel_id",
    how="left"
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n========== FINAL DATASET ==========")

print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))

print("\nColumns:")
print(final_df.columns.tolist())

print("\nFirst 5 records:")
print(final_df.head())


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print(final_df.isnull().sum())


# ============================================================
# SAVE ML DATASET
# ============================================================

final_df.to_csv(
    "data/ml_dataset.csv",
    index=False
)

print("\n==============================================")
print("ML DATASET CREATED SUCCESSFULLY")
print("==============================================")

print("\nSaved as:")
print("data/ml_dataset.csv")