import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

# -------------------------------
# Setup output folder and log file
# -------------------------------
os.makedirs("outputs", exist_ok=True)
log_file = "outputs/preprocessing_log.txt"

def log(message):
    """Print message and write to log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

# -------------------------------
# Feature files
# -------------------------------
files = [
    "data/features_F.csv",
    "data/features_G.csv",
    "data/features_H.csv"
]

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_features(file_path):
    log(f"\nProcessing {file_path}...")

    if not os.path.exists(file_path):
        log(f"⚠️ File not found: {file_path}. Skipping.")
        return None

    data = pd.read_csv(file_path)
    log(f"Initial shape: {data.shape}")

    # Remove duplicates and missing values
    rows_before = len(data)
    data = data.drop_duplicates().dropna()
    log(f"Removed {rows_before - len(data)} rows due to duplicates/NaNs")

    if data.empty:
        log("⚠️ No data left after cleaning. Skipping file.")
        return None

    # -------------------------------
    # Keep only fully numeric columns
    # -------------------------------
    X = pd.DataFrame()
    dropped_cols = []
    for col in data.columns:
        converted = pd.to_numeric(data[col], errors='coerce')
        if converted.notna().all():
            X[col] = converted
        else:
            dropped_cols.append(col)
    log(f"Dropped {len(dropped_cols)} non-numeric columns: {dropped_cols}")

    if X.empty:
        log("⚠️ No numeric columns left after filtering. Skipping file.")
        return None

    # -------------------------------
    # Remove outliers using Z-score
    # -------------------------------
    rows_before = X.shape[0]
    z_scores = np.abs(stats.zscore(X))
    X = X[(z_scores < 3).all(axis=1)]
    log(f"Removed {rows_before - X.shape[0]} rows as outliers")

    if X.empty:
        log("⚠️ No data left after removing outliers. Skipping file.")
        return None

    log(f"Shape after cleaning: {X.shape}")

    # -------------------------------
    # Separate label if present
    # -------------------------------
    y = data["ASL_sign"] if "ASL_sign" in data.columns else None

    # -------------------------------
    # Scale numeric features
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cleaned = pd.DataFrame(X_scaled, columns=X.columns)

    # Reattach label
    if y is not None:
        y_aligned = y.loc[X.index]
        cleaned["ASL_sign"] = y_aligned.values

    return cleaned

# -------------------------------
# Process all files
# -------------------------------
cleaned_dfs = []

# Clear previous log
if os.path.exists(log_file):
    os.remove(log_file)

for file in files:
    cleaned_df = preprocess_features(file)
    if cleaned_df is not None and not cleaned_df.empty:
        out_name = f"outputs/cleaned_{os.path.basename(file)}"
        cleaned_df.to_csv(out_name, index=False)
        log(f"Saved: {out_name}")
        cleaned_dfs.append(cleaned_df)
    else:
        log(f"Skipping saving for {file} (no valid data).")

# -------------------------------
# Combine all cleaned data
# -------------------------------
if cleaned_dfs:
    combined = pd.concat(cleaned_dfs, ignore_index=True)
    combined.to_csv("outputs/cleaned_features_all.csv", index=False)
    log("\nAll preprocessing complete. Combined CSV saved.")
else:
    log("\nNo data to combine. Nothing saved in combined CSV.")