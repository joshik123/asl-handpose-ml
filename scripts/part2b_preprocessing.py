import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Feature files
files = [
    "data/features_F.csv",
    "data/features_G.csv",
    "data/features_H.csv"
]

def preprocess_features(file_path):
    print(f"\nProcessing {file_path}...")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}. Skipping.")
        return None

    data = pd.read_csv(file_path)
    print("Initial shape:", data.shape)

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Remove rows with missing values
    data = data.dropna()

    # Identify numeric columns only
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found. Skipping file.")
        return None

    # Remove outliers using Z-score (numeric features only)
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    data = data[(z_scores < 3).all(axis=1)]

    if data.empty:
        print("No data left after cleaning. Skipping file.")
        return None

    print("Shape after cleaning:", data.shape)

    # Separate label if present
    y = data["ASL_sign"] if "ASL_sign" in data.columns else None

    # Keep numeric features only for scaling
    X = data[numeric_cols]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cleaned = pd.DataFrame(X_scaled, columns=numeric_cols)

    # Reattach label (not scaled)
    if y is not None:
        cleaned["ASL_sign"] = y.values

    return cleaned

# Process each file
cleaned_dfs = []

for file in files:
    cleaned_df = preprocess_features(file)
    if cleaned_df is not None and not cleaned_df.empty:
        out_name = f"outputs/cleaned_{os.path.basename(file)}"
        cleaned_df.to_csv(out_name, index=False)
        print(f"Saved: {out_name}")
        cleaned_dfs.append(cleaned_df)
    else:
        print(f"Skipping saving for {file} (no valid data).")

# Combine all cleaned data if there is anything
if cleaned_dfs:
    combined = pd.concat(cleaned_dfs, ignore_index=True)
    combined.to_csv("outputs/cleaned_features_all.csv", index=False)
    print("\nAll preprocessing complete. Combined CSV saved.")
else:
    print("\nNo data to combine. Nothing saved in combined CSV.")