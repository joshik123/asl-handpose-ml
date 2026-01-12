# Part 2b: Data Preprocessing for Hand Pose Features
# This script loads feature CSV files, cleans the data (removes duplicates, missing values, and outliers),
# scales numeric features, and saves cleaned individual and combined datasets.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# List of raw feature CSV files to process
feature_files = [
    "data/features_F.csv",
    "data/features_G.csv",
    "data/features_H.csv"
]

def preprocess_file(file_path):
    """
    Load and clean a single feature CSV file:
    - Remove duplicates
    - Remove rows with missing values
    - Remove outliers using Z-score (threshold = 3)
    - Scale numeric features
    Returns a cleaned pandas DataFrame or None if file is empty/invalid.
    """
    print(f"\nProcessing {file_path}...")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return None

    # Load data
    df = pd.read_csv(file_path)
    print("Original shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna()

    # Only keep numeric columns for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found. Skipping this file.")
        return None

    # Remove outliers using Z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    if df.empty:
        print("No data left after cleaning. Skipping this file.")
        return None

    print("Shape after cleaning:", df.shape)

    # Keep label column if it exists
    y = df["ASL_sign"] if "ASL_sign" in df.columns else None

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    # Create cleaned DataFrame
    cleaned = pd.DataFrame(X_scaled, columns=numeric_cols)

    # Add label column back
    if y is not None:
        cleaned["ASL_sign"] = y.values

    return cleaned

# List to store all cleaned DataFrames
cleaned_dataframes = []

# Process each feature file
for file in feature_files:
    cleaned_df = preprocess_file(file)
    if cleaned_df is not None:
        # Save individual cleaned file
        output_path = f"outputs/cleaned_{os.path.basename(file)}"
        cleaned_df.to_csv(output_path, index=False)
        print(f"Saved cleaned file: {output_path}")

        # Add to list for combining
        cleaned_dataframes.append(cleaned_df)
    else:
        print(f"Skipped file: {file}")

# Combine all cleaned data into a single CSV
if cleaned_dataframes:
    combined_df = pd.concat(cleaned_dataframes, ignore_index=True)
    combined_df.to_csv("outputs/cleaned_features_all.csv", index=False)
    print("\nAll files processed and combined successfully!")
    print("Combined CSV saved as: outputs/cleaned_features_all.csv")
else:
    print("\nNo valid data processed. Combined CSV not created.")