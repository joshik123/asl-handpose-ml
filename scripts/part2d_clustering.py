import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

INPUT_FILE = "outputs/cleaned_features_F.csv"

def run_clustering():
    print(f"Loading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("Columns found:")
    print(df.columns.tolist())

    # Detect label column automatically
    possible_label_cols = ["label", "class", "asl_sign", "sign", "gesture"]

    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            "No label column found. Columns are: "
            + ", ".join(df.columns)
        )

    print(f"Using label column: {label_col}")

    # Separate
    y_true = df[label_col]
    X = df.drop(columns=[label_col])

    # Keep numeric only (CRITICAL)
    X = X.select_dtypes(include="number")

    n_clusters = y_true.nunique()
    print(f"Number of clusters: {n_clusters}")

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X)

    # Metrics
    results = pd.DataFrame({
        "Method": ["KMeans", "Hierarchical"],
        "ARI": [
            adjusted_rand_score(y_true, kmeans_labels),
            adjusted_rand_score(y_true, hier_labels)
        ],
        "NMI": [
            normalized_mutual_info_score(y_true, kmeans_labels),
            normalized_mutual_info_score(y_true, hier_labels)
        ],
        "Silhouette": [
            silhouette_score(X, kmeans_labels),
            silhouette_score(X, hier_labels)
        ]
    })

    results.to_csv("outputs/part2d_clustering_results.csv", index=False)

    print("\n=== Clustering Results ===")
    print(results)

if __name__ == "__main__":
    run_clustering()