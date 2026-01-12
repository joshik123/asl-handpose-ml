import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Input CSV file with cleaned feature data
INPUT_FILE = "outputs/cleaned_features_all.csv"

def run_clustering():
    """Load features and perform KMeans and Hierarchical clustering."""
    
    print(f"Loading feature data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    print("Columns in dataset:")
    print(df.columns.tolist())

    # Keep only numeric columns for clustering
    X = df.select_dtypes(include="number")

    # We know we have 3 groups: F, G, H
    n_clusters = 3
    print(f"Number of clusters: {n_clusters}")

    # ----- KMEANS -----
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_score = silhouette_score(X, kmeans_labels)

    # ----- HIERARCHICAL -----
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X)
    hier_score = silhouette_score(X, hier_labels)

    # Save results to CSV
    results = pd.DataFrame({
        "Method": ["KMeans", "Hierarchical"],
        "Silhouette Score": [round(kmeans_score, 4), round(hier_score, 4)]
    })

    results_file = "outputs/part2d_clustering_results.csv"
    results.to_csv(results_file, index=False)
    print(f"\nClustering results saved to {results_file}")

    # Print results in console
    print("\n=== Part 2d: Clustering Results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    run_clustering()