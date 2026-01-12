import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

INPUT_FILE = "outputs/cleaned_features_all.csv"


def run_clustering():
    print(f"Loading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("Columns found:")
    print(df.columns.tolist())

    # Use all numeric features
    X = df.select_dtypes(include="number")

    # Choose number of clusters (F, G, H = 3)
    n_clusters = 3
    print(f"Number of clusters: {n_clusters}")

    # ---------- KMEANS ----------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)

    # ---------- HIERARCHICAL ----------
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hierarchical.fit_predict(X)
    hier_silhouette = silhouette_score(X, hier_labels)

    # ---------- RESULTS TABLE ----------
    results = pd.DataFrame({
        "Method": ["KMeans", "Hierarchical"],
        "Silhouette Score": [
            kmeans_silhouette,
            hier_silhouette
        ]
    }).round(4)

    results.to_csv("outputs/part2d_clustering_results.csv", index=False)

    print("\n=== Part 2d: Clustering Results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    run_clustering()