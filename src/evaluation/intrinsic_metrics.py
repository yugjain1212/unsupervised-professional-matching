import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, sample_size: int = 5000
) -> dict:
    """
    Compute all intrinsic clustering metrics for a given label set.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for clustering (PCA space).
    labels : np.ndarray
        Cluster assignments for each sample.
    sample_size : int
        Subsample size for silhouette computation on large datasets.

    Returns
    -------
    dict with keys: silhouette, davies_bouldin, calinski_harabasz,
                    n_clusters, n_noise (for DBSCAN), cluster_sizes
    """
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = int(np.sum(labels == -1)) if -1 in unique_labels else 0

    if n_clusters < 2:
        cluster_sizes = {}
        for lbl in unique_labels:
            if lbl == -1:
                cluster_sizes["noise"] = int(np.sum(labels == -1))
            else:
                cluster_sizes[int(lbl)] = int(np.sum(labels == lbl))

        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": cluster_sizes,
        }

    # For DBSCAN, we exclude noise points from metric calculation
    mask = labels != -1
    X_metrics = X[mask]
    labels_metrics = labels[mask]

    effective_labels = labels_metrics
    effective_X = X_metrics

    if len(X_metrics) > 20000:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_metrics), size=min(sample_size, len(X_metrics)), replace=False)
        effective_X = X_metrics[indices]
        effective_labels = labels_metrics[indices]

        unique_eff = np.unique(effective_labels)
        if len(unique_eff) < 2:
            sil = np.nan
        else:
            sil = silhouette_score(effective_X, effective_labels)
    else:
        sil = silhouette_score(effective_X, effective_labels)

    db = davies_bouldin_score(X_metrics, labels_metrics)
    ch = calinski_harabasz_score(X_metrics, labels_metrics)

    cluster_sizes = {}
    for lbl in unique_labels:
        if lbl == -1:
            cluster_sizes["noise"] = int(np.sum(labels == -1))
        else:
            cluster_sizes[int(lbl)] = int(np.sum(labels == lbl))

    return {
        "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan,
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes,
    }


def compare_algorithms(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple algorithm results.

    Parameters
    ----------
    results : dict
        Keys are algorithm names, values are dicts from evaluate_clustering().

    Returns
    -------
    pd.DataFrame with one row per algorithm and metric columns.
    """
    rows = []
    for algo_name, metrics in results.items():
        row = {"algorithm": algo_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    display_cols = [
        "algorithm",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "n_clusters",
        "n_noise",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    return df[display_cols]
