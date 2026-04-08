import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from src.config import PLOTS_DIR
import os


def run_hierarchical_experiment(
    X_pca: np.ndarray,
    n_clusters_range: list = None,
    linkage_methods: list = None,
    sample_size: int = 5000,
) -> pd.DataFrame:
    """
    Run Agglomerative Clustering across different cluster counts and linkage methods.
    Records silhouette score for each combination.

    WHY: Hierarchical clustering builds a tree of clusters (dendrogram)
    and doesn't require specifying K upfront. We test different linkage
    strategies to find which produces the most coherent clusters.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed feature matrix.
    n_clusters_range : list
        Number of clusters to test.
    linkage_methods : list
        Linkage strategies: 'ward', 'complete', 'average', 'single'.
    sample_size : int
        Sample size for silhouette computation on large datasets.

    Returns
    -------
    pd.DataFrame with columns: n_clusters, linkage, silhouette, inertia_approx
    """
    if n_clusters_range is None:
        n_clusters_range = [3, 5, 8, 10, 12, 15, 20]
    if linkage_methods is None:
        linkage_methods = ["ward", "complete", "average"]

    results = []
    total = len(n_clusters_range) * len(linkage_methods)
    count = 0

    print(f"Running Hierarchical Clustering: {total} combinations...")
    for n_clusters in n_clusters_range:
        for linkage_method in linkage_methods:
            count += 1

            if linkage_method == "ward" and X_pca.shape[1] > 100:
                continue

            try:
                hc = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage=linkage_method, metric="euclidean"
                )
                labels = hc.fit_predict(X_pca)

                n_unique = len(set(labels))
                if n_unique < 2:
                    results.append(
                        {
                            "n_clusters": n_clusters,
                            "linkage": linkage_method,
                            "silhouette": np.nan,
                            "inertia_approx": np.nan,
                        }
                    )
                    print(
                        f"  [{count}/{total}] n={n_clusters:3d} linkage={linkage_method:10s} | clusters={n_unique} | sil=n/a"
                    )
                    continue

                if len(X_pca) > 20000:
                    rng = np.random.RandomState(42)
                    indices = rng.choice(
                        len(X_pca), size=min(sample_size, len(X_pca)), replace=False
                    )
                    sil = silhouette_score(X_pca[indices], labels[indices])
                else:
                    sil = silhouette_score(X_pca, labels)

                inertia_approx = np.sum(np.var(X_pca, axis=0)) * (1 - sil)

                results.append(
                    {
                        "n_clusters": n_clusters,
                        "linkage": linkage_method,
                        "silhouette": round(sil, 4),
                        "inertia_approx": round(inertia_approx, 2),
                    }
                )
                print(
                    f"  [{count}/{total}] n={n_clusters:3d} linkage={linkage_method:10s} | clusters={n_unique} | sil={sil:.4f}"
                )

            except Exception as e:
                results.append(
                    {
                        "n_clusters": n_clusters,
                        "linkage": linkage_method,
                        "silhouette": np.nan,
                        "inertia_approx": np.nan,
                    }
                )
                print(
                    f"  [{count}/{total}] n={n_clusters:3d} linkage={linkage_method:10s} | ERROR: {str(e)[:50]}"
                )

    return pd.DataFrame(results)


def plot_dendrogram(
    X_pca: np.ndarray, n_clusters: int = 15, sample_size: int = 1000
) -> None:
    """
    Plot a dendrogram for hierarchical clustering.
    Uses a sample of data to keep the plot readable.
    """
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(
        len(X_pca), size=min(sample_size, len(X_pca)), replace=False
    )
    X_sample = X_pca[sample_idx]

    linkage_matrix = linkage(X_sample, method="ward")

    plt.figure(figsize=(14, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        p=30,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        leaf_font_size=8,
    )
    plt.title(
        f"Hierarchical Clustering Dendrogram (sample n={sample_size}, K={n_clusters})",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Sample index")
    plt.ylabel("Euclidean distance")
    plt.axhline(
        y=np.mean(linkage_matrix[:, 2]) * 0.7,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Suggested cut (K≈{n_clusters})",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "14_dendrogram.png"))
    plt.show()
    print(f"Dendrogram saved → outputs/plots/14_dendrogram.png")


def plot_hierarchical_results(results_df: pd.DataFrame) -> None:
    """
    Plot hierarchical clustering results as bar chart and heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    valid = results_df.dropna(subset=["silhouette"])
    if len(valid) == 0:
        print("No valid hierarchical clustering results to plot.")
        return

    best = valid.loc[valid["silhouette"].idxmax()]

    colors = {"ward": "#4A90D9", "complete": "#E8834E", "average": "#7B68EE"}

    for linkage_method in valid["linkage"].unique():
        subset = valid[valid["linkage"] == linkage_method]
        color = colors.get(linkage_method, "#999999")
        axes[0].plot(
            subset["n_clusters"],
            subset["silhouette"],
            marker="o",
            label=linkage_method,
            color=color,
            linewidth=2,
        )

    axes[0].axvline(
        x=int(best["n_clusters"]),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best: n={int(best['n_clusters'])}, linkage={best['linkage']}",
    )
    axes[0].set_title(
        "Silhouette score vs number of clusters", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Silhouette score")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    pivot = valid.pivot_table(
        index="linkage", columns="n_clusters", values="silhouette"
    )
    im = axes[1].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    axes[1].set_title(
        "Silhouette heatmap (linkage × clusters)", fontsize=12, fontweight="bold"
    )
    axes[1].set_xticks(range(len(pivot.columns)))
    axes[1].set_xticklabels([str(int(c)) for c in pivot.columns])
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    plt.colorbar(im, ax=axes[1])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                axes[1].text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if val > 0.12 else "black",
                )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "15_hierarchical_results.png"))
    plt.show()

    print(f"\nBest Hierarchical config:")
    print(
        f"  n_clusters={int(best['n_clusters'])}, linkage={best['linkage']} | "
        f"silhouette={best['silhouette']}"
    )


def fit_final_hierarchical(
    X_pca: np.ndarray, n_clusters: int, linkage_method: str = "ward"
) -> np.ndarray:
    """
    Fit the final Agglomerative Clustering model.
    Returns cluster label array of shape (n_profiles,).
    """
    hc = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_method, metric="euclidean"
    )
    labels = hc.fit_predict(X_pca)

    print(
        f"✓ Final Hierarchical Clustering fitted  (K={n_clusters}, linkage={linkage_method})"
    )
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(
            f"    Cluster {cluster}: {count:,} profiles "
            f"({100 * count / len(labels):.1f}%)"
        )

    return labels
