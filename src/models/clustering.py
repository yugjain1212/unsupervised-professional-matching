import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.config import K_RANGE, PLOTS_DIR
import os


def run_kmeans_experiment(X_pca: np.ndarray) -> pd.DataFrame:
    """
    Run K-Means for every K in K_RANGE.
    Records inertia and silhouette score for each K.
    Returns a dataframe of results for plotting.

    WHY: We cannot know the correct K in advance. Running across
    a range and evaluating with two complementary metrics gives
    a principled basis for choosing K rather than guessing.
    """
    results = []

    print(f"Running K-Means for K = {K_RANGE.start} to {K_RANGE.stop - 1}...")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X_pca)

        inertia = km.inertia_
        sil = silhouette_score(X_pca, labels, sample_size=5000, random_state=42)

        results.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  K={k:2d} | inertia={inertia:,.0f} | silhouette={sil:.4f}")

    return pd.DataFrame(results)


def plot_elbow_and_silhouette(results_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))  # wider figure for K=3..30

    # Elbow curve
    axes[0].plot(
        results_df["k"], results_df["inertia"], marker="o", color="#4A90D9", linewidth=2
    )
    axes[0].set_title("Elbow curve — inertia vs K", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(results_df["k"][::2])  # every 2nd tick to avoid crowding

    # Silhouette scores
    best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
    colors = ["#E8834E" if k == best_k else "#7B68EE" for k in results_df["k"]]
    axes[1].bar(
        results_df["k"], results_df["silhouette"], color=colors, edgecolor="white"
    )
    axes[1].set_title(
        "Silhouette score vs K  (higher = better)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_xticks(results_df["k"][::2])  # every 2nd tick
    axes[1].annotate(
        f"Best K={best_k}",
        xy=(best_k, results_df.loc[results_df["k"] == best_k, "silhouette"].values[0]),
        xytext=(best_k - 3, results_df["silhouette"].max() * 0.97),  # shifted left
        fontsize=9,
        color="#E8834E",
        arrowprops=dict(arrowstyle="->", color="#E8834E"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_elbow_and_silhouette.png"), dpi=150)
    plt.show()
    print(f"\nBest K by silhouette score: {int(best_k)}")

def fit_final_kmeans(X_pca: np.ndarray, k: int) -> np.ndarray:
    """
    Fit the final K-Means model at the chosen K.
    Returns cluster label array of shape (n_profiles,).
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X_pca)
    print(f"✓ Final K-Means fitted  (K={k})")
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(
            f"    Cluster {cluster}: {count:,} profiles "
            f"({100 * count / len(labels):.1f}%)"
        )
    return labels


def profile_clusters(
    labels: np.ndarray, df_original: pd.DataFrame, skill_vocab: list
) -> pd.DataFrame:
    """
    For each cluster, compute:
    - top 5 skills
    - dominant industry
    - dominant role
    - median years experience
    - most common seniority level
    - size (number of profiles)

    Returns a summary dataframe — one row per cluster.
    This is what makes clusters interpretable and nameable.
    """
    import ast

    def safe_parse(val):
        if pd.isna(val):
            return []
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    df_working = df_original.copy()
    df_working["cluster"] = labels
    summaries = []

    for cluster_id in sorted(df_working["cluster"].unique()):
        subset = df_working[df_working["cluster"] == cluster_id]

        # Top 5 skills
        all_skills = []
        for val in subset["skills"]:
            parsed = safe_parse(val)
            if isinstance(parsed, list):
                all_skills.extend(
                    [s.strip().lower() for s in parsed if isinstance(s, str)]
                )
        from collections import Counter

        top_skills = [s for s, _ in Counter(all_skills).most_common(5)]

        summaries.append(
            {
                "cluster": cluster_id,
                "size": len(subset),
                "pct_of_total": round(100 * len(subset) / len(df_working), 1),
                "top_skills": ", ".join(top_skills),
                "dominant_industry": subset["industry"].mode()[0],
                "dominant_role": subset["current_role"].mode()[0],
                "median_exp": subset["years_experience"].median(),
                "top_seniority": subset["seniority_level"].mode()[0],
            }
        )

    summary_df = pd.DataFrame(summaries)
    return summary_df


def run_dbscan_experiment(
    X_pca: np.ndarray,
    eps_values: list = None,
    min_samples_values: list = None,
    n_components: int = 8,
) -> pd.DataFrame:
    """
    Run DBSCAN across a grid of eps and min_samples values.
    Uses cosine distance on a reduced PCA space for better density estimation.

    WHY: DBSCAN struggles in high dimensions because all pairwise distances
    become similar (curse of dimensionality). We reduce to n_components
    and use cosine distance, which works much better for sparse feature data.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed feature matrix.
    eps_values : list
        eps values to test (defaults to cosine-appropriate range).
    min_samples_values : list
        min_samples values to test.
    n_components : int
        Number of PCA components to use for DBSCAN (lower = better density).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if eps_values is None:
        eps_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    if min_samples_values is None:
        min_samples_values = [5, 10, 20, 50]

    # Reduce dimensions further for DBSCAN
    if X_pca.shape[1] > n_components:
        pca_reduce = PCA(n_components=n_components, random_state=42)
        X_dbscan = pca_reduce.fit_transform(X_pca)
        print(
            f"  Reduced to {n_components} components for DBSCAN "
            f"({pca_reduce.explained_variance_ratio_.sum() * 100:.1f}% variance)"
        )
    else:
        X_dbscan = X_pca.copy()

    # Normalize to unit vectors so cosine distance = euclidean distance
    X_dbscan = normalize(X_dbscan, norm="l2")

    results = []
    total = len(eps_values) * len(min_samples_values)
    count = 0

    print(f"Running DBSCAN grid search: {total} combinations...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            count += 1
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = db.fit_predict(X_dbscan)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_pct = 100 * n_noise / len(labels)

            sil = np.nan
            if n_clusters >= 2 and n_noise < len(labels) * 0.85:
                mask = labels != -1
                non_noise_labels = labels[mask]
                if len(set(non_noise_labels)) >= 2:
                    if mask.sum() > 5000:
                        rng = np.random.RandomState(42)
                        sub_idx = np.where(mask)[0]
                        sample_idx = rng.choice(sub_idx, size=5000, replace=False)
                        sample_labels = labels[sample_idx]
                        if len(set(sample_labels)) >= 2:
                            sil = silhouette_score(X_dbscan[sample_idx], sample_labels)
                    else:
                        sil = silhouette_score(X_dbscan[mask], non_noise_labels)

            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_pct": round(noise_pct, 1),
                    "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan,
                }
            )

            sil_str = f"sil={sil:.4f}" if not np.isnan(sil) else "sil=n/a"
            print(
                f"  [{count}/{total}] eps={eps:.2f} min_samples={min_samples:3d} | "
                f"clusters={n_clusters:3d} | noise={noise_pct:5.1f}% | {sil_str}"
            )

    return pd.DataFrame(results)


def fit_final_dbscan(
    X_pca: np.ndarray, eps: float, min_samples: int, n_components: int = 8
) -> np.ndarray:
    """
    Fit the final DBSCAN model with chosen parameters.
    Uses cosine distance on reduced PCA space.
    Returns cluster label array of shape (n_profiles,).
    Noise points are labeled -1.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    if X_pca.shape[1] > n_components:
        pca_reduce = PCA(n_components=n_components, random_state=42)
        X_dbscan = pca_reduce.fit_transform(X_pca)
    else:
        X_dbscan = X_pca.copy()

    X_dbscan = normalize(X_dbscan, norm="l2")

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X_dbscan)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✓ Final DBSCAN fitted  (eps={eps}, min_samples={min_samples})")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise:,} ({100 * n_noise / len(labels):.1f}%)")

    if n_clusters > 0:
        print(f"  Cluster sizes:")
        for cluster_id in sorted(set(labels) - {-1}):
            count = np.sum(labels == cluster_id)
            print(
                f"    Cluster {cluster_id}: {count:,} profiles "
                f"({100 * count / len(labels):.1f}%)"
            )

    return labels


def plot_dbscan_results(results_df: pd.DataFrame) -> None:
    """
    Plot DBSCAN grid search results as heatmaps for clusters, noise, and silhouette.
    """
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pivot_clusters = results_df.pivot_table(
        index="min_samples", columns="eps", values="n_clusters"
    )
    pivot_noise = results_df.pivot_table(
        index="min_samples", columns="eps", values="noise_pct"
    )
    pivot_sil = results_df.pivot_table(
        index="min_samples", columns="eps", values="silhouette"
    )

    sns.heatmap(pivot_clusters, cmap="Blues", annot=True, fmt=".0f", ax=axes[0])
    axes[0].set_title("Number of clusters", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("eps (cosine distance)")
    axes[0].set_ylabel("min_samples")

    sns.heatmap(pivot_noise, cmap="Reds", annot=True, fmt=".1f", ax=axes[1])
    axes[1].set_title("Noise percentage (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("eps (cosine distance)")
    axes[1].set_ylabel("min_samples")

    sns.heatmap(pivot_sil, cmap="Greens", annot=True, fmt=".3f", ax=axes[2])
    axes[2].set_title("Silhouette score", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("eps (cosine distance)")
    axes[2].set_ylabel("min_samples")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "11_dbscan_grid_search.png"))
    plt.show()

    best = results_df.dropna(subset=["silhouette"])
    if len(best) > 0:
        best_row = best.loc[best["silhouette"].idxmax()]
        print(f"\nBest DBSCAN by silhouette:")
        print(
            f"  eps={best_row['eps']}, min_samples={best_row['min_samples']} | "
            f"clusters={int(best_row['n_clusters'])} | "
            f"noise={best_row['noise_pct']}% | "
            f"silhouette={best_row['silhouette']}"
        )
    else:
        print("\n⚠ No DBSCAN configuration produced valid silhouette scores.")
        print("  This is expected for sparse high-dimensional data — DBSCAN is")
        print("  inherently less suited than K-Means for this dataset.")
