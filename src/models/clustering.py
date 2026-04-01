import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)

        inertia  = km.inertia_
        sil      = silhouette_score(X_pca, labels, sample_size=5000,
                                    random_state=42)

        results.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  K={k:2d} | inertia={inertia:,.0f} | silhouette={sil:.4f}")

    return pd.DataFrame(results)


def plot_elbow_and_silhouette(results_df: pd.DataFrame) -> None:
    """
    Plot inertia (elbow curve) and silhouette score side by side.
    Both plots saved to outputs/plots/.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Elbow curve
    axes[0].plot(results_df["k"], results_df["inertia"],
                 marker="o", color="#4A90D9", linewidth=2)
    axes[0].set_title("Elbow curve — inertia vs K",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(results_df["k"])

    # Silhouette scores
    best_k = results_df.loc[results_df["silhouette"].idxmax(), "k"]
    colors = ["#E8834E" if k == best_k else "#7B68EE"
              for k in results_df["k"]]
    axes[1].bar(results_df["k"], results_df["silhouette"],
                color=colors, edgecolor="white")
    axes[1].set_title("Silhouette score vs K  (higher = better)",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of clusters (K)")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_xticks(results_df["k"])
    axes[1].annotate(f"Best K={best_k}",
                     xy=(best_k, results_df.loc[
                         results_df["k"] == best_k, "silhouette"].values[0]),
                     xytext=(best_k + 0.3,
                             results_df["silhouette"].max() * 0.97),
                     fontsize=9, color="#E8834E")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_elbow_and_silhouette.png"))
    plt.show()
    print(f"\nBest K by silhouette score: {int(best_k)}")


def fit_final_kmeans(X_pca: np.ndarray, k: int) -> np.ndarray:
    """
    Fit the final K-Means model at the chosen K.
    Returns cluster label array of shape (n_profiles,).
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    print(f"✓ Final K-Means fitted  (K={k})")
    print(f"  Cluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"    Cluster {cluster}: {count:,} profiles "
              f"({100*count/len(labels):.1f}%)")
    return labels


def profile_clusters(labels: np.ndarray,
                     df_original: pd.DataFrame,
                     skill_vocab: list) -> pd.DataFrame:
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
                all_skills.extend([
                    s.strip().lower() for s in parsed
                    if isinstance(s, str)
                ])
        from collections import Counter
        top_skills = [s for s, _ in Counter(all_skills).most_common(5)]

        summaries.append({
            "cluster":         cluster_id,
            "size":            len(subset),
            "pct_of_total":    round(100 * len(subset) / len(df_working), 1),
            "top_skills":      ", ".join(top_skills),
            "dominant_industry": subset["industry"].mode()[0],
            "dominant_role":   subset["current_role"].mode()[0],
            "median_exp":      subset["years_experience"].median(),
            "top_seniority":   subset["seniority_level"].mode()[0],
        })

    summary_df = pd.DataFrame(summaries)
    return summary_df