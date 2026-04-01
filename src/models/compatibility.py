import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_profiles(query_idx: int,
                       X: np.ndarray,
                       labels: np.ndarray,
                       df_original: pd.DataFrame,
                       top_n: int = 5) -> pd.DataFrame:
    """
    Given a query profile index, return the top_n most similar
    profiles from the same cluster, ranked by cosine similarity.

    WHY two stages:
    1. Cluster membership (from K-Means) narrows the search space
       from 50,000 profiles down to one cluster — coarse filtering.
    2. Cosine similarity within the cluster ranks the remaining
       profiles by directional similarity — fine-grained ranking.

    The model only sees row indices.
    Full profile details are fetched from df_original at the end —
    the model never had access to names, emails, or identifiers.

    Parameters
    ----------
    query_idx   : row index of the profile to find matches for
    X           : full feature matrix (50000, N) — NOT PCA-reduced
                  cosine similarity works best on the full feature space
    labels      : cluster label array from fit_final_kmeans()
    df_original : full dataframe with all columns for display
    top_n       : number of recommendations to return

    Returns
    -------
    DataFrame of top_n similar profiles with similarity score attached
    """
    query_cluster = labels[query_idx]

    # Get indices of all profiles in the same cluster
    cluster_indices = np.where(labels == query_cluster)[0]

    # Remove the query profile itself from candidates
    cluster_indices = cluster_indices[cluster_indices != query_idx]

    # Compute cosine similarity between query and all cluster members
    query_vec    = X[query_idx].reshape(1, -1)
    cluster_vecs = X[cluster_indices]
    similarities = cosine_similarity(query_vec, cluster_vecs)[0]

    # Rank by similarity descending
    ranked_positions = np.argsort(similarities)[::-1][:top_n]
    top_indices      = cluster_indices[ranked_positions]
    top_scores       = similarities[ranked_positions]

    # Fetch full profile details from df_original
    results = df_original.iloc[top_indices].copy()
    results["similarity_score"] = top_scores.round(4)
    results["cluster"]          = query_cluster

    # Drop PII that should not be displayed in a demo context
    # Keep name for display — this is the display layer, not the model
    display_cols = [
        "name", "current_role", "current_company", "industry",
        "seniority_level", "years_experience", "location",
        "remote_preference", "skills", "goals", "needs",
        "can_offer", "similarity_score", "cluster"
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    return results[display_cols].reset_index(drop=True)