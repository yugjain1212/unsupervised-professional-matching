import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def recommend_profiles(query_idx: int,
                       X: np.ndarray,
                       labels: np.ndarray,
                       df_original: pd.DataFrame,
                       top_n: int = 5,
                       normalize_features: bool = True) -> pd.DataFrame:
    """
    Given a query profile index, return the top_n most similar
    profiles from the same cluster, ranked by cosine similarity.

    IMPROVEMENTS MADE:
    1. Normalization to fix feature block domination (skills dominating)
    2. Cluster of size 1 fallback handling
    3. Better error handling

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
    labels      : cluster label array from fit_final_kmeans()
    df_original : full dataframe with all columns for display
    top_n       : number of recommendations to return
    normalize_features : if True, L2-normalize features before cosine similarity
                     This prevents feature blocks with more columns (e.g., skills=100)
                     from dominating the similarity calculation

    Returns
    -------
    DataFrame of top_n similar profiles with similarity score attached
    """
    query_cluster = labels[query_idx]

    # Get indices of all profiles in the same cluster
    cluster_indices = np.where(labels == query_cluster)[0]

    # Remove the query profile itself from candidates
    cluster_indices = cluster_indices[cluster_indices != query_idx]

    # Handle cluster of size 1 (only the query profile)
    if len(cluster_indices) == 0:
        # Fallback: find profiles in nearest cluster by centroid distance
        cluster_centroid = X[query_idx].mean()
        
        # Find cluster with most members (excluding query's cluster)
        other_clusters = [c for c in np.unique(labels) if c != query_cluster]
        if len(other_clusters) == 0:
            # No other clusters - return empty with message
            results = pd.DataFrame()
            results["_warning"] = ["No other profiles in cluster for recommendation"]
            return results
        
        # Find cluster with smallest average distance to query
        best_cluster = None
        best_distance = float('inf')
        
        for c in other_clusters:
            c_indices = np.where(labels == c)[0]
            if len(c_indices) == 0:
                continue
            c_centroid = X[c_indices].mean()
            distance = abs(cluster_centroid - c_centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = c
        
        if best_cluster is not None:
            # Use fallback cluster
            cluster_indices = np.where(labels == best_cluster)[0]
            if len(cluster_indices) > top_n:
                cluster_indices = cluster_indices[:top_n]
        else:
            # Return empty result
            results = pd.DataFrame()
            results["_warning"] = ["No suitable profiles found for recommendation"]
            return results

    # Get feature vectors
    query_vec = X[query_idx].reshape(1, -1)
    cluster_vecs = X[cluster_indices]

    # FIX: Normalize to prevent feature block domination
    # Without normalization, skills (100 cols × 2.0 weight = 200) dominates
    # L2 normalization ensures each feature contributes proportionally
    if normalize_features:
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1  # prevent division by zero
        query_vec_normalized = query_vec / query_norm
        
        # Normalize cluster vectors
        cluster_norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        cluster_norms[cluster_norms == 0] = 1
        cluster_vecs_normalized = cluster_vecs / cluster_norms
        
        # Compute cosine similarity on normalized vectors
        similarities = cosine_similarity(query_vec_normalized, cluster_vecs_normalized)[0]
    else:
        # Original behavior (without normalization)
        similarities = cosine_similarity(query_vec, cluster_vecs)[0]

    # Rank by similarity descending
    ranked_positions = np.argsort(similarities)[::-1][:top_n]
    top_indices = cluster_indices[ranked_positions]
    top_scores = similarities[ranked_positions]

    # Fetch full profile details from df_original
    results = df_original.iloc[top_indices].copy()
    results["similarity_score"] = top_scores.round(4)
    results["cluster"] = query_cluster

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


def validate_recommendations(query_idx: int,
                             X: np.ndarray,
                             labels: np.ndarray,
                             df_original: pd.DataFrame,
                             recommendations: pd.DataFrame) -> dict:
    """
    Validate recommendation quality manually without ground truth.
    
    Returns metrics that can be checked:
    - skill_overlap: How many skills match between query and recommendations
    - role_similarity: Same/different roles
    - industry_match: Same/different industries
    - experience_compatibility: Within 5 years
    """
    query = df_original.iloc[query_idx]
    
    query_skills = set(eval(query['skills'])) if isinstance(query['skills'], str) else set()
    
    validation = {
        'query_role': query['current_role'],
        'query_industry': query['industry'],
        'query_experience': query['years_experience'],
        'query_skills_count': len(query_skills),
        'recommendations': []
    }
    
    for idx, row in recommendations.iterrows():
        rec_skills = set(eval(row['skills'])) if isinstance(row['skills'], str) else set()
        
        skill_overlap = len(query_skills & rec_skills)
        role_match = 1 if row['current_role'] == query['current_role'] else 0
        industry_match = 1 if row['industry'] == query['industry'] else 0
        exp_diff = abs(float(row['years_experience']) - float(query['years_experience']))
        
        validation['recommendations'].append({
            'name': row['name'],
            'similarity': row['similarity_score'],
            'skill_overlap': skill_overlap,
            'role_match': role_match,
            'industry_match': industry_match,
            'experience_diff': exp_diff
        })
    
    return validation