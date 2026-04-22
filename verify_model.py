import os
import sys
import numpy as np
import pandas as pd
import ast

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import RAW_DATA_PATH, PROCESSED_PATH, PCA_COMPONENTS
from src.models.clustering import fit_final_kmeans
from src.models.compatibility import recommend_profiles, validate_recommendations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_check():
    print("Loading data...")
    if not os.path.exists(PROCESSED_PATH):
        print(f"Error: {PROCESSED_PATH} not found. Please run preprocessing first.")
        return

    X = np.load(PROCESSED_PATH)
    df_original = pd.read_csv(RAW_DATA_PATH)
    
    print(f"X shape: {X.shape}")
    print(f"Profiles: {len(df_original)}")

    # Preprocessing for clustering (consistent with notebook 05)
    print("Preparing PCA space...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Use a fixed K for demonstration (e.g., 15 as commonly found in EDA)
    # In a real scenario we'd use the best K from experiments
    K = 15 
    print(f"Fitting K-Means with K={K}...")
    kmeans_labels = fit_final_kmeans(X_pca, K)

    # Test Recommendation
    QUERY_IDX = 42
    print(f"\nTesting recommendation for profile index {QUERY_IDX}...")
    
    query_profile = df_original.iloc[QUERY_IDX]
    print(f"Query Name: {query_profile['name']}")
    print(f"Query Role: {query_profile['current_role']}")
    print(f"Query Skills: {query_profile['skills']}")

    recommendations = recommend_profiles(
        query_idx=QUERY_IDX,
        X=X,
        labels=kmeans_labels,
        df_original=df_original,
        top_n=5
    )

    print("\nRecommendations:")
    print(recommendations[['name', 'current_role', 'similarity_score', 'cluster']].to_string())

    # Validation
    print("\nValidating recommendations...")
    validation = validate_recommendations(QUERY_IDX, X, kmeans_labels, df_original, recommendations)
    
    print(f"Query Industry: {validation['query_industry']}")
    for i, rec in enumerate(validation['recommendations']):
        print(f"Rec {i+1}: {rec['name']} | Sim: {rec['similarity']:.4f} | Skill Overlap: {rec['skill_overlap']} | Role Match: {rec['role_match']} | Industry Match: {rec['industry_match']}")

if __name__ == "__main__":
    run_check()
